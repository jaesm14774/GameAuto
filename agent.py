"""Vision Client + Game Agent — 視覺辨識與決策引擎"""

import base64
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

from knowledge import CharacterKnowledgeBase
from state_machine import StateMachine
from strategy import Action, GameStrategy, create_strategy

load_dotenv()
log = logging.getLogger("agent")


# ======================================================================
# Grid Overlay — 在截圖上繪製座標網格
# ======================================================================
class GridOverlay:
    """在截圖上疊加字母-數字座標網格，讓 LLM 能精確指定點擊位置"""

    def __init__(self, cols: int = 20, rows: int = 10):
        self.cols = cols
        self.rows = rows

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """在 frame 上繪製網格並標記座標，回傳標記後的副本"""
        h, w = frame.shape[:2]
        out = frame.copy()
        cw, ch = w / self.cols, h / self.rows

        for i in range(1, self.cols):
            x = int(i * cw)
            cv2.line(out, (x, 0), (x, h), (0, 255, 0), 1, cv2.LINE_AA)
        for j in range(1, self.rows):
            y = int(j * ch)
            cv2.line(out, (0, y), (w, y), (0, 255, 0), 1, cv2.LINE_AA)

        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.3, min(cw, ch) / 120)
        for i in range(self.cols):
            for j in range(self.rows):
                label = f"{chr(65 + i)}{j + 1}"
                tx = int(i * cw + 2)
                ty = int(j * ch + ch * 0.35)
                cv2.putText(out, label, (tx, ty), font, scale,
                            (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(out, label, (tx, ty), font, scale,
                            (0, 255, 0), 1, cv2.LINE_AA)
        return out

    def grid_to_pixel(self, grid_ref: str, frame_w: int, frame_h: int) -> Tuple[int, int]:
        """將網格座標 (如 "C7") 轉成像素中心座標"""
        grid_ref = grid_ref.strip().upper()
        col_char = grid_ref[0]
        row_num = int(grid_ref[1:])

        col = ord(col_char) - ord('A')
        row = row_num - 1

        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))

        cw, ch = frame_w / self.cols, frame_h / self.rows
        cx = int(col * cw + cw / 2)
        cy = int(row * ch + ch / 2)
        return cx, cy


# ======================================================================
# Screen Change Detection
# ======================================================================
class ScreenChangeDetector:
    """偵測畫面是否有顯著變化，避免重複呼叫 LLM"""

    def __init__(self, threshold: float = 0.02):
        self.threshold = threshold
        self._prev: Optional[np.ndarray] = None

    def changed(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (160, 90))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small

        if self._prev is None:
            self._prev = gray
            return True

        diff = cv2.absdiff(self._prev, gray)
        ratio = np.count_nonzero(diff > 30) / diff.size
        self._prev = gray
        return ratio > self.threshold

    def reset(self):
        self._prev = None


# ======================================================================
# Agent Memory
# ======================================================================
class AgentMemory:
    """儲存/載入遊戲經驗"""

    def __init__(self, path: str = "memory.json"):
        self.path = Path(path)
        self.history: List[Dict] = []
        self.patterns: Dict[str, Any] = {}
        self._max_history = 20
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                self.patterns = data.get("patterns", {})
                log.info("Memory loaded: %d patterns", len(self.patterns))
            except Exception as e:
                log.warning("Failed to load memory: %s", e)

    def save(self):
        data = {"patterns": self.patterns}
        self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                             encoding="utf-8")

    def add_action(self, phase: str, action: Dict, result: str = ""):
        self.history.append({
            "time": time.time(),
            "phase": phase,
            "action": action,
            "result": result,
        })
        if len(self.history) > self._max_history:
            self.history = self.history[-self._max_history:]

    def recent_summary(self, n: int = 5) -> str:
        if not self.history:
            return "No previous actions."
        lines = []
        for entry in self.history[-n:]:
            act = entry.get("action", {})
            desc = act.get("desc", act.get("type", "?"))
            lines.append(f"- {entry.get('phase', '?')}: {desc}")
        return "\n".join(lines)


# ======================================================================
# VisionClient — 純粹的 LLM 視覺辨識
# ======================================================================
class VisionClient:
    """
    純粹的 LLM 視覺辨識客戶端。
    只做「看」和「讀」，不做決策。
    """

    def __init__(self, config: Dict[str, Any]):
        vision_cfg = config.get("vision", config.get("agent", {}))

        self.api_key = os.getenv("OPENAI_API_KEY", "")
        self.model = os.getenv("LLM_MODEL", vision_cfg.get("model", "gemini-2.5-flash"))
        self.base_url = os.getenv("OPENAI_API_BASE",
                                  "https://generativelanguage.googleapis.com/v1beta/openai/")
        self._client = None

        self.grid = GridOverlay(
            cols=vision_cfg.get("grid_cols", 20),
            rows=vision_cfg.get("grid_rows", 10),
        )

        self._last_call = 0.0
        self._min_interval = vision_cfg.get("min_interval", 2.0)
        self._thinking_level = vision_cfg.get("thinking_level", None)

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI
            headers = {}
            if "openrouter.ai" in self.base_url:
                headers["HTTP-Referer"] = "https://github.com/GameAuto"
                headers["X-Title"] = "GameAuto"
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=headers or None,
            )

    def _encode_frame(self, frame: np.ndarray, annotate: bool = True) -> str:
        """將 frame 編碼為 base64 JPEG"""
        img = self.grid.annotate(frame) if annotate else frame
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])
        return base64.b64encode(buf).decode()

    def _can_call(self) -> bool:
        """檢查是否可以呼叫 LLM（頻率限制）"""
        now = time.time()
        if now - self._last_call < self._min_interval:
            return False
        if not self.api_key:
            log.warning("No API key configured")
            return False
        return True

    def _call_llm(self, prompt: str, frame: np.ndarray,
                  annotate: bool = True, bypass_interval: bool = False) -> Optional[str]:
        """呼叫 LLM，回傳原始文字回應"""
        if not bypass_interval and not self._can_call():
            return None
        if not self.api_key:
            log.warning("No API key configured")
            return None

        self._last_call = time.time()
        self._ensure_client()

        b64 = self._encode_frame(frame, annotate=annotate)

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                },
            ],
        }]

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 1024,
        }
        if self._thinking_level is not None:
            kwargs["reasoning_effort"] = self._thinking_level

        for attempt in range(3):
            try:
                log.info("LLM call (model=%s, attempt=%d)", self.model, attempt + 1)
                resp = self._client.chat.completions.create(**kwargs)
                content = resp.choices[0].message.content
                log.debug("LLM response: %s", content[:200])
                return content
            except Exception as e:
                wait = 5 * (attempt + 1)
                log.warning("LLM call failed (attempt %d): %s, retry in %ds",
                            attempt + 1, e, wait)
                time.sleep(wait)

        log.error("All LLM attempts failed")
        return None

    def detect_state(self, frame: np.ndarray, state_hints: Dict[str, str]) -> str:
        """
        偵測目前畫面屬於哪個遊戲狀態。

        Args:
            frame: 截圖
            state_hints: {state_name: description} 的對照表

        Returns:
            最匹配的 state name，或 "unknown"
        """
        hints_text = "\n".join(
            f'- "{name}": {desc}'
            for name, desc in state_hints.items()
            if name != "unknown"
        )

        prompt = f"""Analyze this game screenshot and determine which state it matches.

Possible states:
{hints_text}

Return ONLY a JSON object: {{"state": "state_name", "confidence": 0.0-1.0, "observations": ["what you see"]}}
No markdown, no explanation."""

        content = self._call_llm(prompt, frame)
        if content is None:
            return ""  # Empty = keep current state

        data = self._parse_json(content)
        if data:
            state = data.get("state", "unknown")
            confidence = data.get("confidence", 0)
            observations = data.get("observations", [])
            log.info("Detected state: %s (confidence=%.2f) obs=%s",
                     state, confidence, observations[:3])
            return state

        return ""  # Parse failed, keep current state

    def analyze(self, frame: np.ndarray, prompt: str,
                bypass_interval: bool = False) -> Optional[Dict]:
        """
        通用分析：看截圖，依 prompt 回傳結構化 JSON。
        用於聖物辨識、商店卡牌辨識等。
        """
        full_prompt = prompt + "\n\nReturn ONLY valid JSON. No markdown, no explanation."
        content = self._call_llm(full_prompt, frame, bypass_interval=bypass_interval)
        if content is None:
            return None
        return self._parse_json(content)

    def read_text(self, frame: np.ndarray, region_desc: str = "") -> Optional[str]:
        """讀取畫面上的文字"""
        prompt = f"Read and return all visible text on screen."
        if region_desc:
            prompt = f"Read the text in the {region_desc} area of the screen."
        prompt += '\n\nReturn JSON: {"text": "the text you read"}'

        content = self._call_llm(prompt, frame, annotate=False)
        if content is None:
            return None
        data = self._parse_json(content)
        return data.get("text") if data else None

    @staticmethod
    def _parse_json(content: str) -> Optional[Dict]:
        """從 LLM 回應中解析 JSON"""
        raw = content
        content = content.strip()

        # 移除 markdown code fence
        if "```" in content:
            lines = content.split("\n")
            lines = [line for line in lines
                     if not line.strip().startswith("```")]
            content = "\n".join(lines).strip()

        # 直接解析
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass

        # 嘗試從回應中擷取 JSON 物件
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(content[start:end])
            except json.JSONDecodeError:
                pass

        # 嘗試修復常見問題：尾端多餘逗號
        if start >= 0 and end > start:
            import re
            fixed = re.sub(r',\s*}', '}', content[start:end])
            fixed = re.sub(r',\s*]', ']', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        log.warning("Cannot parse LLM response as JSON: %s", raw[:300])
        return None


# ======================================================================
# GameAgent — 決策引擎
# ======================================================================
class GameAgent:
    """
    決策引擎：整合狀態機 + 規則引擎 + LLM 兜底。

    流程：
    1. 用 VisionClient 偵測目前狀態
    2. 用 GameStrategy (規則引擎) 產生動作
    3. 如果規則無法處理，用 VisionClient (LLM) 兜底
    """

    def __init__(self, config: Dict[str, Any]):
        game_cfg = config.get("game", {})
        vision_cfg = config.get("vision", config.get("agent", {}))
        strategy_cfg = config.get("strategy", {})

        self.vision = VisionClient(config)
        self.state_machine = StateMachine(config)
        self.strategy = create_strategy(game_cfg.get("name", ""), strategy_cfg)

        self.detector = ScreenChangeDetector(
            threshold=vision_cfg.get("change_threshold", 0.02),
        )

        memory_path = vision_cfg.get("memory_path",
                                     f"memory_{game_cfg.get('name', 'game')}.json")
        self.memory = AgentMemory(memory_path)

        # 角色知識庫
        kb_path = strategy_cfg.get("knowledge_base",
                                   f"characters_{game_cfg.get('name', 'game')}.json")
        self.knowledge = CharacterKnowledgeBase(kb_path)

        # 通知策略引擎是否有知識庫可用
        if hasattr(self.strategy, '_has_knowledge_base'):
            self.strategy._has_knowledge_base = not self.knowledge.is_empty

        self._last_tick_time = 0.0  # Track last meaningful tick for force-tick
        self._llm_prep_count = 0   # LLM 準備輪計數，用於 2:1 升人口模式

        # 戰鬥預分析快取
        self._pre_analysis: Optional[Dict] = None
        self._pre_analyzed = False

        # 庇護/敗場追蹤
        self._defeat_count = 0          # 已使用的庇護次數
        self._expect_protection = False  # 是否正在等待庇護彈窗
        self._should_stop = False        # 是否應該終止程式
        self._stop_reason = ""           # 終止原因

        self._force_llm_next_tick = False  # 點準備後強制下次 tick 呼叫 LLM

        self._system_prompt = self._build_system_prompt(game_cfg, strategy_cfg)

    def _build_system_prompt(self, game_cfg: Dict, strategy_cfg: Dict) -> str:
        game_name = game_cfg.get("name", "unknown game")
        game_desc = game_cfg.get("description", "")
        strategy_text = self.strategy.build_strategy_prompt()

        return f"""You are a game automation agent for "{game_name}". {game_desc}
You receive screenshots with a coordinate grid overlay (columns A-T, rows 1-10).

Your job:
1. Analyze the screenshot
2. Decide what actions to take
3. Return a JSON response

Response format (STRICT JSON, no markdown):
{{
  "phase": "current game phase name",
  "observations": ["what you see on screen"],
  "reasoning": "brief explanation",
  "actions": [
    {{"type": "click", "grid": "C7", "desc": "what you're clicking"}},
    {{"type": "wait", "seconds": 0.5}},
    {{"type": "skip", "desc": "reason to do nothing"}}
  ]
}}

Strategy:
{strategy_text}

Recent actions:
{{recent_actions}}

IMPORTANT:
- Be precise with grid coordinates.
- If unsure, use "skip". Don't click randomly.
- Return ONLY valid JSON."""

    # ------------------------------------------------------------------
    # 角色掃描（discover 模式）
    # ------------------------------------------------------------------
    def discover_characters(self, capture, controller):
        """
        掃描英雄圖鑑，逐一點擊每個英雄讀取詳情，建立角色知識庫。

        Args:
            capture: GameCapture 實例（截圖用）
            controller: GameController 實例（點擊/拖曳用）
        """
        log.info("=== 開始角色掃描 ===")

        # 暫存 controller 供 helper 方法使用
        self._discover_controller = controller
        positions = self.strategy._strategy_cfg.get("positions", {})

        # 1. 點擊英雄圖鑑按鈕
        hero_btn = positions.get("hero_menu_btn")
        if hero_btn and (hero_btn.get("x", 0) > 0 or hero_btn.get("y", 0) > 0):
            log.info("點擊英雄圖鑑按鈕 (%d, %d)", hero_btn["x"], hero_btn["y"])
            controller.click(hero_btn["x"], hero_btn["y"])
        else:
            # LLM 找按鈕
            frame = capture.grab()
            result = self.vision.analyze(
                frame,
                "Find the HERO COLLECTION / HERO ROSTER button on this main menu.\n"
                "This is the button to view ALL your owned heroes/characters and their details.\n"
                "It is usually a helmet/knight icon, or labeled '英雄' / '圖鑑' / '角色'.\n\n"
                "IMPORTANT: Do NOT click the shop/store (商城/商店) button. "
                "Do NOT click the battle (對戰) button. "
                "I need the hero INDEX / COLLECTION / ROSTER page.\n\n"
                'Return JSON: {"grid": "C5", "found": true, "label": "what the button says"}',
                bypass_interval=True,
            )
            if result and result.get("found") and result.get("grid"):
                fw, fh = frame.shape[1], frame.shape[0]
                x, y = self.vision.grid.grid_to_pixel(result["grid"], fw, fh)
                log.info("LLM 找到英雄圖鑑按鈕: grid=%s → (%d,%d)", result["grid"], x, y)
                controller.click(x, y)
            else:
                log.error("找不到英雄圖鑑按鈕，請在 config 中設定 hero_menu_btn")
                return
        time.sleep(2.0)

        # 2. 掃描英雄列表
        scanned_names: set = set()
        roster_grid = positions.get("hero_roster_grid", [])

        if roster_grid:
            # ===== 有預設座標：直接按格位掃描（不用 LLM 找圖標）=====
            total = sum(len(row) for row in roster_grid)
            log.info("使用預設座標掃描，共 %d 個格位 (%d 排)",
                     total, len(roster_grid))

            idx = 0
            for row_i, row in enumerate(roster_grid):
                for col_j, pos in enumerate(row):
                    idx += 1
                    x, y = pos["x"], pos["y"]
                    log.info("掃描英雄 [%d/%d] R%dC%d (%d,%d) ...",
                             idx, total, row_i, col_j, x, y)
                    controller.click(x, y)
                    time.sleep(1.5)

                    # 讀取英雄詳情
                    detail_frame = capture.grab()
                    char_data = self._scan_hero_detail(detail_frame)

                    if char_data and char_data.get("name"):
                        name = char_data["name"]
                        if name not in scanned_names:
                            self.knowledge.add(char_data)
                            scanned_names.add(name)
                            log.info("  → %s (role=%s, pos=%s, pri=%d)",
                                     name, char_data.get("role", "?"),
                                     char_data.get("position_pref", "?"),
                                     char_data.get("priority", 50))
                        else:
                            log.debug("  → %s 已掃描，跳過", name)
                    else:
                        log.warning("  → 格位 R%dC%d 無法讀取（可能是空格位）",
                                    row_i, col_j)

                    # 返回列表
                    self._click_back_btn(positions, detail_frame, capture)
                    time.sleep(1.0)
        else:
            # ===== 無預設座標：用 LLM 找英雄圖標 =====
            log.info("無 hero_roster_grid，用 LLM 偵測英雄圖標位置")
            max_scrolls = 5

            for scroll_round in range(max_scrolls + 1):
                frame = capture.grab()

                roster_data = self.vision.analyze(
                    frame,
                    "You see a hero roster/collection screen.\n"
                    "List ALL hero icons visible. For each, give grid position.\n"
                    "Return ONLY JSON:\n"
                    '{"heroes": [{"name": "", "grid": "C5"}, ...],'
                    ' "can_scroll_down": true}',
                    bypass_interval=True,
                )

                if not roster_data or not roster_data.get("heroes"):
                    log.warning("第 %d 頁：無法識別英雄列表", scroll_round + 1)
                    break

                heroes = roster_data["heroes"]
                log.info("第 %d 頁：發現 %d 個英雄圖標",
                         scroll_round + 1, len(heroes))

                fw, fh = frame.shape[1], frame.shape[0]
                for i, hero in enumerate(heroes):
                    grid = hero.get("grid", "")
                    if not grid:
                        continue

                    hx, hy = self.vision.grid.grid_to_pixel(grid, fw, fh)
                    log.info("點擊英雄 #%d (grid=%s) ...", i + 1, grid)
                    controller.click(hx, hy)
                    time.sleep(1.5)

                    detail_frame = capture.grab()
                    char_data = self._scan_hero_detail(detail_frame)

                    if char_data and char_data.get("name"):
                        name = char_data["name"]
                        if name not in scanned_names:
                            self.knowledge.add(char_data)
                            scanned_names.add(name)
                            log.info("  → %s (role=%s, pos=%s, pri=%d)",
                                     name, char_data.get("role", "?"),
                                     char_data.get("position_pref", "?"),
                                     char_data.get("priority", 50))
                    else:
                        log.warning("  → 無法讀取英雄詳情")

                    self._click_back_btn(positions, detail_frame, capture)
                    time.sleep(1.0)

                can_scroll = roster_data.get("can_scroll_down", False)
                if not can_scroll:
                    break

                mid_x = fw // 2
                controller.drag(mid_x, int(fh * 0.7), mid_x, int(fh * 0.3))
                time.sleep(1.0)

        # 5. 儲存知識庫
        self.knowledge.save()
        log.info("=== 角色掃描完成：共掃描 %d 個新角色，知識庫共 %d 個 ===",
                 len(scanned_names), len(self.knowledge))

        # 更新策略引擎的知識庫旗標
        if hasattr(self.strategy, '_has_knowledge_base'):
            self.strategy._has_knowledge_base = not self.knowledge.is_empty

        # 6. 返回主選單
        frame = capture.grab()
        self._click_back_btn(positions, frame, capture)
        time.sleep(1.0)

        # 清除暫存
        self._discover_controller = None

    def _click_back_btn(self, positions: Dict, detail_frame: np.ndarray,
                        capture) -> None:
        """點擊返回按鈕（優先用座標，fallback 用 LLM 找）"""
        back_btn = positions.get("hero_back_btn")
        if back_btn and (back_btn.get("x", 0) > 0 or back_btn.get("y", 0) > 0):
            self._discover_controller.click(back_btn["x"], back_btn["y"])
            return

        back_result = self.vision.analyze(
            detail_frame,
            "Find the back/return/close button on this screen. "
            'Return JSON: {"grid": "A1", "found": true}',
            bypass_interval=True,
        )
        if back_result and back_result.get("found") and back_result.get("grid"):
            fw, fh = detail_frame.shape[1], detail_frame.shape[0]
            bx, by = self.vision.grid.grid_to_pixel(
                back_result["grid"], fw, fh,
            )
            self._discover_controller.click(bx, by)
        else:
            self._discover_controller.click(50, 50)

    def _scan_hero_detail(self, frame: np.ndarray) -> Optional[Dict]:
        """
        用 LLM 讀取英雄詳情頁面，回傳結構化角色資訊。

        LLM 負責：
        - 讀取名稱、職業標籤、技能描述
        - 根據技能判斷戰鬥角色 (tank/mage/assassin/...)
        - 根據角色判斷偏好站位 (front/mid/back/back_corner)
        - 評估優先度 (1-100)
        """
        prompt = (
            "Read this hero detail screen. Extract the character info.\n\n"
            "role must be one of: tank, warrior, melee_dps, ranged_dps, mage, support, healer, assassin, summoner\n"
            "position_pref must be one of: front, mid, back, back_corner\n"
            "priority: 1-100 (lower=stronger, AoE damage=10-20, tank=15-25, support=20-30, average=40-60, weak=70-90)\n\n"
            "Reply with ONLY this JSON, nothing else:\n"
            '{"name":"角色名","role":"mage","classes":["法師"],"abilities":["技能名"],'
            '"ability_desc":"what abilities do","position_pref":"back","priority":20,"notes":"strengths"}'
        )

        content = self.vision._call_llm(prompt, frame,
                                         annotate=False, bypass_interval=True)
        if content is None:
            return None

        result = VisionClient._parse_json(content)
        if result and result.get("name"):
            return result

        # 第一次失敗 → 用更簡短的 prompt 重試
        log.info("英雄詳情解析失敗，用簡化 prompt 重試")
        retry_prompt = (
            "What is the name of this character? What does it do?\n"
            "Reply ONLY with JSON: "
            '{"name":"名字","role":"tank or mage or support or warrior or assassin or ranged_dps or healer or melee_dps or summoner",'
            '"position_pref":"front or mid or back","priority":50,"classes":[],"abilities":[],'
            '"ability_desc":"","notes":""}'
        )
        content2 = self.vision._call_llm(retry_prompt, frame,
                                          annotate=False, bypass_interval=True)
        if content2 is None:
            return None
        return VisionClient._parse_json(content2)

    # ------------------------------------------------------------------
    # 智慧商店 + 棋盤讀取（合併 LLM call）
    # ------------------------------------------------------------------
    def _llm_read_shop_and_board(self, frame: np.ndarray) -> Optional[Dict]:
        """
        一次 LLM call 同時讀取：商店卡片名稱 + 棋盤角色位置 + 金幣 + 人口。
        取代分開呼叫讀金幣/讀棋盤，減少延遲。
        """
        prompt = (
            "Analyze this auto-chess game screenshot during preparation phase.\n"
            "Read ALL of the following:\n\n"
            "1. SHOP CARDS at the bottom: list each card's name\n"
            "2. UNITS on the player's board (bottom half of the chess board):\n"
            "   - Row 0 = front (closest to enemy), Row 3 = back (closest to shop)\n"
            "   - Col 0 = leftmost, Col 4 = rightmost\n"
            "3. GOLD amount (the coin number)\n"
            "4. POPULATION: find the helmet icon showing 'X/Y' (current/max units)\n\n"
            "Return ONLY valid JSON:\n"
            "{\n"
            '  "shop": [{"name": "card_name", "slot": 0}, ...],\n'
            '  "board": [{"name": "unit_name", "row": 0, "col": 2}, ...],\n'
            '  "gold": 5,\n'
            '  "current_pop": 3,\n'
            '  "max_pop": 5\n'
            "}"
        )

        content = self.vision._call_llm(prompt, frame,
                                         annotate=False, bypass_interval=True)
        if content is None:
            return None
        return VisionClient._parse_json(content)

    # ------------------------------------------------------------------
    # 智慧購買
    # ------------------------------------------------------------------
    def _smart_buy_cards(self, shop_data: List[Dict],
                         frame: np.ndarray) -> List[Action]:
        """
        根據知識庫決定購買哪些商店卡片。

        Args:
            shop_data: [{"name": "角色名", "slot": 0}, ...]
            frame: 當前截圖（用於取得卡片座標）

        Returns:
            點擊要購買的卡片的動作列表
        """
        positions = self.strategy._strategy_cfg.get("positions", {})
        card_slots = positions.get("card_slots", [])

        if not card_slots or not shop_data:
            return []

        # 用知識庫排序
        card_names = [c.get("name", "") for c in shop_data]
        ranked = self.knowledge.rank_shop_cards(card_names)

        actions: List[Action] = []
        for name, priority, should_buy in ranked:
            if not should_buy:
                log.info("跳過不需要的卡: %s (priority=%d)", name, priority)
                continue

            # 找到對應的商店欄位
            for card in shop_data:
                if card.get("name") == name:
                    slot_idx = card.get("slot", 0)
                    if slot_idx < len(card_slots):
                        slot = card_slots[slot_idx]
                        actions.append(Action(
                            type="click", x=slot["x"], y=slot["y"],
                            desc=f"買 {name} (pri={priority})",
                        ))
                    break

        return actions

    # ------------------------------------------------------------------
    # 戰鬥預分析
    # ------------------------------------------------------------------
    def _pre_analyze_board(self, frame: np.ndarray):
        """
        在戰鬥階段預先分析棋盤，快取結果供下次準備階段使用。
        隱藏 LLM 延遲。
        """
        if self._pre_analyzed:
            return

        log.info("戰鬥中預分析棋盤...")
        result = self._llm_read_shop_and_board(frame)
        if result and result.get("board"):
            self._pre_analysis = result
            self._pre_analyzed = True
            log.info("預分析完成: %d 個棋盤角色",
                     len(result.get("board", [])))

    # ------------------------------------------------------------------
    # 聖物選擇
    # ------------------------------------------------------------------
    def _llm_select_relic(self, frame: np.ndarray) -> List[Action]:
        """
        LLM 判讀聖物選擇畫面，依照優先清單選擇最佳聖物。
        如果沒有優先聖物，不點擊（讓遊戲自動選推薦的）。
        """
        prompt = (
            "This is a relic/artifact selection screen in an auto-chess game.\n"
            "Read the NAME of every relic/artifact shown on screen.\n"
            "For each relic, also note its grid position so I can click it.\n\n"
            "Reply ONLY with JSON:\n"
            '{"relics": [{"name": "聖物名稱", "grid": "C5"}, ...]}'
        )

        data = self.vision.analyze(frame, prompt, bypass_interval=True)
        if not data or not data.get("relics"):
            log.warning("無法讀取聖物，跳過選擇（讓遊戲自動推薦）")
            return [Action(type="wait", seconds=3, desc="聖物判讀失敗，等待自動推薦")]

        relics = data["relics"]
        relic_names = [r.get("name", "") for r in relics]
        log.info("偵測到聖物: %s", relic_names)

        # 用 RuleEngine 的優先序排序
        best_relic = None
        best_priority = 999
        best_grid = ""

        for relic in relics:
            name = relic.get("name", "")
            grid = relic.get("grid", "")
            if not name or not grid:
                continue
            priority = self.strategy.rules.get_relic_priority(name)
            if priority < best_priority:
                best_priority = priority
                best_relic = name
                best_grid = grid

        # 只有在優先清單內的聖物才點選（priority < 99 = 在清單內）
        if best_relic and best_priority < 99:
            log.info("選擇聖物: %s (優先序=%d, grid=%s)", best_relic, best_priority, best_grid)
            fw, fh = frame.shape[1], frame.shape[0]
            x, y = self.vision.grid.grid_to_pixel(best_grid, fw, fh)
            return [Action(type="click", x=x, y=y, grid=best_grid,
                           desc=f"選擇聖物: {best_relic}")]
        else:
            log.info("無優先聖物，不點擊（讓遊戲自動推薦）")
            return [Action(type="wait", seconds=3, desc="無優先聖物，等待自動推薦")]

    @property
    def current_phase(self) -> str:
        return self.state_machine.current_state

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    # States where we skip LLM detection and just execute strategy actions
    _FAST_ACTION_STATES = {"preparation"}
    # How often to call LLM for state verification in fast-action states (seconds)
    _FAST_STATE_LLM_INTERVAL = 15.0
    # Force a tick if idle for this many seconds (handles animation gaps / screen not changing)
    _FORCE_CHECK_INTERVAL = 5.0

    def tick(self, frame: np.ndarray) -> List[Action]:
        """
        主要決策方法：

        1. 檢查畫面是否變化（或閒置過久則強制觸發）
        2. 在準備階段跳過 LLM，直接執行策略動作（省時間）
        3. 其他狀態用 LLM 偵測狀態
        4. 規則引擎產生動作
        5. 如果需要 LLM 兜底，呼叫 LLM
        6. 回傳動作列表
        """
        screen_changed = self.detector.changed(frame)
        now = time.time()
        current = self.state_machine.current_state

        # Force a tick if idle for too long (fixes gaps after relic select, etc.)
        time_since_tick = now - self._last_tick_time
        force_tick = time_since_tick > self._FORCE_CHECK_INTERVAL

        if not screen_changed and not force_tick:
            return []

        # Decide whether to call LLM for state detection
        time_since_llm = now - self.vision._last_call
        skip_llm = (
            current in self._FAST_ACTION_STATES
            and time_since_llm < self._FAST_STATE_LLM_INTERVAL
            and not force_tick
            and not self._force_llm_next_tick
        )

        if not skip_llm:
            state_hints = self.state_machine.get_detect_hints()
            detected = self.vision.detect_state(frame, state_hints)
            self.state_machine.tick(detected)
            # 只有在狀態確實離開 preparation 後，才清除強制偵測旗標
            new_state = self.state_machine.current_state
            if self._force_llm_next_tick and new_state != "preparation":
                self._force_llm_next_tick = False
                log.info("準備→%s 轉場偵測成功，恢復正常節流", new_state)
        else:
            log.debug("Fast tick: skipping LLM in %s state", current)

        current = self.state_machine.current_state

        # ----------------------------------------------------------
        # 庇護追蹤邏輯
        # ----------------------------------------------------------
        # 進入失敗結算 → 標記等待庇護
        if current == "match_result_defeat":
            self._expect_protection = True

        # 進入庇護彈窗 → 清除等待標記，計數+1
        if current == "protection_popup":
            self._expect_protection = False
            self._defeat_count += 1
            log.info("庇護 #%d — 接受庇護中", self._defeat_count)
            if self._defeat_count >= 10:
                log.warning("已使用 10 次庇護，接受本次後將終止程式")

        # 失敗結算後直接回到主選單（沒有庇護彈窗）→ 庇護已用完，終止
        if current == "main_menu" and self._expect_protection:
            self._expect_protection = False
            self._should_stop = True
            self._stop_reason = (
                f"對戰失敗但庇護未出現（已使用 {self._defeat_count} 次庇護）。"
                "庇護次數可能已用完，為避免扣分，程式終止。"
            )
            log.error(self._stop_reason)

        # 第 10 次庇護接受後，回到主選單時終止
        if current == "main_menu" and self._defeat_count >= 10:
            self._should_stop = True
            self._stop_reason = (
                f"今日已使用完 10 次庇護（共 {self._defeat_count} 次敗場），"
                "為避免扣分，程式終止。"
            )
            log.error(self._stop_reason)

        # ----------------------------------------------------------

        # 離開準備階段時重置 LLM 輪計數
        if current != "preparation":
            self._llm_prep_count = 0
            # 進入新的準備階段時清除預分析快取
            if current == "battle":
                self._pre_analyzed = False

        # Build context
        context: Dict[str, Any] = {
            "timeout": self.state_machine.is_timed_out(),
            "time_in_state": self.state_machine.time_in_state,
        }

        # Try rules first
        actions = self.strategy.on_state(current, context)

        # Process actions — resolve LLM fallbacks
        resolved: List[Action] = []
        for action in actions:
            if action.type == "llm_position":
                # LLM 偵測角色位置，生成拖曳指令調整陣型
                pos_actions = self._llm_position_units(frame)
                resolved.extend(pos_actions)
            elif action.type == "llm_smart_prep":
                # 智慧準備：合併讀取商店+棋盤+金幣，知識庫決策
                smart_actions = self._handle_smart_preparation(frame)
                resolved.extend(smart_actions)
            elif action.type == "llm_decide_prep":
                # LLM 只決定 buy 或 ready，然後用固定座標執行
                prep_actions = self._llm_decide_preparation(frame, action.desc)
                resolved.extend(prep_actions)
            elif action.type == "llm_relic_select":
                # LLM 判讀聖物，選最佳聖物
                relic_actions = self._llm_select_relic(frame)
                resolved.extend(relic_actions)
            elif action.type == "llm_pre_analyze":
                # 戰鬥中預分析棋盤
                self._pre_analyze_board(frame)
                # 不產生動作
            elif action.type == "llm_fallback":
                llm_actions = self._llm_decide(frame, action.desc)
                resolved.extend(llm_actions)
            elif action.type == "click" and not action.grid and not action.x:
                # Need LLM to find the click target on screen
                located = self._llm_locate_target(frame, action.target)
                if located:
                    action.grid = located.get("grid", "")
                    if action.grid:
                        fw, fh = frame.shape[1], frame.shape[0]
                        action.x, action.y = self.vision.grid.grid_to_pixel(
                            action.grid, fw, fh,
                        )
                    resolved.append(action)
                else:
                    log.warning("Cannot locate target: %s", action.target)
            else:
                # Resolve grid to pixel if needed
                if action.grid and not action.x:
                    fw, fh = frame.shape[1], frame.shape[0]
                    action.x, action.y = self.vision.grid.grid_to_pixel(
                        action.grid, fw, fh,
                    )
                resolved.append(action)

        # Record actions
        for action in resolved:
            self.memory.add_action(
                current,
                {"type": action.type, "grid": action.grid, "desc": action.desc},
            )

        if resolved:
            self._last_tick_time = now
            log.info("Actions: %s",
                     " -> ".join(f"{a.type}({a.grid or a.target or a.desc})"
                                 for a in resolved))

        # 點準備後強制下次 tick 呼叫 LLM（涵蓋策略層的超時按準備路徑）
        if current == "preparation" and resolved:
            ready_pos = self.strategy._strategy_cfg.get("positions", {}).get("ready_btn")
            if ready_pos:
                for a in resolved:
                    if (a.type == "click"
                            and a.x == ready_pos["x"] and a.y == ready_pos["y"]):
                        self._force_llm_next_tick = True
                        break

        # 選完聖物後直接進入準備階段，省掉一次 LLM 偵測
        if current == "relic_select" and resolved:
            self.state_machine.transition_to("preparation")

        return resolved

    # ------------------------------------------------------------------
    # 智慧準備階段（知識庫驅動）
    # ------------------------------------------------------------------
    def _handle_smart_preparation(self, frame: np.ndarray) -> List[Action]:
        """
        智慧準備階段：
        1. 一次 LLM call 讀取商店+棋盤+金幣+人口
        2. 知識庫決定買哪些卡
        3. 知識庫決定站位
        4. 決定是否升人口

        如果有戰鬥預分析快取，用快取的棋盤資料（省一次 LLM call）。
        """
        self._llm_prep_count += 1
        positions = self.strategy._strategy_cfg.get("positions", {})
        card_slots = positions.get("card_slots", [])

        # 嘗試用預分析快取
        data = None
        if self._pre_analysis and self._pre_analysis.get("board"):
            log.info("使用戰鬥預分析快取")
            # 預分析有棋盤資料，但商店可能已變 → 仍需讀商店
            # 只用棋盤資料，商店重新讀
            data = self._llm_read_shop_and_board(frame)
            if data and self._pre_analysis.get("board"):
                # 合併：用新的商店+金幣，但棋盤用快取
                # 不，其實準備階段棋盤也可能變了（買了新卡），還是用新的
                pass
            self._pre_analysis = None

        if data is None:
            data = self._llm_read_shop_and_board(frame)

        if data is None:
            log.warning("智慧準備：LLM 讀取失敗，fallback 到按準備")
            return self._build_ready_actions()

        gold = data.get("gold", 0)
        current_pop = data.get("current_pop", 0)
        max_pop = data.get("max_pop", 0)
        shop = data.get("shop", [])
        board = data.get("board", [])

        log.info("智慧準備: gold=%d, pop=%d/%d, shop=%d張, board=%d角色",
                 gold, current_pop, max_pop, len(shop), len(board))

        # 金幣 = 0 → 直接準備
        if gold <= 0:
            log.info("金幣=0，點擊準備")
            return self._build_ready_actions()

        actions: List[Action] = []

        # --- 升人口判斷 ---
        need_upgrade = False
        if max_pop >= 1 and current_pop >= max_pop:
            if max_pop < 6:
                need_upgrade = True
                log.info("人口已滿 (%d/%d, <6)，升人口", current_pop, max_pop)
            elif random.random() < 0.3:
                need_upgrade = True
                log.info("人口 %d/%d (>=6)，30%%機率升人口", current_pop, max_pop)
        if need_upgrade:
            pop_pos = positions.get("population_btn")
            if pop_pos:
                actions.append(Action(type="click", x=pop_pos["x"], y=pop_pos["y"],
                                      desc="升人口"))

        # --- 智慧購買 ---
        if shop and not self.knowledge.is_empty:
            buy_actions = self._smart_buy_cards(shop, frame)
            actions.extend(buy_actions)
        elif card_slots:
            # 知識庫為空，退化為全買
            for i, slot in enumerate(card_slots):
                actions.append(Action(type="click", x=slot["x"], y=slot["y"],
                                      desc=f"買卡{i+1}"))

        # --- 刷新再買（把剩餘金幣花完） ---
        if gold >= 2:
            refresh_pos = positions.get("refresh_btn")
            if refresh_pos:
                actions.append(Action(type="click", x=refresh_pos["x"],
                                      y=refresh_pos["y"], desc="刷新商店"))
            for i, slot in enumerate(card_slots):
                actions.append(Action(type="click", x=slot["x"], y=slot["y"],
                                      desc=f"買卡{i+1}"))

        # 點秒數框防卡
        safe_pos = positions.get("safe_click")
        if safe_pos:
            actions.append(Action(type="click", x=safe_pos["x"], y=safe_pos["y"],
                                  desc="點擊秒數框"))

        # 買完就按準備（不等下一輪 LLM 確認 gold=0）
        actions.extend(self._build_ready_actions())

        return actions

    def _llm_decide_preparation(self, frame: np.ndarray, prompt: str) -> List[Action]:
        """
        LLM 讀金幣和人口 (current/max)，程式決定動作。
        gold=0 → 直接準備
        LLM 能判斷人口 → 根據 current/max 決定升不升
        LLM 判斷不了 → 用 2:1 模式（2 次刷新配 1 次升人口）
        """
        self._llm_prep_count += 1

        content = self.vision._call_llm(prompt, frame, bypass_interval=True)
        if content is None:
            log.warning("LLM prep decision failed, clicking Ready as fallback")
            return self._build_ready_actions()

        data = VisionClient._parse_json(content)
        if data is None:
            log.warning("LLM prep response not parseable, clicking Ready as fallback")
            return self._build_ready_actions()

        gold = data.get("gold", 0)
        current_pop = data.get("current_pop", 0)
        max_pop = data.get("max_pop", 0)
        log.info("LLM read: gold=%s, pop=%s/%s", gold, current_pop, max_pop)

        # 金幣 = 0 → 直接按準備
        if gold <= 0:
            log.info("Gold = 0, clicking Ready")
            return self._build_ready_actions()

        positions = self.strategy._strategy_cfg.get("positions", {})
        need_upgrade = False

        if max_pop >= 1:
            # LLM 成功讀到人口
            if current_pop >= max_pop and max_pop < 6:
                # 角色已滿，上限 < 6 → 升人口
                log.info("Board full (%d/%d, <6), upgrading", current_pop, max_pop)
                need_upgrade = True
            elif current_pop >= max_pop and max_pop >= 6:
                # max_pop >= 6，30% 機率仍升人口
                if random.random() < 0.3:
                    log.info("Pop %d/%d (>=6), 30%% chance: upgrading", current_pop, max_pop)
                    need_upgrade = True
                else:
                    log.info("Pop %d/%d (>=6), 70%% chance: just buying", current_pop, max_pop)
            elif current_pop < max_pop:
                # 還有空位 → 不升，只買卡填角色
                log.info("Slots available (%d/%d), just buying", current_pop, max_pop)
        else:
            # LLM 沒讀到人口 → 2:1 模式（每 3 輪升 1 次）
            if self._llm_prep_count % 3 == 0:
                log.info("Pop unknown, fallback pattern: upgrading this round")
                need_upgrade = True
            else:
                log.info("Pop unknown, fallback pattern: refreshing this round")

        actions: List[Action] = []
        if need_upgrade:
            pop_pos = positions.get("population_btn")
            if pop_pos:
                actions.append(Action(type="click", x=pop_pos["x"], y=pop_pos["y"],
                                      desc="升人口"))
            # 升人口後直接買卡（不刷新）
            actions.extend(self._build_buy_only())
        else:
            # 刷新+買卡
            actions.extend(self._build_buy_round())

        # 買完就按準備（不等下一輪 LLM 確認 gold=0）
        actions.extend(self._build_ready_actions())

        return actions

    def _build_buy_only(self) -> List[Action]:
        """只買卡左中右（不刷新，升人口後用）"""
        positions = self.strategy._strategy_cfg.get("positions", {})
        card_slots = positions.get("card_slots", [])
        actions: List[Action] = []
        for i, slot in enumerate(card_slots):
            actions.append(Action(type="click", x=slot["x"], y=slot["y"],
                                  desc=f"買卡{i+1}"))
        safe_pos = positions.get("safe_click")
        if safe_pos:
            actions.append(Action(type="click", x=safe_pos["x"], y=safe_pos["y"],
                                  desc="點擊秒數框"))
        return actions

    def _build_buy_round(self) -> List[Action]:
        """一輪標準操作：刷新 → 買左中右 → 點秒數框"""
        positions = self.strategy._strategy_cfg.get("positions", {})
        card_slots = positions.get("card_slots", [])
        actions: List[Action] = []

        refresh_pos = positions.get("refresh_btn")
        if refresh_pos:
            actions.append(Action(type="click", x=refresh_pos["x"], y=refresh_pos["y"],
                                  desc="刷新商店"))
        for i, slot in enumerate(card_slots):
            actions.append(Action(type="click", x=slot["x"], y=slot["y"],
                                  desc=f"買卡{i+1}"))

        safe_pos = positions.get("safe_click")
        if safe_pos:
            actions.append(Action(type="click", x=safe_pos["x"], y=safe_pos["y"],
                                  desc="點擊秒數框"))

        return actions

    def _build_ready_actions(self) -> List[Action]:
        """用固定座標點擊準備按鈕，並設定旗標強制下次 tick 呼叫 LLM 偵測狀態"""
        positions = self.strategy._strategy_cfg.get("positions", {})
        ready_pos = positions.get("ready_btn")
        if ready_pos:
            # 點完準備後，強制下次 tick 呼叫 LLM 偵測狀態
            # 避免 FAST_ACTION_STATES 節流導致 preparation→battle 轉場漏偵測
            self._force_llm_next_tick = True
            return [Action(type="click", x=ready_pos["x"], y=ready_pos["y"],
                           desc="點擊準備")]
        log.error("No ready_btn position configured!")
        return []

    # ------------------------------------------------------------------
    # 陣型調整
    # ------------------------------------------------------------------
    def _llm_position_units(self, frame: np.ndarray) -> List[Action]:
        """
        用 LLM 偵測棋盤上角色位置，產生拖曳指令調整陣型。

        修正：
        - 使用 annotate=False，不疊加 A1-T10 網格（會混淆 LLM）
        - 驗證 row/col 範圍
        - 後排角色優先搬動，避免前排衝突
        """
        board_grid = self.strategy._strategy_cfg.get("positions", {}).get("board_grid", [])
        if not board_grid:
            log.warning("board_grid not configured, skipping positioning")
            return []

        prompt = (
            "This is an auto-chess game board screenshot.\n"
            "The board is split: enemy on top, player on bottom.\n\n"
            "Focus ONLY on the PLAYER's area (bottom half of the board).\n"
            "The player area is a grid with 4 rows and 5 columns:\n"
            "  - Row 0 = front row (the row closest to the center/enemy)\n"
            "  - Row 1, 2 = middle rows\n"
            "  - Row 3 = back row (closest to the bottom edge / shop)\n"
            "  - Col 0 = leftmost column\n"
            "  - Col 4 = rightmost column\n\n"
            "List every unit you see on the player's side.\n"
            "For each unit, give its name and grid position.\n\n"
            "Also: is there a unit called '勝光天使' "
            "(Victory Light Angel — a large bright angel/winged character)?\n\n"
            "Return ONLY valid JSON:\n"
            '{"units": [{"name": "unit_name", "row": 0, "col": 2}], '
            '"has_victory_angel": false}'
        )

        # 不疊加 A1-T10 網格，避免與棋盤 4×5 座標混淆
        content = self.vision._call_llm(prompt, frame,
                                        annotate=False, bypass_interval=True)
        if content is None:
            log.warning("LLM position detection failed, skipping")
            return []

        data = VisionClient._parse_json(content)
        if data is None:
            log.warning("LLM position response not parseable, skipping")
            return []

        units = data.get("units", [])
        has_angel = data.get("has_victory_angel", False)

        if not units:
            log.info("No units detected on board, skipping positioning")
            return []

        # 驗證並 clamp row/col 到合法範圍
        for unit in units:
            unit["row"] = max(0, min(3, int(unit.get("row", 0))))
            unit["col"] = max(0, min(4, int(unit.get("col", 0))))

        log.info("Detected %d units at %s, 勝光天使=%s",
                 len(units),
                 [(u.get("name", "?"), u["row"], u["col"]) for u in units],
                 has_angel)

        # 計算目標位置
        # 優先使用知識庫（如果有角色資料）
        if not self.knowledge.is_empty:
            # 檢查知識庫是否認識至少一個角色
            known_count = sum(1 for u in units if self.knowledge.get(u.get("name", "")))
            if known_count > 0:
                log.info("使用知識庫站位 (%d/%d 角色已知)", known_count, len(units))
                targets = self.knowledge.compute_formation(units)
            elif has_angel:
                targets = self._formation_angel_column(units)
            else:
                targets = self._formation_front_row(units)
        elif has_angel:
            targets = self._formation_angel_column(units)
        else:
            targets = self._formation_front_row(units)

        # 生成拖曳指令（只移動需要移動的）
        actions: List[Action] = []
        for unit_info, (target_row, target_col) in targets:
            src_row = unit_info["row"]
            src_col = unit_info["col"]
            if src_row == target_row and src_col == target_col:
                continue  # 已在正確位置

            src = board_grid[src_row][src_col]
            dst = board_grid[target_row][target_col]
            actions.append(Action(
                type="drag",
                x=src["x"], y=src["y"],
                desc=f"移動 {unit_info.get('name','?')} ({src_row},{src_col})->({target_row},{target_col})",
                extra={"to_x": dst["x"], "to_y": dst["y"], "_src_row": src_row},
            ))

        # 拖曳順序：後排先搬（row 3→2→1→0），避免前排衝突
        actions.sort(key=lambda a: -a.extra.get("_src_row", 0))

        if actions:
            log.info("Positioning: %d drags queued", len(actions))
        else:
            log.info("All units already in position")

        return actions

    def _formation_front_row(self, units: List[Dict]) -> List[tuple]:
        """
        預設陣型：所有角色排在第一排 (Row 0)。

        邏輯：
        - 已在 Row 0 的角色留在原位（不移動）
        - 其他行的角色搬到 Row 0 的空位
        - 超過 5 個溢出到 Row 1
        """
        # 分離前排和後排角色
        front_units = [u for u in units if u["row"] == 0]
        back_units = [u for u in units if u["row"] != 0]

        # 前排角色佔用的欄位
        occupied_cols = {u["col"] for u in front_units}
        # Row 0 剩餘可用的欄位（依序分配）
        available_cols = [c for c in range(5) if c not in occupied_cols]

        result = []

        # 前排角色：維持原位
        for u in front_units:
            result.append((u, (0, u["col"])))

        # 後排角色：填入 Row 0 空位，溢出到 Row 1
        for i, u in enumerate(back_units):
            if i < len(available_cols):
                result.append((u, (0, available_cols[i])))
            else:
                overflow_col = i - len(available_cols)
                result.append((u, (1, overflow_col % 5)))

        return result

    def _formation_angel_column(self, units: List[Dict]) -> List[tuple]:
        """
        勝光天使陣型：勝光天使在 Row 0 中央 (col 2)，
        其餘角色排在同一行 (col 2) 的 Row 1, 2, 3。
        超過 4 個角色時，溢出到前排兩側。

        邏輯：
        - 已在正確位置的角色不移動
        - 其餘角色填入空的目標格
        """
        angel = None
        others = []
        for unit in units:
            name = unit.get("name", "")
            if "勝光" in name or "天使" in name or "angel" in name.lower() or "victory" in name.lower():
                angel = unit
            else:
                others.append(unit)

        center_col = 2

        # 目標格位（依優先序）
        # 勝光天使佔 (0, 2)，其餘排在後方同列，溢出到兩側
        target_slots = [
            (1, center_col), (2, center_col), (3, center_col),
            (0, 1), (0, 3), (0, 0), (0, 4),
        ]

        result = []

        # 勝光天使 → (0, 2)
        if angel:
            result.append((angel, (0, center_col)))

        # 找出已在目標格的角色（不需移動）
        assigned_slots = set()
        unassigned = []
        for unit in others:
            pos = (unit["row"], unit["col"])
            if pos in target_slots and pos not in assigned_slots:
                result.append((unit, pos))
                assigned_slots.add(pos)
            else:
                unassigned.append(unit)

        # 剩餘角色填入空的目標格
        remaining_slots = [s for s in target_slots if s not in assigned_slots]
        for i, unit in enumerate(unassigned):
            if i < len(remaining_slots):
                result.append((unit, remaining_slots[i]))
            else:
                result.append((unit, (3, i % 5)))

        return result

    def _llm_decide(self, frame: np.ndarray, context_desc: str) -> List[Action]:
        """LLM 兜底決策"""
        prompt = self._system_prompt.replace(
            "{recent_actions}", self.memory.recent_summary()
        )
        if context_desc:
            prompt += f"\n\nAdditional context: {context_desc}"

        prompt += "\n\nAnalyze this screenshot and decide what to do next."

        content = self.vision._call_llm(prompt, frame, bypass_interval=True)
        if content is None:
            return []

        data = VisionClient._parse_json(content)
        if data is None:
            return []

        # Only update phase if LLM returns a known state name
        if "phase" in data:
            phase = data["phase"]
            if self.state_machine.get_state(phase) is not None:
                self.state_machine.tick(phase)

        observations = data.get("observations", [])
        reasoning = data.get("reasoning", "")
        if observations:
            log.info("LLM obs: %s", "; ".join(observations[:3]))
        if reasoning:
            log.info("LLM reasoning: %s", reasoning[:100])

        actions = []
        fw, fh = frame.shape[1], frame.shape[0]
        for a in data.get("actions", []):
            action = Action(
                type=a.get("type", "skip"),
                grid=a.get("grid", ""),
                desc=a.get("desc", ""),
                seconds=a.get("seconds", 0.5),
            )
            if action.type == "click" and action.grid:
                action.x, action.y = self.vision.grid.grid_to_pixel(
                    action.grid, fw, fh,
                )
            actions.append(action)

        return actions

    def _llm_locate_target(self, frame: np.ndarray, target: str) -> Optional[Dict]:
        """用 LLM 找到目標元素在畫面上的位置"""
        prompt = f"""Look at this screenshot and find the element: "{target}"
Return JSON: {{"grid": "C7", "found": true, "desc": "description"}}
If not found, return: {{"found": false}}"""

        result = self.vision.analyze(frame, prompt, bypass_interval=True)
        if result and result.get("found"):
            return result
        return None

    def observe_result(self, frame: np.ndarray, action: Action, success: bool):
        """觀察動作執行後的結果"""
        result = "success" if success else "no_change"
        if self.memory.history:
            self.memory.history[-1]["result"] = result

    def save_memory(self):
        self.memory.save()
