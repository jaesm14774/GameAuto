"""Vision Client + Game Agent — 視覺辨識與決策引擎"""

import base64
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

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
        content = content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            lines = [line for line in lines if not line.strip().startswith("```")]
            content = "\n".join(lines)

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                try:
                    return json.loads(content[start:end])
                except json.JSONDecodeError:
                    pass
        log.warning("Cannot parse LLM response as JSON")
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

        self._last_tick_time = 0.0  # Track last meaningful tick for force-tick
        self._llm_prep_count = 0   # LLM 準備輪計數，用於 2:1 升人口模式

        # 庇護/敗場追蹤
        self._defeat_count = 0          # 已使用的庇護次數
        self._expect_protection = False  # 是否正在等待庇護彈窗
        self._should_stop = False        # 是否應該終止程式
        self._stop_reason = ""           # 終止原因

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
        )

        if not skip_llm:
            state_hints = self.state_machine.get_detect_hints()
            detected = self.vision.detect_state(frame, state_hints)
            self.state_machine.tick(detected)
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
            elif action.type == "llm_decide_prep":
                # LLM 只決定 buy 或 ready，然後用固定座標執行
                prep_actions = self._llm_decide_preparation(frame, action.desc)
                resolved.extend(prep_actions)
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

        # 選完聖物後直接進入準備階段，省掉一次 LLM 偵測
        if current == "relic_select" and resolved:
            self.state_machine.transition_to("preparation")

        return resolved

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
            elif current_pop < max_pop:
                # 還有空位 → 不升，只買卡填角色
                log.info("Slots available (%d/%d), just buying", current_pop, max_pop)
            else:
                # max_pop >= 6 → 不升
                log.info("Max pop %d >= 6, just buying", max_pop)
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
        """用固定座標點擊準備按鈕"""
        positions = self.strategy._strategy_cfg.get("positions", {})
        ready_pos = positions.get("ready_btn")
        if ready_pos:
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
        if has_angel:
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
