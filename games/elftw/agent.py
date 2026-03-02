"""ElfTW Agent — 指尖棋兵遊戲專用決策引擎"""

import logging
import random
import time
from typing import Any, Dict, List, Optional

import numpy as np

from core.agent import GameAgent
from core.strategy import Action
from core.vision import VisionClient
from games.elftw.formations import formation_angel_column, formation_front_row
from games.elftw.knowledge import CharacterKnowledgeBase

log = logging.getLogger("agent")


class ElfTWAgent(GameAgent):
    """
    指尖棋兵專用決策引擎。
    繼承 GameAgent，覆寫 resolve_action() 處理 elftw 特定動作類型。
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        game_cfg = config.get("game", {})
        strategy_cfg = config.get("strategy", {})

        # 角色知識庫
        kb_path = strategy_cfg.get("knowledge_base",
                                   f"characters_{game_cfg.get('name', 'game')}.json")
        self.knowledge = CharacterKnowledgeBase(kb_path)

        # 通知策略引擎是否有知識庫可用
        if hasattr(self.strategy, '_has_knowledge_base'):
            self.strategy._has_knowledge_base = not self.knowledge.is_empty

        self._llm_prep_count = 0   # LLM 準備輪計數，用於 2:1 升人口模式

        # 戰鬥預分析快取
        self._pre_analysis: Optional[Dict] = None
        self._pre_analyzed = False

        # 庇護/敗場追蹤
        self._defeat_count = 0          # 已使用的庇護次數
        self._expect_protection = False  # 是否正在等待庇護彈窗
        self._should_stop = False        # 是否應該終止程式
        self._stop_reason = ""           # 終止原因

        # 對戰次數檢查（每次回到主選單重新檢查）
        self._battle_count_checked = False

        # 覆蓋 system prompt 為 elftw 專用
        self._system_prompt = self._build_system_prompt(game_cfg, strategy_cfg)

    @property
    def should_stop(self) -> bool:
        return self._should_stop

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    # ------------------------------------------------------------------
    # Game-specific state tracking (protection/defeat)
    # ------------------------------------------------------------------
    def _post_state_update(self, current: str, frame: np.ndarray):
        """庇護追蹤邏輯 + 準備階段計數重置

        注意：第1名和第2名都算勝利，不會出現庇護彈窗。
        庇護只在第3名以後（match_result_defeat）才可能出現。
        """
        # 進入失敗結算（第3名以後）→ 標記等待庇護
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

        # 離開主選單時重置對戰次數檢查旗標
        if current != "main_menu":
            self._battle_count_checked = False

        # 離開準備階段時重置 LLM 輪計數
        if current != "preparation":
            self._llm_prep_count = 0
            # 進入新的準備階段時清除預分析快取
            if current == "battle":
                self._pre_analyzed = False

    # ------------------------------------------------------------------
    # Post-tick hook: force LLM + relic→preparation shortcut
    # ------------------------------------------------------------------
    def _post_tick(self, current: str, resolved: List[Action],
                   frame: np.ndarray):
        """點準備後強制下次 tick 呼叫 LLM + 選完聖物直接進準備"""
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

    # ------------------------------------------------------------------
    # Resolve game-specific action types
    # ------------------------------------------------------------------
    def resolve_action(self, action: Action, frame: np.ndarray) -> Optional[List[Action]]:
        if action.type == "check_battle_count":
            return self._check_battle_count(frame)
        elif action.type == "llm_position":
            return self._llm_position_units(frame)
        elif action.type == "llm_smart_prep":
            return self._handle_smart_preparation(frame)
        elif action.type == "llm_decide_prep":
            return self._llm_decide_preparation(frame, action.desc)
        elif action.type == "llm_relic_select":
            return self._llm_select_relic(frame)
        elif action.type == "llm_pre_analyze":
            self._pre_analyze_board(frame)
            return []  # No actions produced
        return None  # Not handled — fall through to base class

    # ------------------------------------------------------------------
    # 對戰次數檢查
    # ------------------------------------------------------------------
    def _check_battle_count(self, frame: np.ndarray) -> List[Action]:
        """檢查主選單的剩餘對戰次數 (X/50)，若為 0 則終止程式。"""
        if self._battle_count_checked:
            return self._build_battle_click()

        self._battle_count_checked = True

        prompt = (
            "Look at this game main menu screen.\n"
            "Find the battle count indicator showing '(X/50)' format.\n"
            "X is the number of remaining rated battles.\n\n"
            "Return ONLY this JSON, nothing else:\n"
            '{"remaining": 0, "total": 50}'
        )

        content = self.vision._call_llm(prompt, frame,
                                         annotate=False, bypass_interval=True)
        if content is None:
            log.warning("對戰次數讀取失敗，繼續對戰")
            return self._build_battle_click()

        data = VisionClient._parse_json(content)
        if data is None:
            log.warning("對戰次數解析失敗，繼續對戰")
            return self._build_battle_click()

        remaining = self._safe_int(data.get("remaining"), -1)
        total = self._safe_int(data.get("total"), 50)
        log.info("剩餘對戰次數: %d/%d", remaining, total)

        if remaining == 0:
            self._should_stop = True
            self._stop_reason = (
                f"計分場次已用完 (0/{total})，自動終止程式。"
            )
            log.error(self._stop_reason)
            return []

        return self._build_battle_click()

    def _build_battle_click(self) -> List[Action]:
        """產生點擊對戰按鈕的動作。"""
        positions = self.strategy._strategy_cfg.get("positions", {})
        battle_pos = positions.get("battle_btn")
        if battle_pos:
            return [Action(type="click", x=battle_pos["x"], y=battle_pos["y"],
                           desc="點擊對戰")]
        return [Action(type="llm_fallback", desc="找不到對戰按鈕")]

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
    # 裁切敵方區域
    # ------------------------------------------------------------------
    def _get_player_crop_y(self) -> int:
        """Get Y coordinate to crop above, excluding enemy area.

        Auto-computed from board_grid front row with a manual override option.
        """
        positions = self.strategy._strategy_cfg.get("positions", {})
        # Manual override
        override = positions.get("player_board_y_start")
        if override:
            return int(override)
        # Auto-compute from board_grid front row
        board_grid = positions.get("board_grid", [])
        if not board_grid or not board_grid[0]:
            return 0
        front_row_y = min(pos["y"] for pos in board_grid[0])
        return max(0, front_row_y - 50)

    # ------------------------------------------------------------------
    # 型別安全工具
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        """安全轉換為 int，避免 LLM 回傳字串或 None 導致比較失敗"""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default

    # ------------------------------------------------------------------
    # 讀取金幣 + 人口（專用 prompt，不裁切畫面）
    # ------------------------------------------------------------------
    def _llm_read_gold_and_pop(self, frame: np.ndarray) -> Optional[Dict]:
        """
        用簡單 prompt 讀取金幣和人口。
        使用完整畫面（不裁切），確保 LLM 能看到金幣顯示。
        """
        prompt = (
            "Look at this auto-chess game screen.\n"
            "I ONLY need TWO things:\n\n"
            "1. GOLD: Find the gold/coin number on screen. "
            "It is usually a yellow number near a coin icon.\n"
            "2. POPULATION: Find the helmet/knight icon showing 'X/Y' "
            "(e.g. '3/5'). First number = units on board, second = max.\n\n"
            "Return ONLY this JSON, nothing else:\n"
            '{"gold": 5, "current_pop": 3, "max_pop": 5}'
        )

        content = self.vision._call_llm(prompt, frame,
                                         annotate=False, bypass_interval=True)
        if content is None:
            return None
        return VisionClient._parse_json(content)

    # ------------------------------------------------------------------
    # 智慧商店 + 棋盤讀取（合併 LLM call）
    # ------------------------------------------------------------------
    def _llm_read_shop_and_board(self, frame: np.ndarray) -> Optional[Dict]:
        """
        一次 LLM call 同時讀取：商店卡片名稱 + 棋盤角色位置 + 金幣 + 人口。
        Enemy area is cropped out before sending to LLM.
        """
        # Crop out enemy area
        crop_y = self._get_player_crop_y()
        analysis_frame = frame[crop_y:] if crop_y > 0 else frame

        prompt = (
            "Analyze this auto-chess game screenshot during preparation phase.\n"
            "Read ALL of the following:\n\n"
            "1. SHOP CARDS at the bottom: list each card's name\n"
            "2. UNITS on the board:\n"
            "   - Row 0 = front (top of visible board), Row 3 = back (closest to shop)\n"
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

        content = self.vision._call_llm(prompt, analysis_frame,
                                         annotate=False, bypass_interval=True)
        if content is None:
            return None
        return VisionClient._parse_json(content)

    # ------------------------------------------------------------------
    # 智慧購買
    # ------------------------------------------------------------------
    def _smart_buy_cards(self, shop_data: List[Dict],
                         frame: np.ndarray) -> List[Action]:
        """根據知識庫決定購買哪些商店卡片。"""
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
        """在戰鬥階段預先分析棋盤，快取結果供下次準備階段使用。"""
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
        """LLM 判讀聖物選擇畫面，依照優先清單選擇最佳聖物。"""
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

    # ------------------------------------------------------------------
    # 智慧準備階段（知識庫驅動）
    # ------------------------------------------------------------------
    def _handle_smart_preparation(self, frame: np.ndarray) -> List[Action]:
        """
        智慧準備階段：
        1. 用簡單 prompt 讀取金幣+人口（不裁切畫面，確保看到金幣）
        2. 金幣 < 8 → 直接準備
        3. 金幣 >= 8 → 重置回盲買循環
        """
        self._llm_prep_count += 1
        self._pre_analysis = None  # 清除預分析快取

        data = self._llm_read_gold_and_pop(frame)

        if data is None:
            log.warning("智慧準備：LLM 讀取失敗，fallback 到按準備")
            return self._build_ready_actions()

        gold = self._safe_int(data.get("gold"), 0)
        current_pop = self._safe_int(data.get("current_pop"), 0)
        max_pop = self._safe_int(data.get("max_pop"), 0)

        log.info("智慧準備: gold=%d, pop=%d/%d", gold, current_pop, max_pop)

        # 更新策略引擎的人口追蹤
        if max_pop > 0:
            self.strategy._last_known_max_pop = max_pop

        # 金幣 < 8 → 直接準備
        if gold < 8:
            log.info("金幣=%d (<8)，點擊準備", gold)
            return self._build_ready_actions()

        # 金幣 >= 8 → 重置盲買計數，回到盲買循環
        log.info("金幣=%d (>=8)，重置回盲買循環", gold)
        self.strategy._prep_round = 0
        return []

    def _llm_decide_preparation(self, frame: np.ndarray, prompt: str) -> List[Action]:
        """
        LLM 只讀金幣和人口，決定：金幣 < 8 按準備，否則回到盲買循環。
        """
        self._llm_prep_count += 1

        data = self._llm_read_gold_and_pop(frame)

        if data is None:
            log.warning("LLM prep decision failed, clicking Ready as fallback")
            return self._build_ready_actions()

        gold = self._safe_int(data.get("gold"), 0)
        current_pop = self._safe_int(data.get("current_pop"), 0)
        max_pop = self._safe_int(data.get("max_pop"), 0)
        log.info("LLM read: gold=%d, pop=%d/%d", gold, current_pop, max_pop)

        # 更新策略引擎的人口追蹤
        if max_pop > 0:
            self.strategy._last_known_max_pop = max_pop

        # 金幣 < 8 → 直接按準備
        if gold < 8:
            log.info("Gold=%d (<8), clicking Ready", gold)
            return self._build_ready_actions()

        # 金幣 >= 8 → 重置盲買計數，回到盲買循環（不在 LLM 階段產生購買動作）
        log.info("Gold=%d (>=8), resetting to blind buy cycles", gold)
        self.strategy._prep_round = 0
        return []

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
            self._force_llm_next_tick = True
            return [Action(type="click", x=ready_pos["x"], y=ready_pos["y"],
                           desc="點擊準備")]
        log.error("No ready_btn position configured!")
        return []

    # ------------------------------------------------------------------
    # 陣型調整
    # ------------------------------------------------------------------
    def _llm_position_units(self, frame: np.ndarray) -> List[Action]:
        """用 LLM 偵測棋盤上角色位置，產生拖曳指令調整陣型。
        Enemy area is cropped out before sending to LLM.
        """
        board_grid = self.strategy._strategy_cfg.get("positions", {}).get("board_grid", [])
        if not board_grid:
            log.warning("board_grid not configured, skipping positioning")
            return []

        # Crop out enemy area
        crop_y = self._get_player_crop_y()
        analysis_frame = frame[crop_y:] if crop_y > 0 else frame

        prompt = (
            "This is an auto-chess game board screenshot showing only the player's area.\n"
            "The board is a grid with 4 rows and 5 columns:\n"
            "  - Row 0 = front row (top of visible board)\n"
            "  - Row 1, 2 = middle rows\n"
            "  - Row 3 = back row (closest to the bottom edge / shop)\n"
            "  - Col 0 = leftmost column\n"
            "  - Col 4 = rightmost column\n\n"
            "List every unit you see on the board.\n"
            "For each unit, give its name and grid position.\n\n"
            "Also: is there a unit called '聖光天使' "
            "(Victory Light Angel — a large bright angel/winged character)?\n\n"
            "Return ONLY valid JSON:\n"
            '{"units": [{"name": "unit_name", "row": 0, "col": 2}], '
            '"has_victory_angel": false}'
        )

        content = self.vision._call_llm(prompt, analysis_frame,
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

        log.info("Detected %d units at %s, 聖光天使=%s",
                 len(units),
                 [(u.get("name", "?"), u["row"], u["col"]) for u in units],
                 has_angel)

        # 計算目標位置
        if not self.knowledge.is_empty:
            known_count = sum(1 for u in units if self.knowledge.get(u.get("name", "")))
            if known_count > 0:
                log.info("使用知識庫站位 (%d/%d 角色已知)", known_count, len(units))
                targets = self.knowledge.compute_formation(units)
            elif has_angel:
                targets = formation_angel_column(units)
            else:
                targets = formation_front_row(units)
        elif has_angel:
            targets = formation_angel_column(units)
        else:
            targets = formation_front_row(units)

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
