"""ElfTW 策略實作 — 指尖棋兵遊戲專用策略"""

import logging
import random
from typing import Any, Dict, List, Optional

from core.strategy import Action, GameStrategy

log = logging.getLogger(__name__)


class ElfTWStrategy(GameStrategy):
    """指尖棋兵 — 簡易策略 (固定座標點擊)"""

    def __init__(self, strategy_cfg: Dict[str, Any]):
        super().__init__(strategy_cfg)
        self._pos = strategy_cfg.get("positions", {})
        self._prep_round = 0
        self._total_prep_ticks = 0             # 本次準備總 tick 數（不因 gold>=8 重置）
        self._max_prep_rounds = strategy_cfg.get("max_prep_rounds", 2)
        self._is_first_round_of_match = True  # 每場比賽第一回合
        self._battle_round = 0                 # 本場比賽的戰鬥回合數
        self._positioned_this_round = False     # 本回合是否已調過陣
        self._smart_bought_this_prep = False   # 本次準備是否已做智慧購買
        self._has_knowledge_base = False       # 是否有角色知識庫（由 agent 設定）
        self._last_known_max_pop = 0           # 上次 LLM 讀到的人口上限（由 agent 更新）

    def _click_pos(self, key: str, desc: str) -> Optional[Action]:
        """從 positions 取座標產生 click Action，找不到回傳 None"""
        p = self._pos.get(key)
        if p:
            return Action(type="click", x=p["x"], y=p["y"], desc=desc)
        return None

    def on_state(self, state: str, context: Dict[str, Any]) -> List[Action]:
        # 離開準備階段時重置輪數計數
        if state != "preparation":
            if state == "battle" and self._prep_round > 0:
                # 從準備進入戰鬥：下次準備不再是首輪
                self._is_first_round_of_match = False
                self._battle_round += 1
            self._prep_round = 0
            self._total_prep_ticks = 0
            self._smart_bought_this_prep = False
            self._positioned_this_round = False

        # 回到主選單表示本局結束，重置所有旗標
        if state == "main_menu":
            self._is_first_round_of_match = True
            self._battle_round = 0
            self._positioned_this_round = False
            self._last_known_max_pop = 0

        handler = getattr(self, f"_handle_{state}", None)
        if handler:
            return handler(context)
        return []

    # ------------------------------------------------------------------
    # 準備階段策略：
    #   1. 先跑 max_prep_rounds 輪盲買（根據人口等級用不同循環順序）
    #   2. 盲買完後交給 LLM：調站位 + 讀金幣（< 8 就按準備）
    #
    # 人口循環規則（每輪都含人口）：
    #   < 4:  人口→刷新→買→刷新→買→人口→刷新→買
    #   4~5:  刷新→買→人口→刷新→買→刷新→買
    #   >= 6: (30%人口?)刷新→買→(30%人口?)刷新→買→刷新→買
    # ------------------------------------------------------------------
    def _handle_preparation(self, context: Dict[str, Any]) -> List[Action]:
        self._prep_round += 1
        self._total_prep_ticks += 1
        card_slots = self._pos.get("card_slots", [])
        actions: List[Action] = []
        time_in_state = context.get("time_in_state", 0)

        # 安全網：準備階段超過 25 秒，直接點準備（不管 LLM 判斷）
        if time_in_state > 25:
            a = self._click_pos("ready_btn", "超時點準備")
            return [a] if a else []

        # 安全網：總 tick 數過多（防止 gold>=8 無限重置盲買）
        if self._total_prep_ticks > 8:
            log.info("準備階段已執行 %d 次 tick，強制點準備", self._total_prep_ticks)
            a = self._click_pos("ready_btn", "tick上限，點準備")
            return [a] if a else []

        # Phase 1: 盲買（不呼叫 LLM，節省 30 秒準備時間）
        if self._prep_round <= self._max_prep_rounds:
            # 首局第 1 tick 先買免費卡
            if self._is_first_round_of_match and self._prep_round == 1:
                for i, slot in enumerate(card_slots):
                    actions.append(Action(
                        type="click", x=slot["x"], y=slot["y"],
                        desc=f"買免費卡{i+1}",
                    ))

            # 根據人口等級產生一輪盲買操作（每輪都包含升人口）
            actions.extend(self._build_blind_buy_cycle(card_slots))

            # 點右上角秒數框防卡頓
            safe = self._click_pos("safe_click", "點擊秒數框")
            if safe:
                actions.append(safe)

            return actions

        # Phase 2: 調整陣型（每 3 回合做一次）
        if self._battle_round > 0 and self._battle_round % 3 == 0 and not self._positioned_this_round:
            self._positioned_this_round = True
            return [Action(
                type="llm_position",
                desc=f"第{self._battle_round + 1}回合：偵測棋盤角色位置並調整陣型",
            )]

        # Phase 3: LLM 讀金幣（< 8 按準備，>= 8 繼續買）
        # 安全上限：LLM 決策超過 2 輪就直接按準備
        llm_rounds = self._prep_round - self._max_prep_rounds - 1
        if llm_rounds > 2:
            log.info("LLM 決策已超過 2 輪，直接按準備")
            a = self._click_pos("ready_btn", "超過輪數上限，點準備")
            return [a] if a else []

        # 有知識庫時：用智慧準備（合併讀商店+棋盤+金幣，知識庫決策）
        if self._has_knowledge_base and not self._smart_bought_this_prep:
            self._smart_bought_this_prep = True
            return [Action(
                type="llm_smart_prep",
                desc="智慧準備：讀取商店+棋盤+金幣，知識庫決策購買",
            )]

        # 無知識庫 / 智慧購買後：問 LLM 讀金幣和人口
        return [Action(
            type="llm_decide_prep",
            desc=(
                "Look at this game screen.\n"
                "1. Read the gold/coin number.\n"
                "2. Find the knight helmet icon showing population like '3/5'.\n"
                "   First number = units on board, second = max allowed.\n"
                "Return ONLY: {\"gold\": <number>, \"current_pop\": <first>, \"max_pop\": <second>}"
            ),
        )]

    def _build_blind_buy_cycle(self, card_slots) -> List[Action]:
        """根據當前人口等級產生一輪盲買操作。

        Args:
            card_slots: 商店卡牌座標列表

        人口等級由 _last_known_max_pop 決定（LLM 讀取後由 agent 更新），
        未知時以 battle_round 推估。

        循環規則（每輪都包含升人口）：
          < 4:  人口→刷新→買→刷新→買→人口→刷新→買
          4~5:  刷新→買→人口→刷新→買→刷新→買
          >= 6: (30%人口?)刷新→買→(30%人口?)刷新→買→刷新→買
        """
        actions: List[Action] = []

        def _buy_all():
            for i, slot in enumerate(card_slots):
                actions.append(Action(
                    type="click", x=slot["x"], y=slot["y"],
                    desc=f"買卡{i+1}",
                ))

        def _refresh():
            r = self._click_pos("refresh_btn", "刷新商店")
            if r:
                actions.append(r)

        def _pop_up():
            p = self._click_pos("population_btn", "升人口")
            if p:
                actions.append(p)

        # 根據人口等級決定順序
        pop = self._last_known_max_pop
        # 未知時用回合數推估
        if pop <= 0:
            if self._battle_round <= 1:
                pop = 2   # 早期 → 當 < 4
            elif self._battle_round <= 4:
                pop = 5   # 中期 → 當 4~5
            else:
                pop = 7   # 後期 → 當 >= 6

        if pop < 4:
            # 人口→刷新→買→刷新→買→人口→刷新→買
            _pop_up()
            _refresh()
            _buy_all()
            _refresh()
            _buy_all()
            _pop_up()
            _refresh()
            _buy_all()
        elif pop < 6:
            # 刷新→買→人口→刷新→買→刷新→買
            _refresh()
            _buy_all()
            _pop_up()
            _refresh()
            _buy_all()
            _refresh()
            _buy_all()
        else:
            # >= 6: (30%人口?)刷新→買→(30%人口?)刷新→買→刷新→買
            if random.random() < 0.3:
                _pop_up()
            _refresh()
            _buy_all()
            if random.random() < 0.3:
                _pop_up()
            _refresh()
            _buy_all()
            _refresh()
            _buy_all()

        return actions

    # ------------------------------------------------------------------
    # 其他狀態 — 固定座標點一下就好
    # ------------------------------------------------------------------
    def _handle_main_menu(self, context: Dict[str, Any]) -> List[Action]:
        return [Action(type="check_battle_count", desc="檢查剩餘對戰次數")]

    def _handle_matchmaking(self, context: Dict[str, Any]) -> List[Action]:
        return [Action(type="wait", seconds=5, desc="等待配對")]

    def _handle_relic_select(self, context: Dict[str, Any]) -> List[Action]:
        # 有聖物優先清單 → LLM 判讀選最佳聖物
        if self.rules.relic_priorities:
            return [Action(
                type="llm_relic_select",
                desc="判讀聖物並選擇最佳聖物",
            )]
        # 無清單 → 選第一個
        a = self._click_pos("relic_first", "選第一個聖物")
        return [a] if a else [Action(type="llm_fallback", desc="找不到聖物")]

    def _handle_battle(self, context: Dict[str, Any]) -> List[Action]:
        actions: List[Action] = []
        # 有知識庫時，利用戰鬥空檔預分析棋盤（隱藏 LLM 延遲）
        time_in_state = context.get("time_in_state", 0)
        if self._has_knowledge_base and 3 < time_in_state < 10:
            actions.append(Action(
                type="llm_pre_analyze",
                desc="戰鬥中預分析棋盤",
            ))
        actions.append(Action(type="wait", seconds=5, desc="等待戰鬥結束"))
        return actions

    def _handle_match_result_win(self, context: Dict[str, Any]) -> List[Action]:
        a = self._click_pos("confirm_btn", "確認勝利結算")
        return [a] if a else [Action(type="llm_fallback", desc="找不到確認按鈕")]

    def _handle_match_result_defeat(self, context: Dict[str, Any]) -> List[Action]:
        a = self._click_pos("confirm_btn", "確認失敗結算")
        return [a] if a else [Action(type="llm_fallback", desc="找不到確認按鈕")]

    def _handle_protection_popup(self, context: Dict[str, Any]) -> List[Action]:
        a = self._click_pos("accept_btn", "接受庇護")
        return [a] if a else [Action(type="llm_fallback", desc="找不到庇護按鈕")]

    def _handle_reward_popup(self, context: Dict[str, Any]) -> List[Action]:
        a = self._click_pos("accept_btn", "接受贈禮")
        return [a] if a else [Action(type="llm_fallback", desc="找不到贈禮按鈕")]
