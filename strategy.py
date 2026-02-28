"""規則引擎 + 策略框架 — YAML 驅動的遊戲策略"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ======================================================================
# Action 資料結構
# ======================================================================
@dataclass
class Action:
    """操作指令"""
    type: str             # "click" | "wait" | "drag" | "skip" | "llm_fallback"
    target: str = ""      # 點擊目標描述
    grid: str = ""        # 網格座標，如 "C7"
    x: int = 0            # 像素座標 x
    y: int = 0            # 像素座標 y
    seconds: float = 0.5  # wait 的秒數
    desc: str = ""        # 動作描述
    extra: Optional[Dict] = None


# ======================================================================
# 規則引擎
# ======================================================================
class RuleEngine:
    """
    通用規則引擎，從 YAML 策略配置中讀取規則。

    規則格式：
    - condition: 條件名稱/描述
      actions: 動作列表
      priority: 優先級（數字越小越優先）
    """

    def __init__(self, strategy_cfg: Dict[str, Any]):
        self._strategy = strategy_cfg
        self._relic_priorities: Dict[str, int] = {}
        self._hero_priorities: Dict[str, int] = {}
        self._deploy_rules: Dict[str, List[str]] = {}

        self._load_relics(strategy_cfg.get("relics", []))
        self._load_deck(strategy_cfg.get("deck", []))
        self._deploy_rules = strategy_cfg.get("deploy_rules", {})

    def _load_relics(self, relics: List[Dict]):
        for r in relics:
            name = r.get("name", "")
            priority = r.get("priority", 99)
            if name:
                self._relic_priorities[name] = priority

    def _load_deck(self, deck: List[Dict]):
        for i, hero in enumerate(deck):
            name = hero.get("name", "")
            if name:
                self._hero_priorities[name] = i + 1

    def get_relic_priority(self, relic_name: str) -> int:
        """取得聖物優先序，越小越好。未知聖物回傳 99。"""
        # Exact match first
        if relic_name in self._relic_priorities:
            return self._relic_priorities[relic_name]
        # Substring match
        for name, priority in self._relic_priorities.items():
            if name in relic_name or relic_name in name:
                return priority
        return 99

    def get_hero_priority(self, hero_name: str) -> int:
        """取得英雄優先序，越小越好。未知英雄回傳 99。"""
        if hero_name in self._hero_priorities:
            return self._hero_priorities[hero_name]
        for name, priority in self._hero_priorities.items():
            if name in hero_name or hero_name in name:
                return priority
        return 99

    def get_hero_info(self, hero_name: str) -> Optional[Dict]:
        """取得英雄完整資訊"""
        for hero in self._strategy.get("deck", []):
            if hero.get("name") == hero_name:
                return hero
            if hero.get("name", "") in hero_name or hero_name in hero.get("name", ""):
                return hero
        return None

    def get_deploy_rules(self) -> Dict[str, List[str]]:
        return dict(self._deploy_rules)

    def get_positioning(self) -> Dict[str, Any]:
        return self._strategy.get("positioning", {})

    def get_economy_rules(self) -> Dict[str, str]:
        return self._strategy.get("economy", {})

    def get_daily_shop_rules(self) -> List[Dict]:
        return self._strategy.get("daily_shop", [])

    def rank_relics(self, relic_names: List[str]) -> List[str]:
        """依優先序排序聖物名稱"""
        return sorted(relic_names, key=lambda n: self.get_relic_priority(n))

    def rank_heroes(self, hero_names: List[str]) -> List[str]:
        """依優先序排序英雄名稱"""
        return sorted(hero_names, key=lambda n: self.get_hero_priority(n))

    def should_buy_hero(self, hero_name: str) -> bool:
        """判斷是否應該購買此英雄"""
        return self.get_hero_priority(hero_name) < 99

    @property
    def composition(self) -> str:
        return self._strategy.get("composition", "")

    @property
    def relic_priorities(self) -> Dict[str, int]:
        return dict(self._relic_priorities)

    @property
    def hero_priorities(self) -> Dict[str, int]:
        return dict(self._hero_priorities)


# ======================================================================
# 策略抽象基底類
# ======================================================================
class GameStrategy(ABC):
    """
    每個遊戲實作自己的策略。
    提供 on_state() 方法，根據當前狀態和畫面上下文回傳動作列表。
    """

    def __init__(self, strategy_cfg: Dict[str, Any]):
        self.rules = RuleEngine(strategy_cfg)
        self._strategy_cfg = strategy_cfg

    @abstractmethod
    def on_state(self, state: str, context: Dict[str, Any]) -> List[Action]:
        """
        根據當前狀態和上下文產生動作列表。

        Args:
            state: 目前狀態名稱
            context: 上下文資訊，可能包含：
                - screen_text: LLM 辨識出的文字
                - relics: 可選聖物列表
                - shop_heroes: 商店英雄列表
                - gold: 當前金幣
                - population: 當前人口
                - timeout: 是否已超時

        Returns:
            動作列表
        """
        ...

    def build_strategy_prompt(self) -> str:
        """為 LLM 構建策略提示文字"""
        cfg = self._strategy_cfg
        parts = [f"Composition: {cfg.get('composition', 'unknown')}"]

        deck = cfg.get("deck", [])
        if deck:
            parts.append("Deck:")
            for hero in deck:
                line = f"  - {hero['name']} ({hero.get('class','?')}, {hero.get('role','?')})"
                if hero.get("passive"):
                    line += f" — {hero['passive']}"
                parts.append(line)

        positioning = cfg.get("positioning", {})
        if positioning:
            parts.append(f"Positioning: default={positioning.get('default','?')}")
            overrides = positioning.get("overrides", {})
            for name, pos in overrides.items():
                parts.append(f"  - {name}: {pos}")

        economy = cfg.get("economy", {})
        if economy:
            parts.append(f"Economy: {economy.get('default', '?')}")

        return "\n".join(parts)


# ======================================================================
# ElfTW 策略實作
# ======================================================================
class ElfTWStrategy(GameStrategy):
    """指尖棋兵 — 簡易策略 (固定座標點擊)"""

    def __init__(self, strategy_cfg: Dict[str, Any]):
        super().__init__(strategy_cfg)
        self._pos = strategy_cfg.get("positions", {})
        self._prep_round = 0
        self._max_prep_rounds = strategy_cfg.get("max_prep_rounds", 3)
        self._is_first_round_of_match = True  # 每場比賽第一回合
        self._positioned_this_prep = False     # 本次準備階段是否已調陣

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
            self._prep_round = 0
            self._positioned_this_prep = False

        # 回到主選單表示本局結束，重置首輪旗標
        if state == "main_menu":
            self._is_first_round_of_match = True

        handler = getattr(self, f"_handle_{state}", None)
        if handler:
            return handler(context)
        return []

    # ------------------------------------------------------------------
    # 準備階段 — 簡單策略：
    #   1. 升人口（上限未到 6 時）
    #   2. 不斷 刷新 → 買左中右
    #   3. 錢用完 → 按準備
    #
    # 前 N 次 tick：固定做多輪刷新+買卡（不問 LLM）
    # 超過 N 次後：問 LLM 讀金幣，沒錢就按準備
    # ------------------------------------------------------------------
    def _handle_preparation(self, context: Dict[str, Any]) -> List[Action]:
        self._prep_round += 1
        card_slots = self._pos.get("card_slots", [])
        actions: List[Action] = []
        time_in_state = context.get("time_in_state", 0)

        # 安全網：準備階段超過 25 秒，直接點準備（不管 LLM 判斷）
        if time_in_state > 25:
            a = self._click_pos("ready_btn", "超時點準備")
            return [a] if a else []

        # 固定快速操作（不問 LLM）
        # 首回合：做 max_prep_rounds 次 tick，1:1 升人口與刷新（快速拉人口）
        # 之後每回合：1 tick 全刷新填角色，第 2 tick 起 2:1 模式
        effective_max = self._max_prep_rounds if self._is_first_round_of_match else 1

        if self._prep_round <= effective_max:

            if self._is_first_round_of_match:
                # ===== 首局：1:1 模式（升人口便宜，快速拉滿） =====
                # 第 1 tick 先買免費卡
                if self._prep_round == 1:
                    for i, slot in enumerate(card_slots):
                        actions.append(Action(
                            type="click", x=slot["x"], y=slot["y"],
                            desc=f"買免費卡{i+1}",
                        ))

                # 交替：升人口→買 → 刷新→買 → 升人口→買 → 刷新→買
                for r in range(4):
                    if r % 2 == 0:
                        # 偶數輪：升人口 → 買
                        pop = self._click_pos("population_btn", "升人口")
                        if pop:
                            actions.append(pop)
                    else:
                        # 奇數輪：刷新 → 買
                        refresh = self._click_pos("refresh_btn", "刷新商店")
                        if refresh:
                            actions.append(refresh)

                    for i, slot in enumerate(card_slots):
                        actions.append(Action(
                            type="click", x=slot["x"], y=slot["y"],
                            desc=f"買卡{i+1}",
                        ))
            else:
                # ===== 非首局第 1 tick：全刷新+買（先填空位） =====
                for r in range(3):
                    refresh = self._click_pos("refresh_btn", "刷新商店")
                    if refresh:
                        actions.append(refresh)
                    for i, slot in enumerate(card_slots):
                        actions.append(Action(
                            type="click", x=slot["x"], y=slot["y"],
                            desc=f"買卡{i+1}",
                        ))

            # 點右上角秒數框防卡頓
            safe = self._click_pos("safe_click", "點擊秒數框")
            if safe:
                actions.append(safe)

            return actions

        # 買完卡後：調整陣型（每次準備階段只做一次）
        if not self._positioned_this_prep:
            self._positioned_this_prep = True
            return [Action(
                type="llm_position",
                desc="偵測棋盤角色位置並調整陣型",
            )]

        # 超過 max_prep_rounds 後：問 LLM 讀金幣和人口
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

    # ------------------------------------------------------------------
    # 其他狀態 — 固定座標點一下就好
    # ------------------------------------------------------------------
    def _handle_main_menu(self, context: Dict[str, Any]) -> List[Action]:
        a = self._click_pos("battle_btn", "點擊對戰")
        return [a] if a else [Action(type="llm_fallback", desc="找不到對戰按鈕")]

    def _handle_matchmaking(self, context: Dict[str, Any]) -> List[Action]:
        return [Action(type="wait", seconds=5, desc="等待配對")]

    def _handle_relic_select(self, context: Dict[str, Any]) -> List[Action]:
        a = self._click_pos("relic_first", "選第一個聖物")
        return [a] if a else [Action(type="llm_fallback", desc="找不到聖物")]

    def _handle_battle(self, context: Dict[str, Any]) -> List[Action]:
        return [Action(type="wait", seconds=5, desc="等待戰鬥結束")]

    def _handle_match_result_win(self, context: Dict[str, Any]) -> List[Action]:
        a = self._click_pos("confirm_btn", "確認勝利結算")
        return [a] if a else [Action(type="llm_fallback", desc="找不到確認按鈕")]

    def _handle_match_result_defeat(self, context: Dict[str, Any]) -> List[Action]:
        a = self._click_pos("confirm_btn", "確認失敗結算")
        return [a] if a else [Action(type="llm_fallback", desc="找不到確認按鈕")]

    def _handle_protection_popup(self, context: Dict[str, Any]) -> List[Action]:
        a = self._click_pos("accept_btn", "接受庇護")
        return [a] if a else [Action(type="llm_fallback", desc="找不到庇護按鈕")]


# ======================================================================
# 策略工廠
# ======================================================================
STRATEGY_REGISTRY: Dict[str, type] = {
    "elftw": ElfTWStrategy,
}


def create_strategy(game_name: str, strategy_cfg: Dict[str, Any]) -> GameStrategy:
    """根據遊戲名稱建立對應的策略"""
    cls = STRATEGY_REGISTRY.get(game_name)
    if cls is None:
        log.warning("No strategy registered for '%s', using ElfTW as default", game_name)
        cls = ElfTWStrategy
    return cls(strategy_cfg)
