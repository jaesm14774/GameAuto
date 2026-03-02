"""規則引擎 + 策略框架 — YAML 驅動的遊戲策略"""

import importlib
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
# 策略工廠
# ======================================================================
def create_strategy(game_name: str, strategy_cfg: Dict[str, Any]) -> GameStrategy:
    """根據遊戲名稱建立對應的策略，從 games/ registry 自動發現"""
    from games import GAME_REGISTRY

    entry = GAME_REGISTRY.get(game_name)
    if entry is None:
        log.warning("No strategy registered for '%s', trying elftw as default", game_name)
        entry = GAME_REGISTRY.get("elftw")

    if entry is None:
        raise ValueError(f"No strategy registered for '{game_name}' and no default available")

    module_path, class_name = entry["strategy_cls"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(strategy_cfg)
