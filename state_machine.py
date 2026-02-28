"""通用 YAML 驅動狀態機 — 從配置讀取狀態與轉場規則"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class StateDefinition:
    """單一狀態的定義"""
    name: str
    detect_hint: str = ""           # 給 LLM 的畫面辨識描述
    actions: List[Dict] = field(default_factory=list)
    timeout: float = 0.0           # 超時秒數（0 = 無限）


@dataclass
class Transition:
    """狀態轉場規則"""
    from_state: str   # "*" 表示任何狀態
    to_state: str
    condition: str    # 轉場條件描述（給 LLM 參考）


class StateMachine:
    """
    通用 YAML 驅動狀態機。

    從 config 中讀取 states / transitions，提供：
    - current_state: 目前狀態名稱
    - transition_to(state_name): 手動切換狀態
    - tick(detected_state): 根據偵測結果更新狀態
    - get_state(): 取得目前狀態定義
    - get_detect_hints(): 取得所有狀態的辨識提示
    """

    def __init__(self, config: Dict[str, Any]):
        self._states: Dict[str, StateDefinition] = {}
        self._transitions: List[Transition] = []
        self._current: str = "unknown"
        self._state_enter_time: float = time.time()
        self._prev_state: str = "unknown"

        self._load_states(config.get("states", {}))
        self._load_transitions(config.get("transitions", []))

    def _load_states(self, states_cfg: Dict[str, Any]):
        for name, definition in states_cfg.items():
            state = StateDefinition(
                name=name,
                detect_hint=definition.get("detect", ""),
                actions=definition.get("actions", []),
                timeout=definition.get("timeout", 0.0),
            )
            self._states[name] = state

        # Ensure "unknown" state exists
        if "unknown" not in self._states:
            self._states["unknown"] = StateDefinition(
                name="unknown",
                detect_hint="Unable to identify the current screen",
            )

    def _load_transitions(self, transitions_cfg: List[Dict]):
        for t in transitions_cfg:
            self._transitions.append(Transition(
                from_state=t.get("from", "*"),
                to_state=t.get("to", "unknown"),
                condition=t.get("condition", ""),
            ))

    @property
    def current_state(self) -> str:
        return self._current

    @property
    def previous_state(self) -> str:
        return self._prev_state

    @property
    def time_in_state(self) -> float:
        """目前狀態已持續的秒數"""
        return time.time() - self._state_enter_time

    def get_state(self, name: Optional[str] = None) -> Optional[StateDefinition]:
        """取得狀態定義，預設取目前狀態"""
        return self._states.get(name or self._current)

    def get_all_states(self) -> Dict[str, StateDefinition]:
        return dict(self._states)

    def get_detect_hints(self) -> Dict[str, str]:
        """取得所有狀態的辨識提示 {state_name: detect_hint}"""
        return {
            name: s.detect_hint
            for name, s in self._states.items()
            if s.detect_hint
        }

    def get_valid_transitions(self) -> List[Transition]:
        """取得從目前狀態可能的轉場"""
        return [
            t for t in self._transitions
            if t.from_state == self._current or t.from_state == "*"
        ]

    def transition_to(self, state_name: str) -> bool:
        """手動切換到指定狀態"""
        if state_name not in self._states:
            log.warning("Unknown state: %s", state_name)
            return False

        if state_name == self._current:
            return False

        self._prev_state = self._current
        self._current = state_name
        self._state_enter_time = time.time()
        log.info("State: %s -> %s", self._prev_state, self._current)
        return True

    def tick(self, detected_state: str) -> bool:
        """
        根據偵測結果更新狀態。
        回傳 True 如果狀態有變化。
        """
        if detected_state and detected_state != self._current:
            return self.transition_to(detected_state)
        return False

    def is_timed_out(self) -> bool:
        """檢查目前狀態是否已超時"""
        state = self.get_state()
        if state and state.timeout > 0:
            return self.time_in_state > state.timeout
        return False

    def get_current_actions(self) -> List[Dict]:
        """取得目前狀態定義的動作列表"""
        state = self.get_state()
        if state:
            return list(state.actions)
        return []
