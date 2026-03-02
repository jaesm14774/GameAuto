"""GameAgent — 通用決策引擎基底類"""

import importlib
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from core.memory import AgentMemory
from core.screen_detector import ScreenChangeDetector
from core.state_machine import StateMachine
from core.strategy import Action, GameStrategy, create_strategy
from core.vision import VisionClient

log = logging.getLogger("agent")


class GameAgent:
    """
    通用決策引擎基底類：整合狀態機 + 規則引擎 + LLM 兜底。

    流程：
    1. 用 VisionClient 偵測目前狀態
    2. 用 GameStrategy (規則引擎) 產生動作
    3. 如果規則無法處理，用 VisionClient (LLM) 兜底
    4. 子類可覆寫 resolve_action() 處理遊戲特定動作類型
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

        self._last_tick_time = 0.0
        self._force_llm_next_tick = False

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
        return False

    @property
    def stop_reason(self) -> str:
        return ""

    # States where we skip LLM detection and just execute strategy actions
    _FAST_ACTION_STATES = {"preparation"}
    # How often to call LLM for state verification in fast-action states (seconds)
    _FAST_STATE_LLM_INTERVAL = 15.0
    # Force a tick if idle for this many seconds
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

        # Force a tick if idle for too long
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
            new_state = self.state_machine.current_state
            if self._force_llm_next_tick and new_state != "preparation":
                self._force_llm_next_tick = False
                log.info("準備→%s 轉場偵測成功，恢復正常節流", new_state)
        else:
            log.debug("Fast tick: skipping LLM in %s state", current)

        current = self.state_machine.current_state

        # Hook for subclasses to run game-specific state tracking
        self._post_state_update(current, frame)

        # Build context
        context: Dict[str, Any] = {
            "timeout": self.state_machine.is_timed_out(),
            "time_in_state": self.state_machine.time_in_state,
        }

        # Try rules first
        actions = self.strategy.on_state(current, context)

        # Process actions — resolve LLM fallbacks and game-specific types
        resolved: List[Action] = []
        for action in actions:
            result = self.resolve_action(action, frame)
            if result is not None:
                resolved.extend(result)
            elif action.type == "llm_fallback":
                llm_actions = self._llm_decide(frame, action.desc)
                resolved.extend(llm_actions)
            elif action.type == "click" and not action.grid and not action.x:
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

        # Hook for subclasses to run post-tick logic
        self._post_tick(current, resolved, frame)

        return resolved

    def resolve_action(self, action: Action, frame: np.ndarray) -> Optional[List[Action]]:
        """
        Resolve a game-specific action type.

        Subclasses override this to handle custom action types
        (e.g., llm_smart_prep, llm_position, llm_relic_select).

        Returns:
            List of resolved actions, or None if this action type is not handled
            (falls through to default handling in tick()).
        """
        return None

    def _post_state_update(self, current_state: str, frame: np.ndarray):
        """Hook called after state detection, before action resolution.
        Subclasses override for game-specific state tracking logic."""
        pass

    def _post_tick(self, current_state: str, resolved: List[Action],
                   frame: np.ndarray):
        """Hook called after actions are resolved.
        Subclasses override for post-tick logic (e.g., force LLM flags)."""
        pass

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


# ======================================================================
# Agent 工廠
# ======================================================================
def create_agent(config: Dict[str, Any]) -> GameAgent:
    """根據遊戲名稱建立對應的 Agent，從 games/ registry 自動發現"""
    from games import GAME_REGISTRY

    game_name = config.get("game", {}).get("name", "")
    entry = GAME_REGISTRY.get(game_name)

    if entry is None:
        log.warning("No agent registered for '%s', trying elftw as default", game_name)
        entry = GAME_REGISTRY.get("elftw")

    if entry is None:
        log.warning("No game registry entry found, using base GameAgent")
        return GameAgent(config)

    module_path, class_name = entry["agent_cls"].rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)
