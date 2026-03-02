"""Agent Memory — 儲存/載入遊戲經驗"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

log = logging.getLogger("agent")


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
