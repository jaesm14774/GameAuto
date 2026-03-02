"""Screen Change Detector — 偵測畫面是否有顯著變化"""

import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("agent")


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
