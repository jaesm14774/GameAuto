"""Vision Client + Grid Overlay — LLM 視覺辨識"""

import base64
import json
import logging
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from dotenv import load_dotenv

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
        self.model = os.getenv("LLM_MODEL", vision_cfg.get("model", "gemini-3-flash-preview"))
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
            fixed = re.sub(r',\s*}', '}', content[start:end])
            fixed = re.sub(r',\s*]', ']', fixed)
            try:
                return json.loads(fixed)
            except json.JSONDecodeError:
                pass

        log.warning("Cannot parse LLM response as JSON: %s", raw[:300])
        return None
