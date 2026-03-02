"""截圖擷取模組 — 支援 DXCam (DirectX) / MSS / PyAutoGUI 多種擷取方式"""

import logging
import time
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import win32gui
    import win32con
    HAS_WIN32 = True
except ImportError:
    HAS_WIN32 = False

log = logging.getLogger(__name__)


class GameCapture:
    """負責定位遊戲窗口並擷取截圖，自動選擇最佳擷取方式"""

    DEFAULT_TITLE_KEYWORDS = ["LDPlayerMainFrame", "雷電模擬器", "LDPlayer"]

    def __init__(
        self,
        window_title: Optional[str] = None,
        alt_titles: Optional[list] = None,
        capture_mode: str = "window",
        monitor_index: int = 0,
    ):
        self.hwnd: Optional[int] = None
        self.window_title = window_title
        self.alt_titles = alt_titles or []
        self.capture_mode = capture_mode
        self.monitor_index = monitor_index
        self._region: Optional[Tuple[int, int, int, int]] = None
        self._dxcam = None
        self._dxcam_device_idx = 0
        self._dxcam_output_idx = 0
        self._capture_method = None  # "dxcam" | "mss" | "pyautogui"

    def _init_dxcam(self):
        """初始化 DXCam（DirectX 擷取，適合遊戲）"""
        try:
            import dxcam
            # 偵測哪個輸出（螢幕）有遊戲
            self._dxcam = dxcam.create(
                device_idx=self._dxcam_device_idx,
                output_idx=self.monitor_index,
            )
            # 測試擷取
            test = self._dxcam.grab()
            if test is not None and test.max() > 0:
                self._capture_method = "dxcam"
                log.info("DXCam initialized: output=%d", self.monitor_index)
                return True
            # 測試另一個螢幕
            self._dxcam.release()
            other = 1 - self.monitor_index
            self._dxcam = dxcam.create(device_idx=0, output_idx=other)
            test = self._dxcam.grab()
            if test is not None and test.max() > 0:
                self.monitor_index = other
                self._capture_method = "dxcam"
                log.info("DXCam initialized: output=%d (auto-detected)", other)
                return True
            self._dxcam.release()
            self._dxcam = None
        except Exception as e:
            log.debug("DXCam init failed: %s", e)
            self._dxcam = None
        return False

    def _init_capture(self):
        """自動選擇可用的擷取方式"""
        if self._capture_method:
            return

        # 優先 MSS（多螢幕支援穩定）
        try:
            import mss
            with mss.mss() as sct:
                idx = min(self.monitor_index + 1, len(sct.monitors) - 1)
                test = np.array(sct.grab(sct.monitors[idx]))
                if test.max() > 0:
                    self._capture_method = "mss"
                    log.info("Using MSS capture: monitor=%d (%dx%d)",
                             self.monitor_index,
                             sct.monitors[idx]["width"],
                             sct.monitors[idx]["height"])
                    return
        except Exception as e:
            log.debug("MSS init failed: %s", e)

        # 備選 DXCam（單螢幕遊戲擷取）
        if self._init_dxcam():
            return

        # 最後 PyAutoGUI
        self._capture_method = "pyautogui"
        log.info("Using PyAutoGUI capture (may not work with fullscreen games)")

    # ------------------------------------------------------------------
    # 窗口搜尋
    # ------------------------------------------------------------------
    def find_window(self, title: Optional[str] = None) -> bool:
        """透過標題關鍵字搜尋遊戲窗口"""
        if not HAS_WIN32:
            log.warning("win32gui unavailable")
            return False

        keywords = [title] if title else self.DEFAULT_TITLE_KEYWORDS
        if self.window_title:
            keywords.insert(0, self.window_title)
        keywords.extend(self.alt_titles)

        result = []

        def _enum_cb(hwnd, _):
            if win32gui.IsWindowVisible(hwnd):
                txt = win32gui.GetWindowText(hwnd)
                for kw in keywords:
                    if kw and kw.lower() in txt.lower():
                        result.append(hwnd)

        win32gui.EnumWindows(_enum_cb, None)

        if result:
            self.hwnd = result[0]
            self._update_region()
            log.info("Found window: hwnd=%s title='%s'",
                     self.hwnd, win32gui.GetWindowText(self.hwnd))
            return True

        log.warning("Window not found for keywords=%s", keywords)
        return False

    def _update_region(self):
        if not self.hwnd:
            return
        rect = win32gui.GetWindowRect(self.hwnd)
        self._region = (rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1])

    # ------------------------------------------------------------------
    # 截圖
    # ------------------------------------------------------------------
    def grab(self, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """擷取遊戲畫面，回傳 BGR numpy array"""
        self._init_capture()

        if self._capture_method == "dxcam":
            return self._grab_dxcam(region)
        elif self._capture_method == "mss":
            return self._grab_mss(region)
        else:
            return self._grab_pyautogui(region)

    def _grab_dxcam(self, region) -> np.ndarray:
        """DXCam 擷取（最佳遊戲擷取方式）"""
        frame = self._dxcam.grab()
        if frame is None:
            time.sleep(0.05)
            frame = self._dxcam.grab()
        if frame is None:
            log.warning("DXCam grab returned None, falling back")
            return self._grab_mss(region)

        # DXCam 回傳 RGB，轉 BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if region:
            rx, ry, rw, rh = region
            frame = frame[ry:ry+rh, rx:rx+rw]
        return frame

    def _grab_mss(self, region) -> np.ndarray:
        """MSS 擷取"""
        import mss
        with mss.mss() as sct:
            idx = min(self.monitor_index + 1, len(sct.monitors) - 1)
            mon = sct.monitors[idx]
            if region:
                rx, ry, rw, rh = region
                mon = {"left": mon["left"] + rx, "top": mon["top"] + ry,
                       "width": rw, "height": rh}
            img = np.array(sct.grab(mon))
            # MSS 回傳 BGRA，轉 BGR
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    def _grab_pyautogui(self, region) -> np.ndarray:
        """PyAutoGUI 擷取（不支援 DX 全螢幕）"""
        import pyautogui
        if self.hwnd:
            self._update_region()
        abs_region = self._to_absolute(region)
        img = pyautogui.screenshot(region=abs_region)
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _to_absolute(self, region):
        if self._region is None:
            return region
        wx, wy, _, _ = self._region
        if region is None:
            return self._region
        rx, ry, rw, rh = region
        return (wx + rx, wy + ry, rw, rh)

    # ------------------------------------------------------------------
    # 輔助
    # ------------------------------------------------------------------
    @property
    def window_region(self) -> Optional[Tuple[int, int, int, int]]:
        if self.hwnd:
            self._update_region()
        return self._region

    def bring_to_front(self):
        if self.hwnd and HAS_WIN32:
            try:
                win32gui.SetForegroundWindow(self.hwnd)
                time.sleep(0.3)
            except Exception:
                log.debug("SetForegroundWindow failed, continuing")

    def save_screenshot(self, path: str, region=None):
        frame = self.grab(region)
        cv2.imwrite(path, frame)
        log.info("Screenshot saved: %s (%dx%d)", path, frame.shape[1], frame.shape[0])

    def release(self):
        """釋放 DXCam 資源"""
        if self._dxcam:
            try:
                self._dxcam.release()
            except Exception:
                pass
            self._dxcam = None
