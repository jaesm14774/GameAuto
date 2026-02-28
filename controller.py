"""操作執行模組 — ctypes 直接操控滑鼠，原生支援多螢幕負座標"""

import ctypes
import ctypes.wintypes
import logging
import random
import time
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# Windows API constants
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000

INPUT_MOUSE = 0
INPUT_KEYBOARD = 1

VK_MAP = {
    "enter": 0x0D, "return": 0x0D,
    "escape": 0x1B, "esc": 0x1B,
    "tab": 0x09,
    "space": 0x20,
    "backspace": 0x08,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
}

KEYEVENTF_KEYUP = 0x0002


# ======================================================================
# ctypes structures
# ======================================================================
class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.wintypes.LONG),
        ("dy", ctypes.wintypes.LONG),
        ("mouseData", ctypes.wintypes.DWORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.wintypes.WORD),
        ("wScan", ctypes.wintypes.WORD),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("time", ctypes.wintypes.DWORD),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", MOUSEINPUT),
        ("ki", KEYBDINPUT),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.wintypes.DWORD),
        ("union", INPUT_UNION),
    ]


# ======================================================================
# Virtual desktop metrics for absolute mouse coordinates
# ======================================================================
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79

user32 = ctypes.windll.user32


def _screen_to_absolute(sx: int, sy: int) -> Tuple[int, int]:
    """
    Convert screen pixel coordinates to normalized absolute coordinates
    (0-65535 range) across the virtual desktop. Handles negative coordinates
    from multi-monitor setups.
    """
    vx = user32.GetSystemMetrics(SM_XVIRTUALSCREEN)
    vy = user32.GetSystemMetrics(SM_YVIRTUALSCREEN)
    vw = user32.GetSystemMetrics(SM_CXVIRTUALSCREEN)
    vh = user32.GetSystemMetrics(SM_CYVIRTUALSCREEN)

    # Normalize to 0-65535 range relative to virtual desktop
    abs_x = int(((sx - vx) * 65535) / (vw - 1))
    abs_y = int(((sy - vy) * 65535) / (vh - 1))
    return abs_x, abs_y


def _send_mouse_input(flags: int, dx: int = 0, dy: int = 0):
    """Send a raw mouse input event via SendInput."""
    mi = MOUSEINPUT(
        dx=dx, dy=dy, mouseData=0, dwFlags=flags,
        time=0, dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)),
    )
    inp = INPUT(type=INPUT_MOUSE)
    inp.union.mi = mi
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


def _send_key_input(vk: int, up: bool = False):
    """Send a raw keyboard input event via SendInput."""
    flags = KEYEVENTF_KEYUP if up else 0
    ki = KEYBDINPUT(
        wVk=vk, wScan=0, dwFlags=flags,
        time=0, dwExtraInfo=ctypes.pointer(ctypes.c_ulong(0)),
    )
    inp = INPUT(type=INPUT_KEYBOARD)
    inp.union.ki = ki
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(INPUT))


# ======================================================================
# GameController
# ======================================================================
class GameController:
    """負責將座標轉成實際滑鼠/鍵盤操作（ctypes 實作）"""

    def __init__(
        self,
        window_offset: Tuple[int, int] = (0, 0),
        click_delay: Tuple[float, float] = (0.15, 0.4),
        move_duration: Tuple[float, float] = (0.1, 0.25),
    ):
        self.offset_x, self.offset_y = window_offset
        self.click_delay = click_delay
        self.move_duration = move_duration

    def update_offset(self, x: int, y: int):
        """更新窗口偏移"""
        self.offset_x = x
        self.offset_y = y

    # ------------------------------------------------------------------
    # 基本操作
    # ------------------------------------------------------------------
    def click(self, x: int, y: int, delay: Optional[float] = None):
        """
        點擊遊戲內座標 (相對於遊戲窗口)。
        自動加上窗口偏移 + 隨機微偏移。
        """
        sx, sy = self._to_screen(x, y)
        self._move_to(sx, sy)
        time.sleep(random.uniform(0.02, 0.08))

        # Click
        abs_x, abs_y = _screen_to_absolute(sx, sy)
        flags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        _send_mouse_input(flags | MOUSEEVENTF_LEFTDOWN, abs_x, abs_y)
        time.sleep(random.uniform(0.01, 0.04))
        _send_mouse_input(flags | MOUSEEVENTF_LEFTUP, abs_x, abs_y)

        wait = delay if delay is not None else random.uniform(*self.click_delay)
        time.sleep(wait)
        log.debug("click (%d, %d) -> screen (%d, %d)", x, y, sx, sy)

    def double_click(self, x: int, y: int):
        """雙擊"""
        self.click(x, y, delay=random.uniform(0.05, 0.1))
        sx, sy = self._to_screen(x, y)
        abs_x, abs_y = _screen_to_absolute(sx, sy)
        flags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        _send_mouse_input(flags | MOUSEEVENTF_LEFTDOWN, abs_x, abs_y)
        time.sleep(random.uniform(0.01, 0.04))
        _send_mouse_input(flags | MOUSEEVENTF_LEFTUP, abs_x, abs_y)
        time.sleep(random.uniform(*self.click_delay))

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.4):
        """從 (x1,y1) 拖拽到 (x2,y2)"""
        sx1, sy1 = self._to_screen(x1, y1)
        sx2, sy2 = self._to_screen(x2, y2)

        # Move to start
        self._move_to(sx1, sy1)
        time.sleep(random.uniform(0.02, 0.05))

        # Press down
        abs_x, abs_y = _screen_to_absolute(sx1, sy1)
        flags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK
        _send_mouse_input(flags | MOUSEEVENTF_LEFTDOWN, abs_x, abs_y)

        # Interpolate movement
        steps = max(5, int(duration / 0.02))
        for i in range(1, steps + 1):
            t = i / steps
            cx = int(sx1 + (sx2 - sx1) * t)
            cy = int(sy1 + (sy2 - sy1) * t)
            ax, ay = _screen_to_absolute(cx, cy)
            _send_mouse_input(flags | MOUSEEVENTF_MOVE, ax, ay)
            time.sleep(duration / steps)

        # Release
        abs_x2, abs_y2 = _screen_to_absolute(sx2, sy2)
        _send_mouse_input(flags | MOUSEEVENTF_LEFTUP, abs_x2, abs_y2)
        time.sleep(random.uniform(*self.click_delay))
        log.debug("drag (%d,%d)->(%d,%d)", x1, y1, x2, y2)

    def press_key(self, key: str, delay: float = 0.2):
        """按下鍵盤按鍵"""
        vk = VK_MAP.get(key.lower())
        if vk is None:
            # Single character key
            if len(key) == 1:
                vk = ord(key.upper())
            else:
                log.warning("Unknown key: %s", key)
                return
        _send_key_input(vk, up=False)
        time.sleep(random.uniform(0.02, 0.06))
        _send_key_input(vk, up=True)
        time.sleep(delay)

    # ------------------------------------------------------------------
    # 組合操作
    # ------------------------------------------------------------------
    def click_sequence(self, points: list, interval: Optional[float] = None):
        """依序點擊多個座標。points: [(x, y), ...]"""
        for x, y in points:
            self.click(x, y, delay=interval)

    def click_grid(self, grid_ref: str, frame_w: int, frame_h: int,
                   cols: int = 20, rows: int = 10, delay: Optional[float] = None):
        """點擊網格座標 (如 "C7")，自動轉換為像素座標"""
        grid_ref = grid_ref.strip().upper()
        col = ord(grid_ref[0]) - ord('A')
        row = int(grid_ref[1:]) - 1
        col = max(0, min(col, cols - 1))
        row = max(0, min(row, rows - 1))
        cw, ch = frame_w / cols, frame_h / rows
        cx = int(col * cw + cw / 2)
        cy = int(row * ch + ch / 2)
        self.click(cx, cy, delay=delay)

    # ------------------------------------------------------------------
    # 內部工具
    # ------------------------------------------------------------------
    def _to_screen(self, x: int, y: int) -> Tuple[int, int]:
        """遊戲座標 -> 螢幕座標，加入隨機微偏移"""
        jitter_x = random.randint(-2, 2)
        jitter_y = random.randint(-2, 2)
        return (self.offset_x + x + jitter_x, self.offset_y + y + jitter_y)

    def _move_to(self, sx: int, sy: int):
        """平滑移動滑鼠到螢幕座標"""
        # Get current cursor position
        pt = ctypes.wintypes.POINT()
        user32.GetCursorPos(ctypes.byref(pt))
        cur_x, cur_y = pt.x, pt.y

        duration = random.uniform(*self.move_duration)
        steps = max(3, int(duration / 0.01))
        flags = MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK | MOUSEEVENTF_MOVE

        for i in range(1, steps + 1):
            t = i / steps
            # Ease-in-out interpolation
            t = t * t * (3 - 2 * t)
            cx = int(cur_x + (sx - cur_x) * t)
            cy = int(cur_y + (sy - cur_y) * t)
            ax, ay = _screen_to_absolute(cx, cy)
            _send_mouse_input(flags, ax, ay)
            time.sleep(duration / steps)

    @staticmethod
    def wait(seconds: float):
        """等待（可加入人類化隨機）"""
        actual = seconds + random.uniform(-0.05, 0.1)
        if actual > 0:
            time.sleep(actual)
