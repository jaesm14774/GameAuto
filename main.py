"""GameAuto 主程式 — 狀態機 + 規則引擎 + LLM 兜底 的自動化迴圈"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import yaml

from core.agent import create_agent
from core.capture import GameCapture
from core.controller import GameController
from core.strategy import Action

log = logging.getLogger("GameAuto")


class GameAutoBot:
    """串接截圖、GameAgent（狀態機+規則+LLM）、操作控制的自動化引擎"""

    def __init__(self, config_path: str = "configs/elftw.yaml"):
        self.config = self._load_config(config_path)
        game_cfg = self.config.get("game", {})
        ctrl_cfg = self.config.get("controller", {})
        loop_cfg = self.config.get("loop", {})
        cap_cfg = self.config.get("capture", {})
        self.config_path = config_path

        # 截圖模組
        self.capture = GameCapture(
            window_title=game_cfg.get("window_title"),
            alt_titles=game_cfg.get("alt_titles", []),
            capture_mode=cap_cfg.get("mode", "window"),
            monitor_index=cap_cfg.get("monitor", 0),
        )

        # 操作模組（ctypes）
        self.controller = GameController(
            click_delay=tuple(ctrl_cfg.get("click_delay", [0.15, 0.4])),
            move_duration=tuple(ctrl_cfg.get("move_duration", [0.1, 0.25])),
        )

        # 決策引擎（狀態機 + 規則 + LLM 兜底）
        self.agent = create_agent(self.config)

        # 迴圈參數
        self.tick_interval = loop_cfg.get("tick_interval", 0.5)
        self.max_idle = loop_cfg.get("max_idle_seconds", 120)
        self.save_on_error = loop_cfg.get("screenshot_on_error", True)
        self._last_action_time = time.time()
        self._running = False

    @staticmethod
    def _load_config(path: str) -> dict:
        p = Path(path)
        if not p.exists():
            log.warning("Config not found: %s, using empty config", path)
            return {}
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    # ------------------------------------------------------------------
    # 主迴圈
    # ------------------------------------------------------------------
    def run(self):
        log.info("=== GameAuto starting ===")

        # 定位遊戲窗口
        cap_mode = self.config.get("capture", {}).get("mode", "window")
        if not self.capture.find_window():
            if cap_mode == "fullscreen":
                log.info("Window not found, using fullscreen capture mode")
            else:
                log.error("Cannot find game window.")
                sys.exit(1)

        self.capture.bring_to_front()
        region = self.capture.window_region
        if region:
            self.controller.update_offset(region[0], region[1])
            log.info("Window at (%d, %d) size %dx%d", *region)

        self._running = True
        log.info("Bot running. Ctrl+C to stop.")

        try:
            while self._running:
                self._tick()
                time.sleep(self.tick_interval)
        except KeyboardInterrupt:
            log.info("Stopped by user.")
        except Exception as e:
            log.error("Unexpected error: %s", e, exc_info=True)
            if self.save_on_error:
                self.capture.save_screenshot("screenshots/error.png")
        finally:
            self.agent.save_memory()

        log.info("=== GameAuto stopped ===")

    def discover(self):
        """角色掃描模式：掃描所有英雄角色，建立知識庫"""
        log.info("=== 角色掃描模式 ===")

        cap_mode = self.config.get("capture", {}).get("mode", "window")
        if not self.capture.find_window():
            if cap_mode == "fullscreen":
                log.info("Window not found, using fullscreen capture mode")
            else:
                log.error("Cannot find game window.")
                sys.exit(1)

        self.capture.bring_to_front()
        region = self.capture.window_region
        if region:
            self.controller.update_offset(region[0], region[1])
            log.info("Window at (%d, %d) size %dx%d", *region)

        time.sleep(1)

        # 執行角色掃描
        self.agent.discover_characters(self.capture, self.controller)
        log.info("=== 角色掃描完成 ===")

    def stop(self):
        self._running = False

    def _tick(self):
        """單次迴圈：截圖 → Agent 決策 → 執行動作"""
        # 檢查是否應該終止
        if self.agent.should_stop:
            log.error("程式終止: %s", self.agent.stop_reason)
            self._running = False
            return

        # 更新窗口位置
        region = self.capture.window_region
        if region:
            self.controller.update_offset(region[0], region[1])

        # 截圖
        frame = self.capture.grab()

        # Agent 決策（狀態機 + 規則 + LLM 兜底）
        actions = self.agent.tick(frame)

        # 執行動作
        if actions:
            self._last_action_time = time.time()
            for action in actions:
                self._execute(action, frame)

        # 閒置檢測
        idle = time.time() - self._last_action_time
        if idle > self.max_idle:
            log.warning("Idle for %.0fs, state=%s", idle, self.agent.current_phase)
            time.sleep(5)

    def _execute(self, action: Action, frame):
        """執行單一動作"""
        if action.type == "click" and (action.x or action.y):
            log.info("Click %s (%d, %d): %s",
                     action.grid or action.target, action.x, action.y, action.desc)
            self.controller.click(action.x, action.y)
            time.sleep(0.08)
            new_frame = self.capture.grab()
            self.agent.observe_result(new_frame, action, True)

        elif action.type == "wait":
            log.debug("Wait %.1fs: %s", action.seconds, action.desc)
            time.sleep(action.seconds)

        elif action.type == "skip":
            log.debug("Skip: %s", action.desc)

        elif action.type == "drag" and action.extra:
            to_x = action.extra.get("to_x", 0)
            to_y = action.extra.get("to_y", 0)
            log.info("Drag (%d,%d)->(%d,%d): %s",
                     action.x, action.y, to_x, to_y, action.desc)
            self.controller.drag(action.x, action.y, to_x, to_y)
            time.sleep(0.15)


# ======================================================================
# CLI
# ======================================================================
def calibrate(config_path: str):
    """校準模式：即時顯示滑鼠相對於遊戲窗口的座標"""
    import ctypes
    import ctypes.wintypes

    config = GameAutoBot._load_config(config_path)
    game_cfg = config.get("game", {})
    cap_cfg = config.get("capture", {})

    capture = GameCapture(
        window_title=game_cfg.get("window_title"),
        alt_titles=game_cfg.get("alt_titles", []),
        capture_mode=cap_cfg.get("mode", "window"),
        monitor_index=cap_cfg.get("monitor", 0),
    )

    if not capture.find_window():
        print("找不到遊戲窗口！請先啟動雷電模擬器。")
        sys.exit(1)

    capture.bring_to_front()
    region = capture.window_region
    wx, wy = (region[0], region[1]) if region else (0, 0)

    print("=" * 50)
    print("  校準模式 — 把滑鼠移到按鈕上，記下 Game 座標")
    print("  然後填入 configs/elftw.yaml 的 positions 區段")
    print("  Ctrl+C 結束")
    print("=" * 50)
    print()

    try:
        while True:
            # 更新窗口位置
            region = capture.window_region
            if region:
                wx, wy = region[0], region[1]

            pt = ctypes.wintypes.POINT()
            ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
            rx, ry = pt.x - wx, pt.y - wy
            print(f"\r  Screen: ({pt.x:5d}, {pt.y:5d})  |  Game: ({rx:5d}, {ry:5d})  ", end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n校準結束。")


def main():
    parser = argparse.ArgumentParser(description="GameAuto - Game Automation Bot")
    parser.add_argument("-c", "--config", default="configs/elftw.yaml",
                        help="Path to game config YAML")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--calibrate", action="store_true",
                        help="校準模式：顯示滑鼠相對遊戲窗口的座標")
    parser.add_argument("--discover", action="store_true",
                        help="角色掃描模式：掃描英雄圖鑑建立知識庫")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.calibrate:
        calibrate(args.config)
        return

    os.makedirs("screenshots", exist_ok=True)

    bot = GameAutoBot(config_path=args.config)

    if args.discover:
        bot.discover()
        return

    bot.run()


if __name__ == "__main__":
    main()
