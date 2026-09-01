"""
报警系统

- AlertSystem.raise(level, message, ...)    触发报警
  自动写入数据库；调用系统蜂鸣；可选弹窗
- 同类报警有冷却时间，避免刷屏
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from ..config import CONFIG
from .database import get_db


LEVEL_INFO = "info"
LEVEL_WARN = "warning"
LEVEL_CRIT = "critical"

LEVEL_PRIORITY = {LEVEL_INFO: 0, LEVEL_WARN: 1, LEVEL_CRIT: 2}


@dataclass
class AlertRecord:
    level: str
    message: str
    ts: float
    track_id: int | None = None
    behavior: str | None = None


class AlertSystem:
    """集中管理报警触发与冷却。"""

    def __init__(self) -> None:
        self.cfg = CONFIG.alert
        self._last: dict[str, float] = {}
        self._history: deque = deque(maxlen=200)
        self._popup_cb: Callable[[AlertRecord], None] | None = None

    def set_popup_callback(self, cb: Callable[[AlertRecord], None]) -> None:
        """UI 可注册一个回调，弹出系统通知。"""
        self._popup_cb = cb

    def raise_(
        self,
        level: str,
        message: str,
        *,
        track_id: int | None = None,
        behavior: str | None = None,
        cooldown_key: str | None = None,
        force: bool = False,
    ) -> AlertRecord | None:
        ts = time.time()
        key = cooldown_key or f"{level}:{behavior or '-'}:{track_id or '-'}"
        if not force and key in self._last:
            if ts - self._last[key] < self.cfg.cooldown_sec:
                return None
        self._last[key] = ts

        rec = AlertRecord(
            level=level, message=message, ts=ts,
            track_id=track_id, behavior=behavior,
        )
        self._history.append(rec)

        if self.cfg.log_to_db:
            try:
                get_db().insert_alert(
                    level=level, message=message,
                    track_id=track_id, behavior=behavior, ts=ts,
                )
            except Exception:
                pass

        if self.cfg.enable_sound:
            self._beep(level)

        if self.cfg.enable_desktop_popup and self._popup_cb:
            try:
                self._popup_cb(rec)
            except Exception:
                pass
        return rec

    # ---------------------------------------------------------- sound
    @staticmethod
    def _beep(level: str) -> None:
        """跨平台蜂鸣。失败也不抛。"""
        try:
            import winsound  # type: ignore
            freq = {
                LEVEL_INFO: 800,
                LEVEL_WARN: 1000,
                LEVEL_CRIT: 1500,
            }.get(level, 800)
            dur = 150 if level == LEVEL_INFO else 300
            winsound.Beep(freq, dur)
        except Exception:
            try:
                # 退化方案：终端响铃
                print("\a", end="", flush=True)
            except Exception:
                pass

    # ---------------------------------------------------------- access
    def history(self, n: int = 20) -> list[AlertRecord]:
        return list(self._history)[-n:]


# 单例
_default_alert: AlertSystem | None = None


def get_alert() -> AlertSystem:
    global _default_alert
    if _default_alert is None:
        _default_alert = AlertSystem()
    return _default_alert


if __name__ == "__main__":
    a = AlertSystem()
    a.raise_(LEVEL_WARN, "测试报警", cooldown_key="test")
    a.raise_(LEVEL_WARN, "测试报警", cooldown_key="test")
    print("history:", a.history())