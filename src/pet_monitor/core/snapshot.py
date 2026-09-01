"""
报警快照 / Alert Snapshot

异常/关键报警发生时，把当前帧保存为 JPEG 留作证据，
并把路径写入数据库 snapshots 表，供报告与 GUI 快照画廊展示。

设计要点：
- 纯函数式，便于单测（不依赖 ultralytics / GUI）
- 文件名含时间戳 + 行为 + track_id，便于人工检索
- 带同源冷却，避免每一帧都写盘（默认 5 秒一次）
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from ..config import CONFIG, SNAPSHOT_DIR


class SnapshotManager:
    """管理报警快照的落盘与冷却。"""

    def __init__(self, out_dir: Path | None = None) -> None:
        self.out_dir = Path(out_dir) if out_dir else SNAPSHOT_DIR
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._last: dict[str, float] = {}
        self.cooldown_sec: float = 5.0

    def save(
        self,
        frame: np.ndarray,
        *,
        track_id: int | None = None,
        behavior: str = "anomaly",
        force: bool = False,
    ) -> Path | None:
        """保存一帧；冷却内且非强制时返回 None。"""
        if frame is None or frame.size == 0:
            return None
        key = f"{behavior}:{track_id}"
        now = time.time()
        if not force and key in self._last:
            if now - self._last[key] < self.cooldown_sec:
                return None
        self._last[key] = now

        ts = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        ms = int((now % 1) * 1000)
        fname = f"snap_{behavior}_{track_id or 'unk'}_{ts}_{ms:03d}.jpg"
        path = self.out_dir / fname
        try:
            # 注意：cv2.imwrite 在 Windows 含中文路径下会静默失败（返回 False），
            # 故改用 imencode + 标准文件写入，保证非 ASCII 路径可用。
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                return None
            with open(path, "wb") as f:
                f.write(buf.tobytes())
        except Exception:
            return None
        return path


# 单例
_default_snapshot: SnapshotManager | None = None


def get_snapshot() -> SnapshotManager:
    global _default_snapshot
    if _default_snapshot is None:
        _default_snapshot = SnapshotManager()
    return _default_snapshot


def list_snapshot_files(out_dir: Path | None = None) -> list[Path]:
    """按时间倒序返回快照文件路径列表。"""
    d = Path(out_dir) if out_dir else SNAPSHOT_DIR
    if not d.exists():
        return []
    return sorted(d.glob("snap_*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)


if __name__ == "__main__":
    # 自检：写一张纯色图
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    img[:] = (0, 0, 200)
    m = SnapshotManager()
    p = m.save(img, track_id=1, behavior="anomaly", force=True)
    print("saved:", p)
    print("files:", list_snapshot_files()[:3])
