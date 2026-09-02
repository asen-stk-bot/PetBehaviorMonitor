"""
监控引擎

将 detector + behavior_analyzer + alert + database 串成一条数据通路。

公开 API：
    engine = Monitor(source=0)
    engine.start(callback=on_frame_result)
    engine.stop()
    engine.wait(timeout=...)
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

import cv2
import numpy as np

from .alert import LEVEL_CRIT, LEVEL_INFO, LEVEL_WARN, get_alert
from .behavior import BEHAVIOR_LABELS_CN, BehaviorAnalyzer, BehaviorOutput
from .database import get_db
from .detector import Detection, PetDetector
from .snapshot import get_snapshot
from ..config import CONFIG


log = logging.getLogger("pet_monitor.monitor")


# --------------------------------------------------------------------------- #
# 帧结果
# --------------------------------------------------------------------------- #
@dataclass
class FrameResult:
    """单帧产出物，给到 UI 直接绘制。"""
    frame: np.ndarray                       # 绘制了 bbox + 行为标签
    raw_frame: np.ndarray                   # 原始帧
    detections: list[Detection]
    behaviors: list[BehaviorOutput]
    fps: float = 0.0


# --------------------------------------------------------------------------- #
# 引擎
# --------------------------------------------------------------------------- #
class Monitor:
    """单源（摄像头 / 视频文件）监控引擎。

    内部维护一个后台线程，从 cv2.VideoCapture 读帧，调用 detector + analyzer，
    把 FrameResult 通过 callback 推给 UI；UI 主线程直接绘制。
    """

    def __init__(
        self,
        source: int | str = 0,
        on_result: Callable[[FrameResult], None] | None = None,
    ) -> None:
        self.source = source
        self.on_result = on_result
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._detector = PetDetector()
        self._analyzer = BehaviorAnalyzer()
        self._alert = get_alert()
        self._db = get_db()
        self._snapshot = get_snapshot()
        self._behavior_prev: dict[int, str] = {}
        self._frame_count = 0
        self._total_frames = 0
        self._t_last = time.time()
        self._fps = 0.0
        self._cap: cv2.VideoCapture | None = None

    # ---------------------------------------------------------- 控制
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="Monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ---------------------------------------------------------- 主循环
    def _run(self) -> None:
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            log.error("无法打开视频源: %s", self.source)
            self._alert.raise_(
                LEVEL_CRIT,
                f"无法打开视频源: {self.source}",
                cooldown_key="open_source",
            )
            return
        self._cap = cap
        source_label = str(self.source)
        log.info("监控启动: source=%s", self.source)

        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    # 视频结束或读帧失败
                    if isinstance(self.source, str):
                        break
                    continue
                # 计数器由 _process_frame 内部累加
                result = self._process_frame(frame, source_label)
                if self.on_result:
                    try:
                        self.on_result(result)
                    except Exception as e:
                        log.warning("UI 回调异常: %s", e)
        finally:
            cap.release()
            log.info("监控停止: source=%s, frames=%d", self.source, self._frame_count)

    # ---------------------------------------------------------- 单帧处理
    def _process_frame(self, frame: np.ndarray, source: str) -> FrameResult:
        t0 = time.time()
        detections = self._detector.detect(frame)
        h, w = frame.shape[:2]
        # 传入 frame 供 BehaviorAnalyzer 计算光流（若开启）
        outs = self._analyzer.update(detections, frame_size=(w, h), ts=t0, frame=frame)

        self._frame_count += 1
        self._total_frames += 1

        # 写数据库 / 触发报警
        for det, beh in zip(detections, outs):
            try:
                self._db.insert_event(
                    ts=t0,
                    track_id=det.track_id,
                    species=det.species,
                    behavior=beh.behavior,
                    confidence=det.confidence,
                    bbox=det.bbox,
                    source=source,
                )
            except Exception as e:
                log.debug("DB insert_event 失败: %s", e)
            self._maybe_alert(det, beh, frame)

        # 在帧上绘制
        drawn = self._draw(frame, detections, outs)

        # 帧率（基于滑动窗口）
        if t0 - self._t_last >= 1.0:
            self._fps = self._frame_count / (t0 - self._t_last)
            self._frame_count = 0
            self._t_last = t0

        return FrameResult(
            frame=drawn,
            raw_frame=frame,
            detections=detections,
            behaviors=outs,
            fps=self._fps,
        )

    # ---------------------------------------------------------- 报警
    def _maybe_alert(self, det: Detection, beh: BehaviorOutput,
                     frame: np.ndarray) -> None:
        prev = self._behavior_prev.get(det.track_id)
        if prev == beh.behavior:
            return
        # 状态变化时按级别触发
        if beh.behavior == "anomaly":
            self._alert.raise_(
                LEVEL_CRIT,
                f"检测到异常行为（疑似晕厥/抽搐） track#{det.track_id} {det.species}",
                track_id=det.track_id,
                behavior=beh.behavior,
                cooldown_key=f"anomaly:{det.track_id}",
            )
            # 关键报警：保存快照留证
            self._capture_snapshot(frame, det, beh, force=True)
        elif beh.behavior == "eating":
            self._alert.raise_(
                LEVEL_INFO,
                f"{det.species} #{det.track_id} 正在进食",
                track_id=det.track_id,
                behavior=beh.behavior,
                cooldown_key=f"eating:{det.track_id}",
            )
            self._capture_snapshot(frame, det, beh)
        elif beh.behavior == "drinking":
            self._alert.raise_(
                LEVEL_INFO,
                f"{det.species} #{det.track_id} 正在饮水",
                track_id=det.track_id,
                behavior=beh.behavior,
                cooldown_key=f"drinking:{det.track_id}",
            )
            self._capture_snapshot(frame, det, beh)
        self._behavior_prev[det.track_id] = beh.behavior

    def _capture_snapshot(self, frame: np.ndarray, det: Detection,
                          beh: BehaviorOutput, force: bool = False) -> None:
        """保存报警快照到磁盘并写入数据库（失败不影响主流程）。"""
        try:
            path = self._snapshot.save(
                frame, track_id=det.track_id, behavior=beh.behavior, force=force,
            )
            if path is None:
                return
            self._db.insert_snapshot(
                ts=time.time(), track_id=det.track_id,
                behavior=beh.behavior, path=str(path),
            )
        except Exception as e:
            log.debug("快照保存失败: %s", e)

    # ---------------------------------------------------------- 绘制
    @staticmethod
    def _draw(
        frame: np.ndarray,
        detections: list[Detection],
        outs: list[BehaviorOutput],
    ) -> np.ndarray:
        out = frame.copy()
        beh_map = {b.track_id: b for b in outs}
        for det in detections:
            beh = beh_map.get(det.track_id)
            label_cn = BEHAVIOR_LABELS_CN.get(beh.behavior if beh else "unknown", "?")
            color = {
                "cat": (255, 200, 100),
                "dog": (100, 200, 255),
            }.get(det.species, (200, 200, 200))
            x1, y1, x2, y2 = map(int, det.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                f"{det.species}#{det.track_id} {label_cn} {det.confidence:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
            )
            # anomaly 加粗红框
            if beh and beh.is_anomaly:
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 4)
        return out