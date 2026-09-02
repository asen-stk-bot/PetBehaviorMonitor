"""
行为识别 / Behavior Recognition

策略（无监督 / 启发式）：

定义 5 类行为：
  - resting    休息 / 睡眠：长时间位移极小
  - active     活动 / 玩耍：位移明显、有变化
  - eating     进食：在 food ROI 内停留
  - drinking   饮水：在 water ROI 内停留
  - anomaly    异常：超长静止 + 微抖动（疑似晕厥/抽搐），或彻底消失

每个目标（track_id）维护一个 BehaviorState，内部累计时间窗，
由 update(detections, frame_size, ts) 调用产生 BehaviorOutput。

输出结果同时落库（database.insert_event + insert_behavior_summary），
并触发 alert（alert.AlertSystem）。
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .detector import Detection
from ..config import CONFIG


# --------------------------------------------------------------------------- #
# 输出
# --------------------------------------------------------------------------- #
@dataclass
class BehaviorOutput:
    """单帧单目标行为识别结果。"""
    track_id: int
    species: str
    behavior: str
    confidence: float = 0.0
    duration_in_state_sec: float = 0.0
    is_anomaly: bool = False


BEHAVIOR_LABELS_CN = {
    "resting":  "休息",
    "active":   "活动",
    "eating":   "进食",
    "drinking": "饮水",
    "anomaly":  "异常",
    "unknown":  "未知",
}


# --------------------------------------------------------------------------- #
# 单目标状态
# --------------------------------------------------------------------------- #
@dataclass
class _TrackState:
    track_id: int
    species: str
    history: deque = field(default_factory=lambda: deque(maxlen=300))
    """每个元素: (ts, center, bbox)"""

    current_behavior: str = "unknown"
    state_start_ts: float = field(default_factory=time.time)
    last_update_ts: float = field(default_factory=time.time)

    # 累计本状态的时长（用于 anomaly 判定）
    resting_total_sec: float = 0.0
    last_emit_ts: float = field(default_factory=time.time)

    def push(self, ts: float, center: tuple[float, float],
             bbox: tuple[float, float, float, float]) -> None:
        self.history.append((ts, center, bbox))
        self.last_update_ts = ts


# --------------------------------------------------------------------------- #
# 行为分析器
# --------------------------------------------------------------------------- #
class BehaviorAnalyzer:
    """对单帧的所有 Detection 输出 BehaviorOutput 列表。"""

    def __init__(self) -> None:
        self.cfg = CONFIG.behavior
        self.states: dict[int, _TrackState] = {}
        # 光流辅助：缓存上一帧灰度，计算全局运动幅值
        self._prev_gray: "np.ndarray | None" = None
        self._flow_mag: float = 0.0

    # ---------------------------------------------------------- helpers
    def _compute_flow(self, frame: np.ndarray) -> float:
        """Farneback 稀疏光流平均幅值（降采样加速）。

        光流衡量画面整体像素运动，独立于 bbox 中心抖动：
        - 动物整体在动但检测框位置漂移时，光流能给出"确实在动"的证据；
        - 动物静止、仅检测框抖动时，光流接近 0，可强化 resting 判定。
        """
        import cv2
        h, w = frame.shape[:2]
        target_w = self.cfg.flow_downscale
        if w != target_w:
            scale = target_w / w
            small = cv2.resize(frame, (target_w, max(1, int(h * scale))))
        else:
            small = frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.1, 0,
        )
        self._prev_gray = gray
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        return float(mag.mean())
    @staticmethod
    def _motion_magnitude(history: deque) -> float:
        """最近 history 中两两相邻帧中心点平均像素位移。"""
        if len(history) < 2:
            return 0.0
        centers = np.array([h[1] for h in history], dtype=np.float32)
        diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        return float(np.mean(diffs))

    @staticmethod
    def _jitter_magnitude(history: deque) -> float:
        """最近抖动幅度（标准差），区别于平均位移。"""
        if len(history) < 2:
            return 0.0
        centers = np.array([h[1] for h in history], dtype=np.float32)
        diffs = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        return float(np.std(diffs))

    @staticmethod
    def _aspect(bbox: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = bbox
        w, h = max(1e-3, x2 - x1), max(1e-3, y2 - y1)
        return float(w / h)

    @staticmethod
    def _in_roi(
        center: tuple[float, float],
        roi_norm: tuple[float, float, float, float],
        frame_size: tuple[int, int],
    ) -> bool:
        cx, cy = center
        fw, fh = frame_size
        rx1, ry1, rx2, ry2 = roi_norm
        return (
            rx1 * fw <= cx <= rx2 * fw
            and ry1 * fh <= cy <= ry2 * fh
        )

    # ---------------------------------------------------------- public
    def update(
        self,
        detections: Iterable[Detection],
        frame_size: tuple[int, int],
        ts: float | None = None,
        frame: np.ndarray | None = None,
    ) -> list[BehaviorOutput]:
        """对单帧的检测结果进行更新。

        Args:
            frame: 当前帧（可选）。传入且开启光流时，计算全局光流幅值
                   作为 active/resting 的辅助判据；不传则只用 bbox 位移。
        """
        ts = ts or time.time()
        out: list[BehaviorOutput] = []
        live_ids: set[int] = set()

        # 全局光流（每帧算一次，所有 track 共用）
        if frame is not None and self.cfg.enable_optical_flow:
            try:
                self._flow_mag = self._compute_flow(frame)
            except Exception:
                self._flow_mag = 0.0

        for det in detections:
            tid = det.track_id
            if tid < 0:
                continue
            live_ids.add(tid)

            st = self.states.get(tid)
            if st is None:
                st = _TrackState(track_id=tid, species=det.species)
                self.states[tid] = st
            st.species = det.species
            st.push(ts, det.center, det.bbox)

            behavior = self._classify(st, frame_size)
            dur = ts - st.state_start_ts
            out.append(
                BehaviorOutput(
                    track_id=tid,
                    species=det.species,
                    behavior=behavior,
                    confidence=det.confidence,
                    duration_in_state_sec=dur,
                    is_anomaly=(behavior == "anomaly"),
                )
            )

        # 清理长期消失的轨迹（避免内存膨胀）
        for tid in list(self.states.keys()):
            if tid not in live_ids:
                st = self.states[tid]
                if ts - st.last_update_ts > 60:
                    self.states.pop(tid, None)

        return out

    # ---------------------------------------------------------- core
    def _classify(self, st: _TrackState, frame_size: tuple[int, int]) -> str:
        cfg = self.cfg
        ts = st.last_update_ts
        history = st.history

        if len(history) < 3:
            return "unknown"

        motion = self._motion_magnitude(history)
        jitter = self._jitter_magnitude(history)
        flow = self._flow_mag
        last_center = history[-1][1]
        in_food = self._in_roi(last_center, cfg.food_roi, frame_size)
        in_water = self._in_roi(last_center, cfg.water_roi, frame_size)

        # 进食 / 饮水 ROI 优先判定
        if in_food:
            new = "eating"
        elif in_water:
            new = "drinking"
        elif motion < cfg.resting_motion_px and flow < cfg.flow_resting_threshold:
            # bbox 几乎不动 且 全局光流也极小 → 真静止
            new = "resting"
        elif motion > cfg.active_motion_px_per_frame or flow > cfg.flow_active_threshold:
            # bbox 明显位移 或 全局光流明显 → 活动
            new = "active"
        else:
            new = "active"

        # anomaly 升级：长时间处于 resting 且有微抖动（疑似晕厥）
        # 仅在当前被判为 resting 时升级
        if new == "resting":
            dur = ts - st.state_start_ts
            if (
                dur > cfg.anomaly_min_duration_sec
                and jitter < cfg.anomaly_jitter_px
                and motion < cfg.resting_motion_px * 0.5
            ):
                new = "anomaly"

        # 状态变化处理
        if new != st.current_behavior:
            # 关闭上一段汇总
            prev_dur = ts - st.state_start_ts
            if prev_dur > 0.5:
                # 延迟导入避免循环
                from .database import get_db
                try:
                    get_db().insert_behavior_summary(
                        track_id=st.track_id,
                        behavior=st.current_behavior,
                        start_ts=st.state_start_ts,
                        end_ts=ts,
                        duration_sec=prev_dur,
                    )
                except Exception:
                    pass
            st.current_behavior = new
            st.state_start_ts = ts

        return new

    # ---------------------------------------------------------- API
    def snapshot(self) -> dict[int, str]:
        return {tid: st.current_behavior for tid, st in self.states.items()}


# --------------------------------------------------------------------------- #
# 自检
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from .detector import Detection
    import time as _t

    ba = BehaviorAnalyzer()
    # 模拟一只静止的猫 60 秒
    base = _t.time()
    for i in range(120):
        ts = base + i * 0.5
        det = Detection(track_id=1, species="cat", confidence=0.9,
                        bbox=(100 + (i % 3) * 0.1, 100, 200, 300))
        outs = ba.update([det], frame_size=(640, 480), ts=ts)
        if i % 30 == 0:
            print(ts, outs[0].behavior)