"""
YOLO 检测器

封装 ultralytics YOLOv8，提供：
- 宠物检测（猫/狗，COCO id 15/16）
- 跟踪（简化版：bbox IoU 跟踪，给每个宠物分配稳定 track_id）
- 结果数据结构 Detection

设计要点：
- ultralytics 首次运行自动从官方仓库下载 yolov8n.pt 到当前目录
  本项目把模型缓存到 data/models/ 目录
- detector.detect(frame) 返回 List[Detection]
- tracker.update(detections) 在帧间维持 track_id
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from ..config import CONFIG


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """单帧单目标检测结果。"""
    track_id: int                       # -1 表示未分配
    species: str                        # 'cat' / 'dog'
    confidence: float
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 (pixels)
    center: tuple[float, float] = field(init=False)
    width: float = field(init=False)
    height: float = field(init=False)

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        self.width = max(0.0, x2 - x1)
        self.height = max(0.0, y2 - y1)
        self.center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


# --------------------------------------------------------------------------- #
# 简化 IoU 跟踪器
# --------------------------------------------------------------------------- #
class SimpleTracker:
    """基于 bbox IoU 的轻量跟踪器。

    不使用 SORT/DeepSORT 等重型依赖；教学项目足以演示跟踪思想。
    """

    def __init__(self, max_missing: int = 15, iou_threshold: float = 0.3) -> None:
        self.next_id = 1
        self.max_missing = max_missing
        self.iou_threshold = iou_threshold
        self.tracks: dict[int, dict] = {}
        # 历史中心点（用于行为分析中的运动量计算）
        self.history: dict[int, deque] = {}

    @staticmethod
    def _iou(a: tuple, b: tuple) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / ua if ua > 0 else 0.0

    def update(self, detections: list[Detection]) -> list[Detection]:
        """用 IoU 贪心匹配把 detections 与已有轨迹关联。

        返回的列表中每个 Detection 的 track_id 已被赋值（matched 复用旧 id，
        未匹配的分配新 id）。返回的全部是本帧活跃的目标。
        """
        track_ids = list(self.tracks.keys())
        n_det = len(detections)
        n_trk = len(track_ids)

        # 1. IoU 矩阵
        iou_mat = np.zeros((n_det, n_trk), dtype=np.float32)
        for i, d in enumerate(detections):
            for j, tid in enumerate(track_ids):
                iou_mat[i, j] = self._iou(d.bbox, self.tracks[tid]["bbox"])

        # 2. 贪心匹配：按 IoU 从大到小遍历
        det_used = [False] * n_det
        trk_used = [False] * n_trk
        assigned: dict[int, int] = {}    # det_index -> track_id
        if n_det and n_trk:
            pairs = [
                (iou_mat[i, j], i, j)
                for i in range(n_det) for j in range(n_trk)
            ]
            pairs.sort(key=lambda x: -x[0])
            for score, i, j in pairs:
                if score < self.iou_threshold:
                    break
                if det_used[i] or trk_used[j]:
                    continue
                det_used[i] = True
                trk_used[j] = True
                assigned[i] = track_ids[j]

        # 3. 应用：为每个 detection 分配 track_id（matched 复用，未匹配新建）
        for i, det in enumerate(detections):
            if i in assigned:
                tid = assigned[i]
            else:
                tid = self.next_id
                self.next_id += 1
            det.track_id = tid
            self.tracks[tid] = {
                "bbox": det.bbox,
                "missing": 0,
                "last_ts": time.time(),
                "species": det.species,
            }
            self.history.setdefault(tid, deque(maxlen=120))
            self.history[tid].append(det.center)

        # 4. 未匹配的轨迹累计 missing，超过阈值则删除
        for j, tid in enumerate(track_ids):
            if trk_used[j]:
                continue
            self.tracks[tid]["missing"] += 1
            if self.tracks[tid]["missing"] > self.max_missing:
                self.tracks.pop(tid, None)
                self.history.pop(tid, None)

        # 5. 返回本帧所有活跃检测（含新分配的）
        return list(detections)

    def centers(self, track_id: int) -> list[tuple[float, float]]:
        return list(self.history.get(track_id, []))


# --------------------------------------------------------------------------- #
# 检测器
# --------------------------------------------------------------------------- #
class PetDetector:
    """封装 ultralytics YOLO。

    Usage:
        det = PetDetector()
        for frame in video:
            results = det.detect(frame)
            # results: List[Detection]
    """

    # COCO 类别名（仅用到 15/16）
    COCO_NAMES = {15: "cat", 16: "dog"}

    def __init__(self) -> None:
        from ultralytics import YOLO  # 延迟导入：未装时不阻塞其他模块

        cfg = CONFIG.detector
        # 把模型缓存到 data/models/
        from ..config import MODEL_DIR
        model_path = MODEL_DIR / cfg.model_name
        # ultralytics 会自动下载；通过把 weights 下载到我们指定的目录实现集中管理
        try:
            self.model = YOLO(str(model_path))
        except Exception:
            # 自动从 ultralytics 官方下载
            self.model = YOLO(cfg.model_name)
            try:
                import shutil
                src = Path(self.model.ckpt_path) if hasattr(self.model, "ckpt_path") else None
                if src and src.exists():
                    shutil.copy2(src, model_path)
            except Exception:
                pass  # 不强制

        self.cfg = cfg
        self.tracker = SimpleTracker(
            max_missing=CONFIG.behavior.track_max_missing,
            iou_threshold=cfg.iou_threshold,
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """对单帧进行检测 + 跟踪。"""
        if frame is None or frame.size == 0:
            return []
        # ultralytics 支持 numpy BGR
        results = self.model.predict(
            frame,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device or None,
            verbose=False,
        )
        raw: list[Detection] = []
        if not results:
            return []
        r = results[0]
        boxes = r.boxes
        if boxes is None or boxes.xyxy is None:
            return self.tracker.update([])
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy().astype(int)
        for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
            if k not in self.cfg.pet_class_ids:
                continue
            raw.append(
                Detection(
                    track_id=-1,
                    species=self.COCO_NAMES.get(k, "pet"),
                    confidence=float(c),
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                )
            )
        return self.tracker.update(raw)


# --------------------------------------------------------------------------- #
# 单元自检
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    print("[detector] 直接运行需先安装 ultralytics/opencv", file=sys.stderr)
    print("[detector] 单元测试请使用 pytest tests/test_detector.py", file=sys.stderr)