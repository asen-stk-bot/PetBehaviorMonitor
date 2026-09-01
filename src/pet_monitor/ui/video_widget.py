"""
视频显示控件
封装 QLabel 显示 numpy 帧，支持 ROI 叠加层（绘制 food/water 区域）。
"""
from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QSizePolicy


class VideoWidget(QLabel):
    """视频显示组件 + ROI 绘制。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111; color:#aaa;")
        self.setText("等待视频源…")
        self._frame_size: tuple[int, int] | None = None
        self._roi_overlay: tuple[tuple[float, float, float, float], tuple[float, float, float, float]] | None = None

    def set_roi(self, food: tuple[float, float, float, float],
                water: tuple[float, float, float, float]) -> None:
        self._roi_overlay = (food, water)

    def show_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        if self._roi_overlay:
            frame = self._draw_rois(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg).scaled(
            self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.setPixmap(pix)

    def _draw_rois(self, frame: np.ndarray) -> np.ndarray:
        if not self._frame_size or not self._roi_overlay:
            return frame
        out = frame.copy()
        h, w = self._frame_size
        for (rx1, ry1, rx2, ry2), color in zip(
            self._roi_overlay,
            [(0, 200, 255), (255, 150, 0)],
        ):
            x1, y1, x2, y2 = int(rx1 * w), int(ry1 * h), int(rx2 * w), int(ry2 * h)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        return out