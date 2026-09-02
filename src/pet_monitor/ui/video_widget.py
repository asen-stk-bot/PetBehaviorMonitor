"""
视频显示控件（重写为 QWidget 自定义绘制）

功能：
- 接收 numpy BGR 帧并保持宽高比绘制
- 叠加显示 food/water ROI 区域（半透明填充 + 边框）
- 在编辑模式下支持鼠标拖拽绘制矩形
- 释放鼠标时按当前帧实际显示区域换算归一化坐标，发出 roi_drawn 信号

接口兼容旧版（main_window 仍按 set_roi / show_frame 调用）。
"""
from __future__ import annotations

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QPoint, QRect, QSize, pyqtSignal
from PyQt5.QtGui import QImage, QPainter, QPen, QColor, QBrush, QPixmap
from PyQt5.QtWidgets import QSizePolicy, QWidget


class VideoWidget(QWidget):
    """视频显示 + ROI 绘制 + 鼠标拖拽画框。"""

    # 释放鼠标时发出：kind 标识当前编辑哪类 ROI（"food"/"water"），roi 为归一化 (x1,y1,x2,y2)
    roi_drawn = pyqtSignal(str, tuple)

    # 当前编辑状态："none" | "food" | "water"
    edit_kind: str = "none"

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setStyleSheet("background:#111;")
        self.setMouseTracking(True)
        self._pix: QPixmap | None = None
        self._frame_size: tuple[int, int] | None = None  # (w, h) 原图分辨率
        self._food_roi: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._water_roi: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._drag_start: QPoint | None = None
        self._drag_end: QPoint | None = None

    # ------------------------------------------------------------------ #
    # 对外接口
    # ------------------------------------------------------------------ #
    def set_roi(self, food: tuple, water: tuple) -> None:
        self._food_roi = food
        self._water_roi = water
        self.update()

    def show_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        h, w = frame.shape[:2]
        self._frame_size = (w, h)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format_RGB888).copy()
        self._pix = QPixmap.fromImage(qimg)
        self.update()

    def set_edit_kind(self, kind: str) -> None:
        """设置编辑模式：'food' / 'water' / 'none'。"""
        self.edit_kind = kind
        if kind == "none":
            self._drag_start = self._drag_end = None
            self.unsetCursor()
        else:
            self.setCursor(Qt.CrossCursor)
        self.update()

    # ------------------------------------------------------------------ #
    # 内部：把 QRect 归一化为 0-1
    # ------------------------------------------------------------------ #
    def _display_rect(self) -> QRect:
        """当前帧实际绘制在 widget 上的矩形（保持宽高比，居中）。"""
        if self._pix is None:
            return QRect(0, 0, self.width(), self.height())
        sw, sh = self._pix.width(), self._pix.height()
        ww, wh = self.width(), self.height()
        if sw == 0 or sh == 0 or ww == 0 or wh == 0:
            return QRect(0, 0, ww, wh)
        # 按 keep-aspect 计算实际绘制区域
        ratio_pix = sw / sh
        ratio_w = ww / wh
        if ratio_pix > ratio_w:
            dw = ww
            dh = int(ww / ratio_pix)
        else:
            dh = wh
            dw = int(wh * ratio_pix)
        x = (ww - dw) // 2
        y = (wh - dh) // 2
        return QRect(x, y, dw, dh)

    def _qrect_to_norm(self, qr: QRect) -> tuple[float, float, float, float] | None:
        dr = self._display_rect()
        if dr.width() <= 0 or dr.height() <= 0:
            return None
        # 裁剪到 display rect
        x1 = max(qr.left(), dr.left())
        y1 = max(qr.top(), dr.top())
        x2 = min(qr.right(), dr.right())
        y2 = min(qr.bottom(), dr.bottom())
        if x2 <= x1 or y2 <= y1:
            return None
        nx1 = (x1 - dr.left()) / dr.width()
        ny1 = (y1 - dr.top()) / dr.height()
        nx2 = (x2 - dr.left()) / dr.width()
        ny2 = (y2 - dr.top()) / dr.height()
        return (round(nx1, 4), round(ny1, 4), round(nx2, 4), round(ny2, 4))

    # ------------------------------------------------------------------ #
    # 鼠标事件：仅在 edit_kind != "none" 时启用
    # ------------------------------------------------------------------ #
    def mousePressEvent(self, e) -> None:
        if self.edit_kind == "none" or e.button() != Qt.LeftButton:
            return
        self._drag_start = e.pos()
        self._drag_end = e.pos()
        self.update()

    def mouseMoveEvent(self, e) -> None:
        if self.edit_kind == "none" or self._drag_start is None:
            return
        self._drag_end = e.pos()
        self.update()

    def mouseReleaseEvent(self, e) -> None:
        if self.edit_kind == "none" or e.button() != Qt.LeftButton:
            return
        if self._drag_start is None or self._drag_end is None:
            return
        qr = QRect(self._drag_start, self._drag_end).normalized()
        norm = self._qrect_to_norm(qr)
        kind = self.edit_kind
        # 清拖拽态
        self._drag_start = self._drag_end = None
        if norm is not None:
            if kind == "food":
                self._food_roi = norm
            elif kind == "water":
                self._water_roi = norm
            self.roi_drawn.emit(kind, norm)
        self.update()

    # ------------------------------------------------------------------ #
    # 绘制
    # ------------------------------------------------------------------ #
    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#111"))
        # 1) 视频帧
        if self._pix is not None:
            dr = self._display_rect()
            p.drawPixmap(dr, self._pix, self._pix.rect())
        else:
            p.setPen(QColor("#888"))
            p.drawText(self.rect(), Qt.AlignCenter, "等待视频源…")

        # 2) ROI 叠加
        dr = self._display_rect()
        for roi, color, label in [
            (self._food_roi, QColor(0, 200, 255, 70), "食盆"),
            (self._water_roi, QColor(255, 150, 0, 70), "水盆"),
        ]:
            x1, y1, x2, y2 = roi
            if x2 <= x1 or y2 <= y1:
                continue
            rx1 = dr.left() + int(x1 * dr.width())
            ry1 = dr.top() + int(y1 * dr.height())
            rx2 = dr.left() + int(x2 * dr.width())
            ry2 = dr.top() + int(y2 * dr.height())
            p.fillRect(QRect(rx1, ry1, rx2 - rx1, ry2 - ry1), color)
            p.setPen(QPen(color.darker(200), 1, Qt.DashLine))
            p.drawRect(QRect(rx1, ry1, rx2 - rx1, ry2 - ry1))
            p.setPen(QColor("white"))
            p.drawText(rx1 + 4, ry1 + 14, f"{label}  ({x1:.2f},{y1:.2f})-({x2:.2f},{y2:.2f})")

        # 3) 拖拽中的临时矩形
        if self._drag_start and self._drag_end:
            qr = QRect(self._drag_start, self._drag_end).normalized()
            color = QColor(255, 255, 0, 80) if self.edit_kind == "food" else QColor(0, 255, 255, 80)
            p.fillRect(qr, color)
            p.setPen(QPen(color.darker(150), 2))
            p.drawRect(qr)

        # 4) 编辑模式提示横幅
        if self.edit_kind != "none":
            p.setPen(QColor("yellow"))
            p.drawText(self.rect().adjusted(8, 8, -8, -8), Qt.AlignTop | Qt.AlignLeft,
                       f"编辑模式：拖拽绘制 {self.edit_kind} ROI（按 Esc 取消）")
        p.end()

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_Escape and self.edit_kind != "none":
            self._drag_start = self._drag_end = None
            self.set_edit_kind("none")
            e.accept()
