"""
纯 QPainter 图表组件 / Chart Widget

不引入 matplotlib 等重型依赖，用 QPainter 直接绘制：
- BarChart：行为时长分布横向条形图（支持配色与中文标签）
- 保持轻量、可随窗口缩放
"""
from __future__ import annotations

from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QColor, QFont, QPainter, QPen
from PyQt5.QtWidgets import QWidget

from ..core.behavior import BEHAVIOR_LABELS_CN


BEHAVIOR_COLORS = {
    "resting":  QColor("#5b8def"),
    "active":   QColor("#34c98e"),
    "eating":   QColor("#f2a13b"),
    "drinking": QColor("#42c6d6"),
    "anomaly":  QColor("#e5484d"),
    "unknown":  QColor("#9aa3b0"),
}


class BarChart(QWidget):
    """横向条形图：输入 {behavior: 秒数}。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._data: dict[str, float] = {}
        self.setMinimumHeight(160)

    def set_data(self, data: dict[str, float]) -> None:
        self._data = dict(data)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        w, h = self.width(), self.height()
        # 背景
        p.fillRect(0, 0, w, h, QColor("#ffffff"))

        if not self._data:
            p.setPen(QColor("#999999"))
            p.drawText(QRectF(0, 0, w, h), Qt.AlignCenter, "暂无数据")
            p.end()
            return

        total = sum(self._data.values()) or 1.0
        items = sorted(self._data.items(), key=lambda kv: -kv[1])
        n = len(items)
        gap = 8
        bar_h = max(14, (h - gap * (n - 1) - 24) / n)
        label_w = 70
        value_w = 130
        bar_x = label_w
        bar_max_w = max(10, w - label_w - value_w - 20)

        font = QFont("Microsoft YaHei", 9)
        p.setFont(font)

        for i, (k, v) in enumerate(items):
            y = 12 + i * (bar_h + gap)
            pct = v / total
            color = BEHAVIOR_COLORS.get(k, QColor("#cccccc"))
            cn = BEHAVIOR_LABELS_CN.get(k, k)

            # 标签
            p.setPen(QColor("#333333"))
            p.drawText(QRectF(0, y, label_w - 6, bar_h),
                       Qt.AlignRight | Qt.AlignVCenter, cn)

            # 条形
            bar_w = int(bar_max_w * pct)
            p.setPen(Qt.NoPen)
            p.setBrush(color)
            p.drawRoundedRect(QRectF(bar_x, y, bar_w, bar_h), 4, 4)

            # 数值
            p.setPen(QColor("#666666"))
            p.drawText(QRectF(bar_x + bar_w + 8, y, value_w, bar_h),
                       Qt.AlignLeft | Qt.AlignVCenter,
                       f"{self._fmt_sec(v)} ({pct * 100:.0f}%)")

        p.end()

    @staticmethod
    def _fmt_sec(sec: float) -> str:
        if sec >= 3600:
            return f"{sec / 3600:.2f} h"
        if sec >= 60:
            return f"{sec / 60:.1f} min"
        return f"{sec:.0f} s"
