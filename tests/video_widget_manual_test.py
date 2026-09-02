"""
VideoWidget 纯逻辑单测 / VideoWidget Unit Tests (unittest 版)

不在 CI/默认 pytest collection 中跑——Windows + offscreen 平台下
QApplication(sys.argv) 在 git-bash 环境会卡。改用独立运行入口：

    QT_QPA_PLATFORM=offscreen python tests/test_video_widget.py

或在主测试流中被自动 skip（见 conftest.py / pytest.ini 排除）。
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# 在 import PyQt5.QtWidgets 前建好 QApplication（必须在 import VideoWidget 之前）
from PyQt5.QtWidgets import QApplication  # noqa: E402
_app = QApplication.instance() or QApplication(sys.argv)  # noqa: E402

from PyQt5.QtCore import Qt, QPoint  # noqa: E402
from PyQt5.QtTest import QTest  # noqa: E402
from pet_monitor.ui.video_widget import VideoWidget  # noqa: E402


def _pump(n: int = 5) -> None:
    for _ in range(n):
        QApplication.processEvents()


def _fake_frame(w: int = 640, h: int = 360) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestVideoWidgetBasics(unittest.TestCase):
    def setUp(self) -> None:
        self.w = VideoWidget()
        self.w.resize(640, 360)

    def tearDown(self) -> None:
        self.w.deleteLater()
        _pump()

    def test_initial_state(self) -> None:
        self.assertIsNone(self.w._pix)
        self.assertIsNone(self.w._frame_size)
        self.assertEqual(self.w.edit_kind, "none")
        self.assertEqual(self.w._food_roi, (0.0, 0.0, 0.0, 0.0))
        self.assertEqual(self.w._water_roi, (0.0, 0.0, 0.0, 0.0))

    def test_show_frame_records_size(self) -> None:
        self.w.show_frame(_fake_frame(320, 240))
        _pump()
        self.assertEqual(self.w._frame_size, (320, 240))
        self.assertIsNotNone(self.w._pix)
        self.assertEqual(self.w._pix.width(), 320)
        self.assertEqual(self.w._pix.height(), 240)

    def test_set_roi_stores_values(self) -> None:
        food = (0.1, 0.2, 0.3, 0.4)
        water = (0.5, 0.5, 0.9, 0.9)
        self.w.set_roi(food, water)
        self.assertEqual(self.w._food_roi, food)
        self.assertEqual(self.w._water_roi, water)

    def test_display_rect_no_frame_uses_widget_size(self) -> None:
        self.w.resize(800, 500)
        dr = self.w._display_rect()
        self.assertEqual(dr.width(), 800)
        self.assertEqual(dr.height(), 500)

    def test_display_rect_keeps_aspect_16_9_in_4_3(self) -> None:
        """16:9 视频在 4:3 widget 中应按宽 fit、垂直居中。"""
        self.w.resize(800, 600)  # 4:3
        self.w.show_frame(_fake_frame(640, 360))  # 16:9
        _pump()
        dr = self.w._display_rect()
        self.assertEqual(dr.width(), 800)
        self.assertEqual(dr.height(), 450)  # 800 * 360 / 640
        self.assertEqual(dr.left(), 0)
        self.assertEqual(dr.top(), 75)  # (600 - 450) / 2


class TestVideoWidgetEditMode(unittest.TestCase):
    def setUp(self) -> None:
        self.w = VideoWidget()
        self.w.resize(640, 360)
        self.w.show_frame(_fake_frame(640, 360))
        _pump()

    def tearDown(self) -> None:
        self.w.deleteLater()
        _pump()

    def test_set_edit_kind_toggles(self) -> None:
        self.w.set_edit_kind("food")
        self.assertEqual(self.w.edit_kind, "food")
        self.assertEqual(self.w.cursor().shape(), Qt.CrossCursor)
        self.w.set_edit_kind("water")
        self.assertEqual(self.w.edit_kind, "water")
        self.w.set_edit_kind("none")
        self.assertEqual(self.w.edit_kind, "none")

    def test_esc_cancels_edit(self) -> None:
        self.w.setFocus()
        self.w.set_edit_kind("water")
        QTest.keyClick(self.w, Qt.Key_Escape)
        _pump()
        self.assertEqual(self.w.edit_kind, "none")

    def test_drag_emits_roi_drawn_food(self) -> None:
        self.w.set_edit_kind("food")
        dr = self.w._display_rect()
        captured = []
        self.w.roi_drawn.connect(lambda kind, norm: captured.append((kind, norm)))
        start = QPoint(dr.left() + 50, dr.top() + 50)
        end = QPoint(dr.left() + 200, dr.top() + 150)
        QTest.mousePress(self.w, Qt.LeftButton, pos=start)
        QTest.mouseMove(self.w, end)
        QTest.mouseRelease(self.w, Qt.LeftButton, pos=end)
        _pump()
        self.assertEqual(len(captured), 1)
        kind, norm = captured[0]
        self.assertEqual(kind, "food")
        self.assertEqual(len(norm), 4)
        for v in norm:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)
        self.assertEqual(self.w._food_roi, norm)

    def test_drag_emits_roi_drawn_water(self) -> None:
        self.w.set_edit_kind("water")
        captured = []
        self.w.roi_drawn.connect(lambda kind, norm: captured.append((kind, norm)))
        QTest.mousePress(self.w, Qt.LeftButton, pos=QPoint(100, 100))
        QTest.mouseMove(self.w, QPoint(300, 250))
        QTest.mouseRelease(self.w, Qt.LeftButton, pos=QPoint(300, 250))
        _pump()
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], "water")
        self.assertEqual(self.w._water_roi, captured[0][1])

    def test_drag_clipped_to_normalized_range(self) -> None:
        self.w.resize(400, 300)
        self.w.show_frame(_fake_frame(640, 360))
        _pump()
        self.w.set_edit_kind("water")
        captured = []
        self.w.roi_drawn.connect(lambda kind, norm: captured.append((kind, norm)))
        QTest.mousePress(self.w, Qt.LeftButton, pos=QPoint(0, 0))
        QTest.mouseMove(self.w, QPoint(399, 299))
        QTest.mouseRelease(self.w, Qt.LeftButton, pos=QPoint(399, 299))
        _pump()
        self.assertEqual(len(captured), 1)
        for v in captured[0][1]:
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0, f"归一化越界: {captured[0][1]}")

    def test_right_button_no_emit(self) -> None:
        self.w.set_edit_kind("food")
        captured = []
        self.w.roi_drawn.connect(lambda kind, norm: captured.append((kind, norm)))
        QTest.mousePress(self.w, Qt.RightButton, pos=QPoint(50, 50))
        QTest.mouseMove(self.w, QPoint(150, 150))
        QTest.mouseRelease(self.w, Qt.RightButton, pos=QPoint(150, 150))
        _pump()
        self.assertEqual(len(captured), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
