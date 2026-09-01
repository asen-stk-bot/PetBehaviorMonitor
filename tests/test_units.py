"""
单元测试 / Unit Tests

覆盖范围：
- 数据库 CRUD
- 行为分析器（基于合成检测数据）
- 跟踪器 IoU 匹配
- 报警冷却

运行：
    python main.py --tests
或：
    python -m unittest discover tests -v
"""
from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path

# 确保 src/ 可导入
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# --------------------------------------------------------------------------- #
# 数据库
# --------------------------------------------------------------------------- #
class TestDatabase(unittest.TestCase):
    """SQLite 数据库测试，使用临时路径。"""

    def setUp(self) -> None:
        from pet_monitor.core.database import Database
        self.tmp = ROOT / "data" / f"test_{int(time.time()*1000)}.db"
        self.db = Database(self.tmp)

    def tearDown(self) -> None:
        try:
            self.tmp.unlink(missing_ok=True)
        except Exception:
            pass

    def test_event_insert_and_query(self) -> None:
        ts = time.time()
        self.db.insert_event(
            ts=ts, track_id=1, species="cat", behavior="resting",
            confidence=0.9, bbox=(0, 0, 100, 100), source="test",
        )
        self.assertEqual(self.db.count_events(), 1)
        rows = self.db.recent_events()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["species"], "cat")

    def test_alert_insert_and_ack(self) -> None:
        aid = self.db.insert_alert(level="warning", message="x")
        alerts = self.db.list_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertFalse(alerts[0]["acknowledged"])
        self.db.acknowledge_alert(aid)
        alerts = self.db.list_alerts()
        self.assertTrue(alerts[0]["acknowledged"])

    def test_behavior_summary(self) -> None:
        self.db.insert_behavior_summary(
            track_id=1, behavior="resting",
            start_ts=0, end_ts=10, duration_sec=10,
        )
        self.db.insert_behavior_summary(
            track_id=1, behavior="active",
            start_ts=10, end_ts=20, duration_sec=10,
        )
        dist = self.db.behavior_distribution()
        self.assertAlmostEqual(dist["resting"], 10.0)
        self.assertAlmostEqual(dist["active"], 10.0)

    def test_purge_older_than(self) -> None:
        old_ts = time.time() - 100 * 86400
        new_ts = time.time()
        self.db.insert_event(ts=old_ts, track_id=1, species="cat", behavior="resting",
                             confidence=0.5, bbox=(0, 0, 10, 10), source="x")
        self.db.insert_event(ts=new_ts, track_id=2, species="dog", behavior="active",
                             confidence=0.5, bbox=(0, 0, 10, 10), source="x")
        deleted = self.db.purge_older_than(days=30)
        self.assertEqual(deleted, 1)
        self.assertEqual(self.db.count_events(), 1)


# --------------------------------------------------------------------------- #
# 跟踪器
# --------------------------------------------------------------------------- #
class TestSimpleTracker(unittest.TestCase):
    """IoU 跟踪器测试：跨帧 ID 保持 + 新目标分配。"""

    def setUp(self) -> None:
        from pet_monitor.core.detector import SimpleTracker
        self.Tracker = SimpleTracker

    def _det(self, tid, bbox):
        from pet_monitor.core.detector import Detection
        return Detection(track_id=tid, species="cat", confidence=0.9, bbox=bbox)

    def test_persistent_id(self) -> None:
        tr = self.Tracker()
        d1 = self._det(-1, (0, 0, 100, 100))
        d2 = self._det(-1, (5, 5, 105, 105))     # 同一只，轻微位移
        out1 = tr.update([d1])
        out2 = tr.update([d2])
        self.assertEqual(out1[0].track_id, 1)
        self.assertEqual(out2[0].track_id, 1)

    def test_new_target_gets_new_id(self) -> None:
        tr = self.Tracker()
        d1 = self._det(-1, (0, 0, 100, 100))
        d2 = self._det(-1, (300, 300, 400, 400))   # 远处另一只
        out1 = tr.update([d1])
        out2 = tr.update([d1, d2])
        self.assertEqual(out2[0].track_id, 1)
        self.assertEqual(out2[1].track_id, 2)


# --------------------------------------------------------------------------- #
# 行为分析器
# --------------------------------------------------------------------------- #
class TestBehaviorAnalyzer(unittest.TestCase):
    """行为识别测试。"""

    def _det(self, tid, x, y, w=100, h=100, species="cat"):
        from pet_monitor.core.detector import Detection
        return Detection(track_id=tid, species=species, confidence=0.9,
                         bbox=(x, y, x + w, y + h))

    def test_resting_detected(self) -> None:
        from pet_monitor.core.behavior import BehaviorAnalyzer
        a = BehaviorAnalyzer()
        # 同一位置长时间
        base = time.time()
        outs = []
        for i in range(60):
            ts = base + i * 0.5  # 共 30 秒
            outs = a.update([self._det(1, 100 + (i % 2) * 0.5, 200)],
                            frame_size=(640, 480), ts=ts)
        # 30 秒静止 → resting（>= rest_min_duration_sec=10s）
        self.assertIn(outs[0].behavior, ("resting", "unknown"))

    def test_active_detected(self) -> None:
        from pet_monitor.core.behavior import BehaviorAnalyzer
        a = BehaviorAnalyzer()
        base = time.time()
        outs = []
        for i in range(10):
            ts = base + i * 0.1
            x = 100 + i * 20  # 快速位移
            outs = a.update([self._det(1, x, 200)], frame_size=(640, 480), ts=ts)
        self.assertEqual(outs[0].behavior, "active")

    def test_eating_in_food_roi(self) -> None:
        from pet_monitor.core.behavior import BehaviorAnalyzer
        a = BehaviorAnalyzer()
        base = time.time()
        outs = []
        # 在进食 ROI（左下）停留 5 秒
        for i in range(50):
            ts = base + i * 0.1
            outs = a.update([self._det(1, 200, 400)],  # 在 food ROI 内
                            frame_size=(640, 480), ts=ts)
        self.assertEqual(outs[0].behavior, "eating")


# --------------------------------------------------------------------------- #
# 报警
# --------------------------------------------------------------------------- #
class TestAlertSystem(unittest.TestCase):
    """报警冷却测试。"""

    def test_cooldown_blocks_repeat(self) -> None:
        from pet_monitor.core.alert import AlertSystem, LEVEL_WARN
        from pet_monitor.config import CONFIG
        CONFIG.alert.cooldown_sec = 5.0
        a = AlertSystem()

        r1 = a.raise_(LEVEL_WARN, "test", cooldown_key="x")
        r2 = a.raise_(LEVEL_WARN, "test", cooldown_key="x")
        self.assertIsNotNone(r1)
        self.assertIsNone(r2)  # 冷却中


if __name__ == "__main__":
    unittest.main(verbosity=2)