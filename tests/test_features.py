"""
新增功能单元测试 / Feature Tests

覆盖：
- 数据库统计分析（hourly_activity / daily_summary / stats_overview / 每日汇总 / CSV 导出）
- 快照（SnapshotManager 落盘 + 冷却 + 数据库快照记录）
- 报告生成（HTML 文件有效生成）
- 配置持久化（save / load 往返）

运行：
    python main.py --tests
"""
from __future__ import annotations

import datetime
import sys
import time
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _tmp_db() -> Path:
    return ROOT / "data" / f"test_feat_{int(time.time() * 1000)}.db"


class TestAnalytics(unittest.TestCase):
    """数据库统计与导出。"""

    def setUp(self) -> None:
        from pet_monitor.core.database import Database
        self.tmp = _tmp_db()
        self.db = Database(self.tmp)

    def tearDown(self) -> None:
        try:
            self.tmp.unlink(missing_ok=True)
        except Exception:
            pass

    def _seed(self) -> str:
        today = datetime.date.today().strftime("%Y-%m-%d")
        start = datetime.datetime.strptime(today, "%Y-%m-%d").timestamp()
        for b, sec in [("resting", 3600), ("active", 1800), ("eating", 900)]:
            self.db.insert_event(
                ts=start + 1000, track_id=1, species="cat", behavior=b,
                confidence=0.9, bbox=(0, 0, 10, 10), source="t",
            )
            self.db.insert_behavior_summary(
                track_id=1, behavior=b, start_ts=start, end_ts=start + sec,
                duration_sec=sec,
            )
        return today

    def test_hourly_activity_has_24_buckets(self) -> None:
        today = self._seed()
        buckets = self.db.hourly_activity(today)
        self.assertEqual(len(buckets), 24)
        self.assertGreater(sum(b["total"] for b in buckets), 0)

    def test_daily_summary(self) -> None:
        today = self._seed()
        summary = self.db.daily_summary(today)
        self.assertAlmostEqual(summary["resting"], 3600.0)
        self.assertAlmostEqual(summary["eating"], 900.0)

    def test_stats_overview(self) -> None:
        self._seed()
        ov = self.db.stats_overview()
        self.assertEqual(ov["total_events"], 3)
        self.assertAlmostEqual(ov["total_sec"], 6300.0)

    def test_upsert_daily_stats(self) -> None:
        today = self._seed()
        self.db.upsert_daily_stats(today)
        rows = self.db.list_daily_stats()
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["rest_sec"], 3600.0)

    def test_export_csv(self) -> None:
        self._seed()
        out = ROOT / "data" / f"test_export_{int(time.time() * 1000)}.csv"
        try:
            n = self.db.export_events_csv(out)
            self.assertEqual(n, 3)
            self.assertTrue(out.exists())
        finally:
            out.unlink(missing_ok=True)


class TestSnapshot(unittest.TestCase):
    """快照落盘与冷却。"""

    def setUp(self) -> None:
        self.out_dir = ROOT / "data" / f"test_snap_{int(time.time() * 1000)}"

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_save_and_cooldown(self) -> None:
        from pet_monitor.core.snapshot import SnapshotManager, list_snapshot_files

        m = SnapshotManager(self.out_dir)
        m.cooldown_sec = 1.0
        img = np.zeros((100, 100, 3), dtype=np.uint8)

        p1 = m.save(img, track_id=1, behavior="anomaly", force=True)
        self.assertIsNotNone(p1)
        self.assertTrue(p1.exists())

        # 冷却期内重复保存应返回 None
        p2 = m.save(img, track_id=1, behavior="anomaly")
        self.assertIsNone(p2)

        # force 跳过冷却
        p3 = m.save(img, track_id=1, behavior="anomaly", force=True)
        self.assertIsNotNone(p3)

        files = list_snapshot_files(self.out_dir)
        self.assertEqual(len(files), 2)


class TestReport(unittest.TestCase):
    """HTML 报告生成。"""

    def setUp(self) -> None:
        from pet_monitor.core.database import Database
        self.tmp = _tmp_db()
        self.db = Database(self.tmp)
        self.out_dir = ROOT / "data" / f"test_report_{int(time.time() * 1000)}"
        # 造数据
        start = datetime.datetime.now().replace(hour=0, minute=0, second=0,
                                                microsecond=0).timestamp()
        self.db.insert_event(ts=start, track_id=1, species="cat", behavior="resting",
                             confidence=0.9, bbox=(0, 0, 10, 10), source="t")
        self.db.insert_behavior_summary(track_id=1, behavior="resting",
                                        start_ts=start, end_ts=start + 60,
                                        duration_sec=60)

    def tearDown(self) -> None:
        import shutil
        self.tmp.unlink(missing_ok=True)
        shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_generate_report(self) -> None:
        from pet_monitor.core.report import generate_report
        p = generate_report(self.db, out_dir=self.out_dir)
        self.assertTrue(p.exists())
        html = p.read_text(encoding="utf-8")
        self.assertIn("宠物行为识别监测系统", html)
        self.assertIn("<svg", html)  # 含图表

    def test_export_csv(self) -> None:
        from pet_monitor.core.report import export_csv
        p = export_csv(self.db, out_dir=self.out_dir)
        self.assertTrue(p.exists())
        self.assertIn("behavior", p.read_text(encoding="utf-8"))


class TestMonitorAlerting(unittest.TestCase):
    """监控引擎报警 + 快照接线（回归 LEVEL_INFO 导入 bug）。"""

    def setUp(self) -> None:
        import pet_monitor.core.monitor as mon
        from pet_monitor.core.database import Database
        from pet_monitor.core.snapshot import SnapshotManager
        import numpy as np

        self.mon = mon
        # 避免加载 YOLO：用轻量假检测器替换 PetDetector
        mon.PetDetector = lambda: None  # Monitor.__init__ 仅实例化，不调用 detect

        self.tmp_db = _tmp_db()
        self.tmp_snap = ROOT / "data" / f"test_msnap_{int(time.time() * 1000)}"
        self._db = Database(self.tmp_db)
        self._snap = SnapshotManager(self.tmp_snap)
        self._snap.cooldown_sec = 0.0

        # 把单例替换为测试实例
        self._orig_get_db = mon.get_db
        self._orig_get_snapshot = mon.get_snapshot
        mon.get_db = lambda: self._db
        mon.get_snapshot = lambda: self._snap

        # AlertSystem 内部也调用 get_db，需一并替换为测试库
        import pet_monitor.core.alert as al
        self._alert_mod = al
        self._orig_alert_get_db = al.get_db
        al.get_db = lambda: self._db

        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)

    def tearDown(self) -> None:
        import shutil
        self.mon.get_db = self._orig_get_db
        self.mon.get_snapshot = self._orig_get_snapshot
        self._alert_mod.get_db = self._orig_alert_get_db
        self.tmp_db.unlink(missing_ok=True)
        shutil.rmtree(self.tmp_snap, ignore_errors=True)

    def test_eating_triggers_alert_and_snapshot(self) -> None:
        from pet_monitor.core.detector import Detection
        from pet_monitor.core.behavior import BehaviorOutput

        m = self.mon.Monitor(source="dummy")  # 不 start，仅测 _maybe_alert
        det = Detection(track_id=1, species="cat", confidence=0.9,
                        bbox=(10, 10, 100, 100))
        beh = BehaviorOutput(track_id=1, species="cat", behavior="eating",
                             confidence=0.9)

        m._maybe_alert(det, beh, self.frame)

        self.assertEqual(self._db.count_alerts(), 1)
        self.assertEqual(self._db.count_snapshots(), 1)
        snaps = self._db.list_snapshots()
        self.assertEqual(snaps[0]["behavior"], "eating")
        self.assertTrue(Path(snaps[0]["path"]).exists())


class TestConfigPersistence(unittest.TestCase):
    """配置 save / load 往返。"""

    def test_roundtrip(self) -> None:
        from pet_monitor.config import CONFIG, load_config, save_config

        p = ROOT / "data" / f"test_cfg_{int(time.time() * 1000)}.json"
        # 备份原值，测后还原，避免污染其他测试
        orig_conf = CONFIG.detector.conf_threshold
        orig_roi = CONFIG.behavior.food_roi
        try:
            CONFIG.detector.conf_threshold = 0.61
            CONFIG.behavior.food_roi = (0.1, 0.6, 0.5, 0.9)
            save_config(p)

            # 修改后再加载
            CONFIG.detector.conf_threshold = 0.9
            CONFIG.behavior.food_roi = (0.0, 0.0, 1.0, 1.0)
            self.assertTrue(load_config(p))
            self.assertAlmostEqual(CONFIG.detector.conf_threshold, 0.61)
            self.assertEqual(tuple(round(x, 2) for x in CONFIG.behavior.food_roi),
                             (0.1, 0.6, 0.5, 0.9))
        finally:
            p.unlink(missing_ok=True)
            CONFIG.detector.conf_threshold = orig_conf
            CONFIG.behavior.food_roi = orig_roi

    def test_load_missing_returns_false(self) -> None:
        from pet_monitor.config import load_config
        self.assertFalse(load_config(ROOT / "data" / "no_such_config.json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
