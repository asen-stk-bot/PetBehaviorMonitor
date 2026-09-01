"""
SQLite 数据库访问层

表设计：
- pet_events       监控事件流（每帧/每秒采样聚合）
- behavior_summary 行为汇总（停留时长、占比）
- alerts           报警记录
- daily_stats      按天统计
- app_meta         应用元数据（启动版本/时间）

特点：
- 使用 sqlite3 标准库，零额外依赖
- 通过 context manager 管理连接，with engine.connect() as cur:
- WAL 模式提高并发写读性能
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable

from ..config import CONFIG


SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS pet_events (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,        -- Unix 时间戳
    track_id     INTEGER NOT NULL,
    species      TEXT    NOT NULL,        -- 'cat' / 'dog'
    behavior     TEXT    NOT NULL,        -- 当前行为
    confidence   REAL    NOT NULL,        -- 检测置信度
    bbox_x1      REAL    NOT NULL,
    bbox_y1      REAL    NOT NULL,
    bbox_x2      REAL    NOT NULL,
    bbox_y2      REAL    NOT NULL,
    source       TEXT    NOT NULL         -- 视频源标识
);

CREATE INDEX IF NOT EXISTS idx_events_ts ON pet_events(ts);
CREATE INDEX IF NOT EXISTS idx_events_track ON pet_events(track_id);

CREATE TABLE IF NOT EXISTS behavior_summary (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id     INTEGER NOT NULL,
    behavior     TEXT    NOT NULL,
    start_ts     REAL    NOT NULL,
    end_ts       REAL    NOT NULL,
    duration_sec REAL    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_summary_track ON behavior_summary(track_id);

CREATE TABLE IF NOT EXISTS alerts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    level        TEXT    NOT NULL,        -- info / warning / critical
    track_id     INTEGER,
    behavior     TEXT,
    message      TEXT    NOT NULL,
    acknowledged INTEGER NOT NULL DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(ts);

CREATE TABLE IF NOT EXISTS daily_stats (
    day          TEXT PRIMARY KEY,        -- YYYY-MM-DD
    total_events INTEGER NOT NULL,
    rest_sec     REAL    NOT NULL DEFAULT 0,
    active_sec   REAL    NOT NULL DEFAULT 0,
    eat_sec      REAL    NOT NULL DEFAULT 0,
    drink_sec    REAL    NOT NULL DEFAULT 0,
    anomaly_sec  REAL    NOT NULL DEFAULT 0,
    alert_count  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS app_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS snapshots (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    track_id     INTEGER,
    behavior     TEXT,
    path         TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(ts);
"""


class Database:
    """线程安全的 SQLite 封装。"""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path else CONFIG.db.path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------------ #
    # 连接管理
    # ------------------------------------------------------------------ #
    @contextmanager
    def _connect(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path, timeout=10.0, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(SCHEMA_SQL)

    # ------------------------------------------------------------------ #
    # 事件写入
    # ------------------------------------------------------------------ #
    def insert_event(
        self,
        *,
        ts: float,
        track_id: int,
        species: str,
        behavior: str,
        confidence: float,
        bbox: tuple[float, float, float, float],
        source: str,
    ) -> int:
        x1, y1, x2, y2 = bbox
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO pet_events
                (ts, track_id, species, behavior, confidence,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, track_id, species, behavior, confidence, x1, y1, x2, y2, source),
            )
            return cur.lastrowid

    def insert_events_batch(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO pet_events
                (ts, track_id, species, behavior, confidence,
                 bbox_x1, bbox_y1, bbox_x2, bbox_y2, source)
                VALUES (:ts, :track_id, :species, :behavior, :confidence,
                        :x1, :y1, :x2, :y2, :source)
                """,
                [
                    {
                        "ts": r["ts"], "track_id": r["track_id"],
                        "species": r["species"], "behavior": r["behavior"],
                        "confidence": r["confidence"],
                        "x1": r["bbox"][0], "y1": r["bbox"][1],
                        "x2": r["bbox"][2], "y2": r["bbox"][3],
                        "source": r["source"],
                    }
                    for r in rows
                ],
            )

    # ------------------------------------------------------------------ #
    # 行为汇总
    # ------------------------------------------------------------------ #
    def insert_behavior_summary(
        self, *, track_id: int, behavior: str,
        start_ts: float, end_ts: float, duration_sec: float,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO behavior_summary
                (track_id, behavior, start_ts, end_ts, duration_sec)
                VALUES (?, ?, ?, ?, ?)
                """,
                (track_id, behavior, start_ts, end_ts, duration_sec),
            )
            return cur.lastrowid

    # ------------------------------------------------------------------ #
    # 报警
    # ------------------------------------------------------------------ #
    def insert_alert(
        self, *, level: str, message: str,
        track_id: int | None = None, behavior: str | None = None,
        ts: float | None = None,
    ) -> int:
        ts = ts or time.time()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO alerts (ts, level, track_id, behavior, message)
                VALUES (?, ?, ?, ?, ?)
                """,
                (ts, level, track_id, behavior, message),
            )
            return cur.lastrowid

    def list_alerts(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM alerts ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def acknowledge_alert(self, alert_id: int) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE alerts SET acknowledged = 1 WHERE id = ?", (alert_id,)
            )

    # ------------------------------------------------------------------ #
    # 统计与查询
    # ------------------------------------------------------------------ #
    def count_events(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM pet_events").fetchone()[0]

    def behavior_distribution(
        self, since_ts: float | None = None
    ) -> dict[str, float]:
        """返回行为 → 总时长（秒）。"""
        sql = "SELECT behavior, SUM(duration_sec) AS s FROM behavior_summary"
        args: tuple = tuple()
        if since_ts is not None:
            sql += " WHERE start_ts >= ?"
            args = (since_ts,)
        sql += " GROUP BY behavior"
        with self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
            return {r["behavior"]: float(r["s"] or 0.0) for r in rows}

    def recent_events(self, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM pet_events ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def count_alerts(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]

    def count_snapshots(self) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0]

    # ------------------------------------------------------------------ #
    # 统计分析
    # ------------------------------------------------------------------ #
    def stats_overview(self) -> dict[str, Any]:
        """总览：事件/报警/快照数量、时间跨度、行为分布。"""
        with self._connect() as conn:
            span = conn.execute(
                "SELECT MIN(ts) AS mn, MAX(ts) AS mx FROM pet_events"
            ).fetchone()
        dist = self.behavior_distribution()
        return {
            "total_events": self.count_events(),
            "total_alerts": self.count_alerts(),
            "total_snapshots": self.count_snapshots(),
            "first_ts": span["mn"],
            "last_ts": span["mx"],
            "behavior_distribution": dist,
            "total_sec": sum(dist.values()),
        }

    def hourly_activity(self, date_str: str) -> list[dict[str, Any]]:
        """返回某天（YYYY-MM-DD，本地时区）每个小时各行为的采样点数。

        返回列表长度 24，每项为 {hour, total, behaviors:{behavior:count}}。
        """
        import datetime

        start = datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp()
        end = start + 86400
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT CAST(strftime('%H', ts, 'unixepoch', 'localtime') AS INTEGER) AS hour,
                       behavior, COUNT(*) AS n
                FROM pet_events
                WHERE ts >= ? AND ts < ?
                GROUP BY hour, behavior
                """,
                (start, end),
            ).fetchall()

        buckets: list[dict[str, Any]] = [
            {"hour": h, "total": 0, "behaviors": {}} for h in range(24)
        ]
        for r in rows:
            h = r["hour"]
            if not (0 <= h < 24):
                continue
            buckets[h]["behaviors"][r["behavior"]] = r["n"]
            buckets[h]["total"] += r["n"]
        return buckets

    def daily_summary(self, date_str: str) -> dict[str, Any]:
        """某天的行为时长汇总（秒），来自 behavior_summary。"""
        import datetime

        start = datetime.datetime.strptime(date_str, "%Y-%m-%d").timestamp()
        end = start + 86400
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT behavior, SUM(duration_sec) AS s
                FROM behavior_summary
                WHERE start_ts >= ? AND start_ts < ?
                GROUP BY behavior
                """,
                (start, end),
            ).fetchall()
        return {r["behavior"]: float(r["s"] or 0.0) for r in rows}

    def upsert_daily_stats(self, date_str: str) -> None:
        """把某天行为时长汇总写入 daily_stats 表（存在则更新）。"""
        summary = self.daily_summary(date_str)
        total = sum(summary.values())
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO daily_stats
                (day, total_events, rest_sec, active_sec, eat_sec, drink_sec,
                 anomaly_sec, alert_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(day) DO UPDATE SET
                    total_events = excluded.total_events,
                    rest_sec = excluded.rest_sec,
                    active_sec = excluded.active_sec,
                    eat_sec = excluded.eat_sec,
                    drink_sec = excluded.drink_sec,
                    anomaly_sec = excluded.anomaly_sec,
                    alert_count = excluded.alert_count
                """,
                (
                    date_str,
                    total,
                    summary.get("resting", 0.0),
                    summary.get("active", 0.0),
                    summary.get("eating", 0.0),
                    summary.get("drinking", 0.0),
                    summary.get("anomaly", 0.0),
                    0,  # 报警数按需在调用处补充
                ),
            )

    def list_daily_stats(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM daily_stats ORDER BY day DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # 快照
    # ------------------------------------------------------------------ #
    def insert_snapshot(
        self, *, ts: float, track_id: int | None, behavior: str | None, path: str,
    ) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO snapshots (ts, track_id, behavior, path)
                VALUES (?, ?, ?, ?)
                """,
                (ts, track_id, behavior, path),
            )
            return cur.lastrowid

    def list_snapshots(self, limit: int = 60) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM snapshots ORDER BY ts DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # CSV 导出
    # ------------------------------------------------------------------ #
    def export_events_csv(self, path: str | Path) -> int:
        """把 pet_events 全部导出为 CSV，返回导出行数。"""
        import csv

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT ts, track_id, species, behavior, confidence,
                       bbox_x1, bbox_y1, bbox_x2, bbox_y2, source
                FROM pet_events ORDER BY ts ASC
                """
            ).fetchall()
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["timestamp", "track_id", "species", "behavior",
                        "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                        "source"])
            for r in rows:
                w.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["ts"])),
                    r["track_id"], r["species"], r["behavior"],
                    f"{r['confidence']:.4f}",
                    r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"],
                    r["source"],
                ])
        return len(rows)

    # ------------------------------------------------------------------ #
    # 维护
    # ------------------------------------------------------------------ #
    def vacuum(self) -> None:
        with self._connect() as conn:
            conn.execute("VACUUM")

    def purge_older_than(self, days: int) -> int:
        cutoff = time.time() - days * 86400
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM pet_events WHERE ts < ?", (cutoff,))
            return cur.rowcount


# 单例便捷访问
_default_db: Database | None = None


def get_db() -> Database:
    global _default_db
    if _default_db is None:
        _default_db = Database()
    return _default_db


if __name__ == "__main__":
    # 简单冒烟测试
    db = Database()
    db.insert_event(
        ts=time.time(), track_id=1, species="cat", behavior="resting",
        confidence=0.9, bbox=(10, 20, 100, 200), source="test",
    )
    print(json.dumps({
        "events": db.count_events(),
        "recent": db.recent_events(5),
    }, ensure_ascii=False, indent=2, default=str))