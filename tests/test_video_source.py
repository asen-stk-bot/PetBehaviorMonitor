"""
视频源解析单测 / video_source unit tests

覆盖：
- list_builtin_sources 至少返回一个真实素材
- resolve_builtin_path 返回正确路径
- 未知 key / 不存在文件抛异常
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pet_monitor.core.video_source import (  # noqa: E402
    BUILTIN_SOURCE_LABELS,
    BUILTIN_SOURCE_FILES,
    list_builtin_sources,
    resolve_builtin_path,
)


class TestVideoSource(unittest.TestCase):
    def test_list_builtin_returns_at_least_one(self) -> None:
        """本项目打包了 pets_real.mp4，应至少返回一个内置源。"""
        items = list_builtin_sources()
        self.assertGreaterEqual(len(items), 1, "内置源应至少 1 个")
        for k in items:
            self.assertIn(k, BUILTIN_SOURCE_FILES)

    def test_resolve_builtin_path_returns_existing_file(self) -> None:
        """对 list_builtin_sources 返回的 key，解析路径应存在。"""
        for k in list_builtin_sources():
            p = resolve_builtin_path(k)
            self.assertTrue(Path(p).exists(), f"内置源 {k} 路径不存在: {p}")

    def test_resolve_unknown_key_raises(self) -> None:
        with self.assertRaises(ValueError):
            resolve_builtin_path("not_a_key_xyz")

    def test_labels_have_emoji(self) -> None:
        """中文标签便于 GUI 展示。"""
        for k, label in BUILTIN_SOURCE_LABELS.items():
            self.assertTrue(len(label) > 0, f"{k} 标签为空")


if __name__ == "__main__":
    unittest.main()
