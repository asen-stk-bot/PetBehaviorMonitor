"""
程序入口 / Entry Point

启动方式：
    python main.py                       # 默认摄像头
    python main.py --video path.mp4      # 指定视频
    python main.py --demo                # 使用内置演示视频（自动生成）
    python main.py --tests               # 仅运行单元测试
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


# 让 `python main.py` 直接运行时能找到 src/
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="宠物行为识别监测系统",
    )
    p.add_argument("--video", help="视频文件路径")
    p.add_argument("--camera", type=int, default=0, help="摄像头编号（默认 0）")
    p.add_argument("--demo", action="store_true", help="使用内置演示视频")
    p.add_argument("--tests", action="store_true", help="运行单元测试后退出")
    p.add_argument("--gen-demo", action="store_true", help="仅生成演示视频后退出")
    p.add_argument("--report", action="store_true", help="生成 HTML 报告 + CSV 导出后退出")
    p.add_argument("--date", help="报告统计日期（YYYY-MM-DD，默认今天）")
    p.add_argument("--train", action="store_true", help="微调 YOLOv8 模型")
    p.add_argument("--data", help="训练数据集目录（配合 --train）")
    p.add_argument("--epochs", type=int, default=50, help="训练轮数")
    p.add_argument("--dry-run", action="store_true", help="训练前仅检查数据集（配合 --train）")
    p.add_argument("--log", default="INFO", help="日志级别")
    return p.parse_args()


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def ensure_demo() -> Path:
    """确保 data/videos/demo.mp4 存在；不存在则生成。"""
    from pet_monitor.config import VIDEO_DIR
    demo = VIDEO_DIR / "demo.mp4"
    if demo.exists() and demo.stat().st_size > 1024:
        return demo
    from scripts.generate_demo_video import generate
    return generate(demo)


def main() -> int:
    args = parse_args()
    setup_logging(args.log)

    if args.gen_demo:
        path = run_demo()
        print(f"演示视频已生成: {path}")
        return 0

    if args.tests:
        return run_tests()

    if args.report:
        return run_report(args.date)

    if args.train:
        return run_train(args)

    if args.demo:
        demo = ensure_demo()
        from pet_monitor.ui.main_window import MainWindow
        from PyQt5.QtWidgets import QApplication
        from PyQt5.QtCore import QTimer

        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        win = MainWindow()
        win.ed_source_path.setText(str(demo))
        win.show()
        QTimer.singleShot(500, win._start_monitor)
        return app.exec_()

    from pet_monitor.ui.main_window import run_app
    if args.video:
        # 通过环境变量把视频源传给 run_app：run_app 内部用 UI 默认摄像头 0；
        # 这里改用自定义启动：临时改 UI 默认值
        from pet_monitor.ui.main_window import MainWindow
        from PyQt5.QtWidgets import QApplication

        app = QApplication(sys.argv)
        app.setStyle("Fusion")
        win = MainWindow()
        win.ed_source_path.setText(args.video)
        win.show()
        return app.exec_()

    return run_app()


def run_demo() -> Path:
    """生成演示视频文件路径。"""
    from pet_monitor.config import VIDEO_DIR
    demo = VIDEO_DIR / "demo.mp4"
    from scripts.generate_demo_video import generate
    return generate(demo)


def run_tests() -> int:
    """运行单元测试。"""
    import unittest
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=str(ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return 0 if result.wasSuccessful() else 1


def run_report(date_str: str | None = None) -> int:
    """生成 HTML 报告 + CSV 导出。"""
    from pet_monitor.core.report import export_csv, generate_report

    html_path = generate_report(date_str=date_str)
    csv_path = export_csv()
    print(f"HTML 报告已生成: {html_path}")
    print(f"事件 CSV 已导出: {csv_path}")
    return 0


def run_train(args: argparse.Namespace) -> int:
    """微调 YOLOv8 模型（透传参数给训练脚本）。"""
    from scripts.train_model import main as train_main
    import sys as _sys

    filtered = []
    if args.data:
        filtered += ["--data", args.data]
    filtered += ["--epochs", str(args.epochs)]
    if args.dry_run:
        filtered.append("--dry-run")

    old_argv = _sys.argv
    _sys.argv = ["train_model.py"] + filtered
    try:
        return train_main()
    finally:
        _sys.argv = old_argv


if __name__ == "__main__":
    raise SystemExit(main())