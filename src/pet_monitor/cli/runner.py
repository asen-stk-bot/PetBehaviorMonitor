"""
CLI 彩打模式 / Command-line colored output

不依赖 PyQt5，用 rich 库在控制台实时打印：
- 每帧检测 / 行为结果
- 报警事件立即高亮
- 周期性统计面板（行为分布 + 累计报警）

入口：
    from pet_monitor.cli.runner import run_cli_mode
    run_cli_mode(source="data/videos/pets_real.mp4")
"""
from __future__ import annotations

import threading
import time
from collections import Counter

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from ..core.behavior import BEHAVIOR_LABELS_CN
from ..core.alert import LEVEL_CRIT, get_alert
from ..core.monitor import Monitor

console = Console()

# 行为 emoji + 颜色映射
BEHAVIOR_STYLE = {
    "resting":  ("💤", "blue"),
    "active":   ("🏃", "green"),
    "eating":   ("🍽", "yellow"),
    "drinking": ("💧", "cyan"),
    "anomaly":  ("⚠ ", "bold red"),
    "unknown":  ("❓", "dim"),
}
SPECIES_EMOJI = {"cat": "🐱", "dog": "🐶", "pet": "🐾"}


class _Stats:
    """CLI 模式下的内存统计：行为帧数 + 报警计数 + FPS。"""
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.behavior_count: Counter = Counter()
        self.species_count: Counter = Counter()
        self.alert_count: int = 0
        self.recent_alerts: list[tuple[float, str, str]] = []
        self.last_fps: float = 0.0
        self.total_frames: int = 0
        self.detected_frames: int = 0
        self._t0 = time.time()

    def record_frame(self, r) -> None:
        with self.lock:
            self.total_frames += 1
            if r.detections:
                self.detected_frames += 1
            for d in r.detections:
                self.species_count[d.species] += 1
            for b in r.behaviors:
                self.behavior_count[b.behavior] += 1
            if r.fps > 0:
                self.last_fps = r.fps

    def record_alert(self, level: str, message: str) -> None:
        with self.lock:
            self.alert_count += 1
            self.recent_alerts.append((time.time(), level, message))
            self.recent_alerts = self.recent_alerts[-5:]

    def render_stats_panel(self) -> Panel:
        with self.lock:
            elapsed = max(0.1, time.time() - self._t0)
            coverage = (
                f"{100.0 * self.detected_frames / max(1, self.total_frames):.1f}%"
                if self.total_frames else "-"
            )
            tbl = Table(title="[bold]行为分布[/]", show_header=True, header_style="bold")
            tbl.add_column("行为", style="cyan", no_wrap=True)
            tbl.add_column("帧数", justify="right")
            tbl.add_column("占比", justify="right")
            total = sum(self.behavior_count.values()) or 1
            for k in ["resting", "active", "eating", "drinking", "anomaly", "unknown"]:
                if self.behavior_count.get(k, 0) == 0:
                    continue
                emoji, color = BEHAVIOR_STYLE.get(k, ("·", "white"))
                label = BEHAVIOR_LABELS_CN.get(k, k)
                n = self.behavior_count[k]
                tbl.add_row(
                    f"[{color}]{emoji} {label}[/]",
                    f"{n:,}",
                    f"{100.0 * n / total:.1f}%",
                )
            sp_tbl = Table(title="[bold]物种检测[/]", show_header=True, header_style="bold")
            sp_tbl.add_column("物种", style="cyan")
            sp_tbl.add_column("次数", justify="right")
            for k, n in self.species_count.most_common():
                sp_tbl.add_row(f"{SPECIES_EMOJI.get(k, '🐾')} {k}", f"{n:,}")

            body = Table.grid(padding=1)
            body.add_column()
            body.add_row(
                f"[bold]运行时长[/]   {elapsed:.1f} s\n"
                f"[bold]总帧数[/]     {self.total_frames:,}  "
                f"[dim](检测帧 {self.detected_frames:,} · 覆盖率 {coverage})[/]\n"
                f"[bold]FPS[/]        {self.last_fps:.1f}\n"
                f"[bold]报警累计[/]   {self.alert_count}"
            )
            body.add_row(tbl)
            body.add_row(sp_tbl)

            if self.recent_alerts:
                lines = []
                for ts, lvl, msg in self.recent_alerts[-3:]:
                    icon = "🔴" if lvl == "critical" else "🟡" if lvl == "warning" else "🟢"
                    lines.append(f"{icon} [{lvl}] {msg}")
                body.add_row(Panel("\n".join(lines), title="[bold]最近报警[/]", border_style="red"))
        return Panel(body, title="[bold cyan]宠物行为监测 · CLI[/]", border_style="cyan")


def _on_alert_print(stats: _Stats):
    """报警回调：把 alert 输出打到屏幕并入统计。"""
    def cb(rec) -> None:
        ts_str = time.strftime("%H:%M:%S", time.localtime(rec.ts))
        if rec.level == LEVEL_CRIT:
            console.print(f"  [bold red]🔴 [{rec.level}] {ts_str} {rec.message}[/]")
        elif rec.level == "warning":
            console.print(f"  [bold yellow]🟡 [{rec.level}] {ts_str} {rec.message}[/]")
        else:
            console.print(f"  [bold green]🟢 [{rec.level}] {ts_str} {rec.message}[/]")
        stats.record_alert(rec.level, rec.message)
    return cb


def run_cli_mode(source: int | str = 0, refresh_per_sec: float = 4.0) -> int:
    """CLI 入口：实时打检测/行为/报警，Ctrl+C 优雅退出。"""
    console.print(
        Panel.fit(
            f"[bold cyan]宠物行为识别监测系统 · CLI 模式[/]\n"
            f"[dim]视频源: {source}  |  按 Ctrl+C 停止[/]",
            border_style="cyan",
        )
    )

    stats = _Stats()
    alert = get_alert()
    alert.set_popup_callback(_on_alert_print(stats))

    monitor = Monitor(source=source, on_result=lambda r: stats.record_frame(r))
    monitor.start()
    console.print("[green]✓[/] 监控已启动")

    try:
        with Live(stats.render_stats_panel(), refresh_per_second=refresh_per_sec,
                  console=console, screen=False) as live:
            while monitor.is_running():
                time.sleep(0.5)
                live.update(stats.render_stats_panel())
    except KeyboardInterrupt:
        console.print("\n[yellow]收到 Ctrl+C，正在停止…[/]")
    finally:
        monitor.stop()
        console.print(
            Panel(
                f"总帧数: {stats.total_frames:,}  "
                f"检测覆盖: {100.0 * stats.detected_frames / max(1, stats.total_frames):.1f}%\n"
                f"报警累计: {stats.alert_count}  "
                f"行为分布: " + ", ".join(
                    f"{BEHAVIOR_LABELS_CN.get(k, k)}={n}"
                    for k, n in stats.behavior_count.most_common()
                ),
                title="[bold green]运行结束[/]",
                border_style="green",
            )
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=0)
    args = p.parse_args()
    src = int(args.source) if str(args.source).isdigit() else args.source
    raise SystemExit(run_cli_mode(source=src))
