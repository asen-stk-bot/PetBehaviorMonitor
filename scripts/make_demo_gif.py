"""
生成演示 GIF / Demo GIF Generator

读取内置猫/狗视频，用 YOLOv8s 检测 + 行为判定，在帧上画：
- 检测框 + 物种/ID/置信度
- 行为中文标签
- 进度条 + 时间戳
- 行为分布面板（右下角）

输出：data/demo.gif（适合 PPT/课程展示用）。

用法：
    python scripts/make_demo_gif.py
    python scripts/make_demo_gif.py --source data/videos/dog_real.mp4 --out data/demo_dog.gif
"""
from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pet_monitor.core.behavior import BEHAVIOR_LABELS_CN, BehaviorAnalyzer  # noqa: E402
from pet_monitor.core.detector import PetDetector  # noqa: E402
from pet_monitor.config import CONFIG, load_config  # noqa: E402

# 检测框配色（BGR）
SPECIES_COLOR = {"cat": (255, 200, 100), "dog": (100, 200, 255), "pet": (200, 200, 200)}
BEHAVIOR_COLOR = {
    "resting":  (180, 130, 60),
    "active":   (60, 200, 130),
    "eating":   (60, 140, 240),
    "drinking": (180, 200, 60),
    "anomaly":  (40, 40, 240),
}


def _draw_dashboard(frame: np.ndarray, beh_count: Counter, fps: float,
                    frame_idx: int, total: int) -> np.ndarray:
    """在右下角画行为分布小面板 + 顶部进度条 + 时间戳。"""
    h, w = frame.shape[:2]
    out = frame.copy()

    # 顶部进度条
    bar_h = 6
    cv2.rectangle(out, (0, 0), (w, bar_h), (40, 40, 40), -1)
    fill = int(w * (frame_idx / max(1, total - 1)))
    cv2.rectangle(out, (0, 0), (fill, bar_h), (90, 180, 90), -1)

    # 顶部时间戳
    ts_str = f"frame {frame_idx}/{total}  fps {fps:.1f}"
    cv2.putText(out, ts_str, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 240, 240), 2)

    # 右下角行为分布面板
    panel_w, panel_h = 220, 24 + 18 * max(1, len([k for k, v in beh_count.items() if v > 0]))
    x0, y0 = w - panel_w - 10, h - panel_h - 10
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (30, 30, 30), -1)
    out = cv2.addWeighted(overlay, 0.65, out, 0.35, 0)

    cv2.putText(out, "Behavior Distribution", (x0 + 8, y0 + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)
    total_count = sum(beh_count.values()) or 1
    yy = y0 + 18 + 4
    for k, n in beh_count.most_common(6):
        if n <= 0:
            continue
        color = BEHAVIOR_COLOR.get(k, (180, 180, 180))
        pct = 100.0 * n / total_count
        cv2.putText(out, f"{BEHAVIOR_LABELS_CN.get(k, k):<5} {n:>4} {pct:>4.1f}%",
                    (x0 + 8, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        yy += 16

    # 标题（顶部居中）
    cv2.putText(out, "PetBehaviorMonitor Demo", (w // 2 - 160, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (250, 250, 250), 2)
    return out


def _draw_box(frame: np.ndarray, det, beh) -> np.ndarray:
    """在帧上画单个检测框 + 行为标签。"""
    x1, y1, x2, y2 = map(int, det.bbox)
    base_color = SPECIES_COLOR.get(det.species, (200, 200, 200))
    beh_color = BEHAVIOR_COLOR.get(beh.behavior if beh else "unknown", base_color)
    cv2.rectangle(frame, (x1, y1), (x2, y2), beh_color, 2)
    label_cn = BEHAVIOR_LABELS_CN.get(beh.behavior if beh else "unknown", "?")
    label = f"{det.species}#{det.track_id} {label_cn} {det.confidence:.2f}"
    ty = max(20, y1 - 8)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, ty - th - 4), (x1 + tw + 4, ty + 2), (0, 0, 0), -1)
    cv2.putText(frame, label, (x1 + 2, ty - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, beh_color, 1)
    if beh and beh.is_anomaly:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
    return frame


def make_gif(source: Path, out: Path, stride: int = 2, max_frames: int = 240,
             gif_fps: int = 8) -> None:
    # 显式加载 config.json，确保使用项目推荐的 yolov8s + conf=0.20
    load_config()
    print(f"[gif] 检测器: {CONFIG.detector.model_name}  conf={CONFIG.detector.conf_threshold}")
    detector = PetDetector()
    analyzer = BehaviorAnalyzer()
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[gif] 视频: {source.name}  总帧 {total}  stride={stride}  目标帧 {min(max_frames, total // stride)}")

    frames: list[np.ndarray] = []
    beh_count: Counter = Counter()
    t0 = time.time()
    idx = 0
    last_fps = 0.0
    frame_t_last = t0
    frame_count_window = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % stride != 0:
            idx += 1
            continue
        # 检测
        dets = detector.detect(frame)
        h, w = frame.shape[:2]
        outs = analyzer.update(dets, frame_size=(w, h), ts=time.time(), frame=frame)
        # 计数
        for d, b in zip(dets, outs):
            beh_count[b.behavior] += 1
        # 绘制
        for d, b in zip(dets, outs):
            frame = _draw_box(frame, d, b)
        # FPS
        now = time.time()
        frame_count_window += 1
        if now - frame_t_last >= 1.0:
            last_fps = frame_count_window / (now - frame_t_last)
            frame_count_window = 0
            frame_t_last = now
        frame = _draw_dashboard(frame, beh_count, last_fps, idx, total)
        # 转 RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 缩小到 480 宽度减少 GIF 体积
        if w > 480:
            scale = 480 / w
            rgb = cv2.resize(rgb, (480, int(h * scale)), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
        idx += 1
        if len(frames) >= max_frames:
            break
        if idx % 30 == 0:
            print(f"  ... {idx}/{total}  累计 {len(frames)} 帧  实时 FPS {last_fps:.1f}")

    cap.release()
    print(f"[gif] 共采集 {len(frames)} 帧  总耗时 {time.time() - t0:.1f}s  行为分布 {dict(beh_count)}")
    print(f"[gif] 写 GIF: {out}  fps={gif_fps}  duration={len(frames) / gif_fps:.1f}s")
    out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(out), frames, fps=gif_fps, loop=0)
    print(f"[gif] 完成: {out}  ({out.stat().st_size / 1024:.1f} KB)")


def main() -> int:
    p = argparse.ArgumentParser(description="生成演示 GIF")
    p.add_argument("--source", default=str(ROOT / "data" / "videos" / "pets_real.mp4"),
                   help="输入视频路径")
    p.add_argument("--out", default=str(ROOT / "data" / "demo.gif"),
                   help="输出 GIF 路径")
    p.add_argument("--stride", type=int, default=2, help="每 N 帧采 1 帧")
    p.add_argument("--max-frames", type=int, default=240, help="最大帧数")
    p.add_argument("--fps", type=int, default=8, help="GIF 帧率")
    args = p.parse_args()
    make_gif(Path(args.source), Path(args.out), args.stride, args.max_frames, args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
