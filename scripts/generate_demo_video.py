"""
生成演示视频 / Synthetic Demo Video Generator

在没有真实宠物视频时，使用 OpenCV 绘制一个"模拟宠物"以验证系统。
该模拟在画面上呈现：
- 一个橙色矩形（"猫"）在画面内做以下循环行为：
  1. 站立活动 8s
  2. 移到进食 ROI 区域（画面左下）停留 6s
  3. 移到饮水 ROI 区域（画面右下）停留 6s
  4. 回到画面中部静卧 12s
- 一个青色矩形（"狗"）随机小幅走动

用法：
    python -m scripts.generate_demo_video
    python scripts/generate_demo_video.py --out data/videos/demo.mp4
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path


def _script_path() -> Path:
    return Path(__file__).resolve()


# 添加 src/ 到 path
import sys
ROOT = _script_path().resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def generate(output: Path | str, width: int = 960, height: int = 540,
             fps: int = 25, duration_sec: int = 50) -> Path:
    """生成演示视频到 output，返回绝对路径。"""
    import cv2
    import numpy as np

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频写入器: {output}")

    cx, cy = width // 2, height // 2
    cat_size = (140, 100)
    dog_size = (160, 110)
    cat_color = (60, 140, 240)      # 蓝灰 (BGR)
    dog_color = (240, 160, 100)

    # 预定义动画阶段
    # (duration_frames, type) — type: "active", "food", "water", "rest", "dog"
    stages = [
        (8 * fps, "active"),
        (6 * fps, "food"),
        (6 * fps, "water"),
        (12 * fps, "rest"),
        (8 * fps, "active"),
        (10 * fps, "rest"),
    ]

    rng = random.Random(42)
    dog_traj = [(width * 0.7, height * 0.5)]
    frame_no = 0

    for stage_frames, kind in stages:
        for f in range(stage_frames):
            canvas = np.full((height, width, 3), 245, dtype=np.uint8)
            # 画 ROI（淡色框，提示应用逻辑区域）
            cv2.rectangle(canvas, (int(width * 0.05), int(height * 0.55)),
                          (int(width * 0.45), int(height * 0.95)),
                          (220, 230, 200), -1)
            cv2.rectangle(canvas, (int(width * 0.55), int(height * 0.55)),
                          (int(width * 0.95), int(height * 0.95)),
                          (220, 220, 240), -1)
            cv2.putText(canvas, "FOOD", (int(width * 0.06), int(height * 0.58)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 100, 80), 2)
            cv2.putText(canvas, "WATER", (int(width * 0.56), int(height * 0.58)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 100), 2)

            t = f / stage_frames
            if kind == "active":
                # 在画面内做正弦运动
                cx = int(width * 0.5 + math.sin(frame_no * 0.05) * width * 0.3)
                cy = int(height * 0.4 + math.cos(frame_no * 0.07) * height * 0.2)
                cat_color_now = cat_color
            elif kind == "food":
                cx = int(width * 0.25 + math.sin(frame_no * 0.1) * 10)
                cy = int(height * 0.7 + math.cos(frame_no * 0.1) * 5)
                cat_color_now = cat_color
            elif kind == "water":
                cx = int(width * 0.75 + math.sin(frame_no * 0.1) * 10)
                cy = int(height * 0.7 + math.cos(frame_no * 0.1) * 5)
                cat_color_now = cat_color
            else:  # rest
                cx = int(width * 0.5)
                cy = int(height * 0.55 + math.sin(frame_no * 0.02) * 1.5)
                cat_color_now = cat_color

            w_, h_ = cat_size
            x1, y1 = max(0, cx - w_ // 2), max(0, cy - h_ // 2)
            x2, y2 = min(width - 1, cx + w_ // 2), min(height - 1, cy + h_ // 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), cat_color_now, -1)
            cv2.putText(canvas, "CAT", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)

            # 狗：随机游走
            last_x, last_y = dog_traj[-1]
            dx = rng.uniform(-3, 3)
            dy = rng.uniform(-2, 2)
            new_x = max(50, min(width - 50, last_x + dx))
            new_y = max(50, min(height * 0.5, last_y + dy))
            dog_traj.append((new_x, new_y))
            if len(dog_traj) > 30:
                dog_traj.pop(0)
            w_, h_ = dog_size
            x1, y1 = int(new_x - w_ // 2), int(new_y - h_ // 2)
            x2, y2 = int(new_x + w_ // 2), int(new_y + h_ // 2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), dog_color, -1)
            cv2.putText(canvas, "DOG", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 2)

            # 帧信息
            cv2.putText(canvas, f"Synthetic Demo  frame {frame_no:05d}  stage={kind}",
                        (10, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (80, 80, 80), 1)
            writer.write(canvas)
            frame_no += 1

    writer.release()
    print(f"[demo] 写入完成: {output}  ({frame_no} frames)")
    return output


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "data" / "videos" / "demo.mp4"))
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--duration", type=int, default=50)
    a = p.parse_args()
    generate(a.out, a.width, a.height, a.fps, a.duration)