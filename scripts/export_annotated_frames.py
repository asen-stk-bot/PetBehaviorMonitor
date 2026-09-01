"""抽取视频帧 → 跑完整检测+行为识别 → 画框和中文标签 → 导出图片。

用法:
    python scripts/export_annotated_frames.py <视频路径> [输出目录] [帧数]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np

from pet_monitor.core.detector import PetDetector
from pet_monitor.core.behavior import BehaviorAnalyzer, BEHAVIOR_LABELS_CN
from pet_monitor.config import load_config

load_config()  # 加载 data/config.json（yolov8s + conf 0.2）

SPECIES_CN = {"cat": "猫", "dog": "狗"}
# 颜色 (BGR)
SPECIES_COLOR = {"cat": (0, 165, 255), "dog": (255, 140, 0)}  # 猫=橙, 狗=蓝
BEHAVIOR_COLOR = {
    "resting": (0, 200, 0),      # 绿
    "active": (255, 160, 0),     # 蓝橙
    "eating": (0, 120, 255),     # 橙
    "drinking": (200, 120, 0),   # 蓝
    "anomaly": (0, 0, 255),      # 红
    "unknown": (150, 150, 150),  # 灰
}


def main() -> None:
    video = sys.argv[1] if len(sys.argv) > 1 else "data/videos/pets_real.mp4"
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data/_annotated")
    n_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = PetDetector()
    analyzer = BehaviorAnalyzer()

    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # 逐帧扫描，收集「确实检测到宠物」的帧，间隔至少 0.4 秒
    start = int(fps * 1.5)  # 跳过开头（让跟踪器建立轨迹）
    min_gap = max(1, int(fps * 0.4))
    picked: list[int] = []
    last = -10**9
    for fi in range(start, total, 2):  # 步长 2 提速
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        dets = detector.detect(frame)
        if dets and (fi - last) >= min_gap:
            picked.append(fi)
            last = fi
        if len(picked) >= n_frames:
            break

    print(f"从 {total} 帧中选中 {len(picked)} 帧（含检测目标）")

    saved = 0
    for fi in picked:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        ts = fi / fps

        dets = detector.detect(frame)
        outs = analyzer.update(dets, (frame.shape[1], frame.shape[0]), ts=ts)
        by_tid = {o.track_id: o for o in outs}

        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            color = SPECIES_COLOR.get(d.species, (0, 255, 0))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            o = by_tid.get(d.track_id)
            behavior = o.behavior if o else "unknown"
            bcolor = BEHAVIOR_COLOR.get(behavior, (150, 150, 150))
            label = f"{SPECIES_CN.get(d.species, d.species)} {BEHAVIOR_LABELS_CN.get(behavior, behavior)} {d.confidence:.2f}"

            # 标签背景条
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), bcolor, -1)
            cv2.putText(frame, label, (x1 + 4, max(th + 4, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)

        # 顶部时间戳
        cv2.putText(frame, f"t={ts:.1f}s", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

        out_path = out_dir / f"frame_{saved:02d}.jpg"
        cv2.imwrite(str(out_path), frame)
        print(f"已导出 {out_path}  (检测 {len(dets)} 个目标)")
        saved += 1

    cap.release()
    print(f"\n共导出 {saved} 张标注帧到 {out_dir}")


if __name__ == "__main__":
    main()
