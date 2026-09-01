"""筛选检测稳定的片段并拼接成 pets_real.mp4（每段播 3 遍延长时长）。

筛选标准：5 个抽帧点中 >= 4 个能以 >= 0.25 置信度检测到猫/狗。
"""
import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

CONF = 0.25
W, H, FPS = 640, 640, 30

model = YOLO("data/models/yolov8n.pt")
clips = sorted(glob.glob("data/videos/pets/*.mp4"))
print(f"候选片段: {len(clips)} 个")

good = []
for cp in clips:
    cap = cv2.VideoCapture(cp)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        continue
    hits = 0
    probes = 5
    for i in range(probes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / probes))
        ok, frame = cap.read()
        if not ok:
            continue
        for r in model(frame, conf=CONF, verbose=False):
            if any(int(b.cls[0]) in (15, 16) for b in r.boxes):
                hits += 1
                break
    cap.release()
    if hits >= 3:
        good.append(cp)
        print(f"  保留 {os.path.basename(cp)} ({hits}/{probes})")

print(f"保留 {len(good)} 个稳定片段")
if not good:
    raise SystemExit("没有稳定片段")

out_path = "data/videos/pets_real.mp4"
writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
total_out = 0
for cp in good:
    frames = []
    cap = cv2.VideoCapture(cp)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    for frame in frames * 3:  # 播 3 遍
        h, w = frame.shape[:2]
        s = min(W / w, H / h)
        nw, nh = int(w * s), int(h * s)
        resized = cv2.resize(frame, (nw, nh))
        # 用模拟室内背景代替纯黑（提升 YOLO 检测友好度）
        canvas = np.full((H, W, 3), (200, 190, 180), dtype=np.uint8)
        y0, x0 = (H - nh) // 2, (W - nw) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = resized
        writer.write(canvas)
        total_out += 1
writer.release()
print(f"输出: {out_path}")
print(f"总时长: {total_out / FPS:.1f} 秒 ({total_out} 帧)")
print(f"大小: {os.path.getsize(out_path) / 1024 / 1024:.2f} MB")
