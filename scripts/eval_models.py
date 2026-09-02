"""
模型精度对比实验 / Model Accuracy Benchmark

对比 YOLOv8n vs YOLOv8s 在真实猫/狗视频上、不同置信度阈值下的检测表现。
输出 JSON 结果 + 控制台表格 + PNG 柱状图。

用法：
    python scripts/eval_models.py

抽帧策略：每隔 SAMPLE_STRIDE 帧采样一帧，覆盖率指标（有检测帧/总采样帧）保持有效。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "models"
VIDEO_DIR = ROOT / "data" / "videos"
OUT_DIR = ROOT / "data" / "_eval"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = ["yolov8n.pt", "yolov8s.pt"]
VIDEOS = {"pets_real.mp4": "猫(pets_real)", "dog_real.mp4": "狗(dog_real)"}
CONFS = [0.15, 0.20, 0.25]
PET_CLASSES = (15, 16)  # cat, dog in COCO
SAMPLE_STRIDE = 5  # 每5帧采1帧


def load_model(name: str):
    from ultralytics import YOLO
    p = MODEL_DIR / name
    if p.exists():
        return YOLO(str(p))
    return YOLO(name)  # 自动下载


def eval_one(model, video: Path, conf: float) -> dict:
    cap = cv2.VideoCapture(str(video))
    total = sampled = detected = 0
    sum_targets = 0
    sum_conf = 0.0
    species_counts = {15: 0, 16: 0}
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        total += 1
        if total % SAMPLE_STRIDE != 0:
            continue
        sampled += 1
        res = model.predict(
            frame, conf=conf, iou=0.45, imgsz=640,
            device="", verbose=False,
        )
        if not res:
            continue
        r = res[0]
        if r.boxes is None or r.boxes.xyxy is None or len(r.boxes) == 0:
            continue
        cls = r.boxes.cls.cpu().numpy().astype(int)
        pet_mask = np.isin(cls, list(PET_CLASSES))
        n_pet = int(pet_mask.sum())
        if n_pet == 0:
            continue
        detected += 1
        sum_targets += n_pet
        confs = r.boxes.conf.cpu().numpy()
        sum_conf += float(confs[pet_mask].mean()) if n_pet > 0 else 0.0
        for k in cls[pet_mask]:
            if k in species_counts:
                species_counts[k] += 1
    cap.release()
    dur = time.time() - t0
    return {
        "video": video.name,
        "model": model.model_name if hasattr(model, "model_name") else "?",
        "conf": conf,
        "total_frames": total,
        "sampled_frames": sampled,
        "detected_frames": detected,
        "coverage_pct": round(100.0 * detected / sampled, 1) if sampled else 0.0,
        "avg_targets": round(sum_targets / detected, 2) if detected else 0.0,
        "avg_conf": round(sum_conf / detected, 3) if detected else 0.0,
        "cat_detections": species_counts[15],
        "dog_detections": species_counts[16],
        "infer_time_sec": round(dur, 1),
    }


def main() -> None:
    print(f"[eval] 模型: {MODELS}")
    print(f"[eval] 视频: {list(VIDEOS)}")
    print(f"[eval] 置信度: {CONFS}")
    print(f"[eval] 抽帧步长: 每{SAMPLE_STRIDE}帧采1帧\n")

    results = []
    for mname in MODELS:
        print(f"[eval] 加载 {mname} ...")
        model = load_model(mname)
        for vname, vlabel in VIDEOS.items():
            vpath = VIDEO_DIR / vname
            if not vpath.exists():
                print(f"[eval] 跳过(不存在): {vpath}")
                continue
            for conf in CONFS:
                print(f"[eval] 跑 {mname} × {vlabel} × conf={conf} ...", end=" ", flush=True)
                rec = eval_one(model, vpath, conf)
                rec["video_label"] = vlabel
                rec["model_tag"] = "v8n" if "n" in mname else "v8s"
                results.append(rec)
                print(f"覆盖率={rec['coverage_pct']}% 均目标={rec['avg_targets']} 耗时={rec['infer_time_sec']}s")

    # 写 JSON
    out_json = OUT_DIR / "eval_results.json"
    out_json.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n[eval] JSON 已写: {out_json}")

    # 表格打印
    print("\n" + "=" * 92)
    print(f"{'模型':<6}{'视频':<14}{'conf':<7}{'采样帧':<8}{'检出帧':<8}{'覆盖率%':<10}{'均目标':<8}{'均置信':<8}")
    print("-" * 92)
    for r in results:
        print(f"{r['model_tag']:<6}{r['video_label']:<14}{r['conf']:<7}"
              f"{r['sampled_frames']:<8}{r['detected_frames']:<8}{r['coverage_pct']:<10}"
              f"{r['avg_targets']:<8}{r['avg_conf']:<8}")
    print("=" * 92)

    # 画柱状图
    try:
        _plot(results)
    except Exception as e:
        print(f"[eval] 画图失败(忽略): {e}")


def _plot(results: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # 中文字体：Windows 优先 Microsoft YaHei / SimHei，避免中文变方框
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    # 按 video 分组，每组按 model+conf 画覆盖率柱
    videos = sorted({r["video_label"] for r in results})
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.13
    labels_all = [f"{r['model_tag']}\nconf={r['conf']}" for r in results]
    # 简单按 video 分面板
    n_v = len(videos)
    for vi, vlabel in enumerate(videos):
        sub = [r for r in results if r["video_label"] == vlabel]
        x = np.arange(len(sub))
        for i, r in enumerate(sub):
            color = "#4C9EEB" if r["model_tag"] == "v8s" else "#F5A623"
            ax.bar(vi * 5 + i * width, r["coverage_pct"], width,
                   color=color, edgecolor="#333", linewidth=0.4)
    ax.set_title("YOLOv8n vs YOLOv8s 检测覆盖率对比（猫/狗真实视频）", fontsize=13)
    ax.set_ylabel("检测覆盖率 %（有检测帧 / 采样帧）")
    ax.set_ylim(0, 105)
    # 图例
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor="#F5A623", label="YOLOv8n"),
                       Patch(facecolor="#4C9EEB", label="YOLOv8s")],
              loc="upper right")
    # x 轴：按 video 分组标注
    ticks = []
    for vi, vlabel in enumerate(videos):
        sub = [r for r in results if r["video_label"] == vlabel]
        center = vi * 5 + (len(sub) * width) / 2
        ax.text(center, -8, vlabel, ha="center", fontsize=11)
    ax.set_xticks([])
    plt.tight_layout()
    out_png = OUT_DIR / "eval_coverage.png"
    plt.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"[eval] 柱状图已写: {out_png}")


if __name__ == "__main__":
    main()
