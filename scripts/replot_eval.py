"""读取 eval_results.json 重画柱状图（修复中文字体）。独立脚本，不依赖 eval_models 的模块全局量。"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JSON = ROOT / "data" / "_eval" / "eval_results.json"
OUT = ROOT / "data" / "_eval" / "eval_coverage.png"


def main() -> None:
    results = json.loads(JSON.read_text(encoding="utf-8"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    videos = sorted({r["video_label"] for r in results})
    fig, ax = plt.subplots(figsize=(11, 6.2))
    width = 0.13
    for vi, vlabel in enumerate(videos):
        sub = [r for r in results if r["video_label"] == vlabel]
        for i, r in enumerate(sub):
            color = "#4C9EEB" if r["model_tag"] == "v8s" else "#F5A623"
            ax.bar(vi * 5 + i * width, r["coverage_pct"], width,
                   color=color, edgecolor="#333", linewidth=0.4)
    ax.set_title("YOLOv8n vs YOLOv8s 检测覆盖率对比（猫/狗真实视频）", fontsize=13)
    ax.set_ylabel("检测覆盖率 %（有检测帧 / 采样帧）", fontsize=11)
    ax.set_ylim(0, 108)
    ax.legend(handles=[Patch(facecolor="#F5A623", label="YOLOv8n"),
                       Patch(facecolor="#4C9EEB", label="YOLOv8s")],
              loc="upper right")
    # 每个 video 分组下方标注，组内标 conf
    for vi, vlabel in enumerate(videos):
        sub = [r for r in results if r["video_label"] == vlabel]
        center = vi * 5 + (len(sub) * width) / 2
        ax.text(center, -7, vlabel, ha="center", fontsize=11, fontweight="bold")
        for i, r in enumerate(sub):
            x = vi * 5 + i * width
            ax.text(x + width / 2, r["coverage_pct"] + 1.2,
                    f"{r['coverage_pct']}", ha="center", va="bottom", fontsize=7.5)
            ax.text(x + width / 2, -2.8, str(r["conf"]), ha="center", fontsize=7, color="#666")
    ax.set_xticks([])
    ax.text(0.5, 0.96, "柱顶数字=覆盖率%，柱下数字=置信度阈值",
            transform=ax.transAxes, ha="center", fontsize=8.5, color="#888")
    plt.tight_layout()
    plt.savefig(OUT, dpi=130, bbox_inches="tight")
    print(f"[replot] 已写: {OUT}")


if __name__ == "__main__":
    main()
