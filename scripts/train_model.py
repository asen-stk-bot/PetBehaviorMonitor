"""
模型训练 / YOLOv8 Fine-tuning

在 COCO 预训练权重 yolov8n.pt 基础上，用自定义宠物数据集微调，
提升对特定宠物/行为的检测能力（例如家庭场景下的猫狗品种）。

支持两种数据集来源：
1. 本地 YOLO 格式目录（推荐）：
       dataset/
         images/train/*.jpg
         images/val/*.jpg
         labels/train/*.txt   # YOLO 格式：class x_center y_center w h（归一化）
         labels/val/*.txt
2. Roboflow 下载的目录（含 data.yaml，可直接用）

用法：
    python main.py --train --data path/to/dataset --epochs 50

或直接：
    python scripts/train_model.py --data path/to/dataset --epochs 50

训练完成后：
    - 最佳权重保存到 data/models/best.pt
    - 运行监测时指定 PET_MODEL=best.pt 即可加载自训练模型
    - 训练曲线 / 验证结果保存到 runs/ 目录
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许直接运行本脚本
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pet_monitor.config import DATASET_DIR, MODEL_DIR  # noqa: E402


# 可选类别（可自行扩展；编号从 0 开始）
# 注意：训练/推理必须使用同一份标签映射
CLASS_NAMES = ["cat", "dog"]


def write_dataset_yaml(dataset_dir: Path, out_yaml: Path, class_names: list[str]) -> Path:
    """根据本地 YOLO 目录生成 data.yaml（Ultralytics 训练所需）。"""
    images_train = dataset_dir / "images" / "train"
    images_val = dataset_dir / "images" / "val"
    if not images_train.exists() and (dataset_dir / "train" / "images").exists():
        # Roboflow 结构：train/images, valid/images
        images_train = dataset_dir / "train" / "images"
        images_val = dataset_dir / "valid" / "images"

    names = {i: n for i, n in enumerate(class_names)}
    content = (
        f"path: {dataset_dir.resolve()}\n"
        f"train: {images_train.resolve()}\n"
        f"val: {images_val.resolve()}\n"
        f"\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in names.items())
    )
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    out_yaml.write_text(content, encoding="utf-8")
    return out_yaml


def count_images(d: Path) -> int:
    if not d.exists():
        return 0
    return len([p for p in d.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])


def main() -> int:
    p = argparse.ArgumentParser(description="YOLOv8 宠物检测微调")
    p.add_argument("--data", help="数据集目录（YOLO 格式）")
    p.add_argument("--epochs", type=int, default=50, help="训练轮数（默认 50）")
    p.add_argument("--imgsz", type=int, default=640, help="输入尺寸")
    p.add_argument("--batch", type=int, default=16, help="批大小")
    p.add_argument("--model", default="yolov8n.pt", help="预训练权重")
    p.add_argument("--device", default="", help="'cpu' / 'cuda:0'，空=自动")
    p.add_argument("--dry-run", action="store_true", help="仅检查数据集与生成 yaml，不训练")
    args = p.parse_args()

    dataset_dir = Path(args.data) if args.data else DATASET_DIR
    if not dataset_dir.exists():
        print(f"[错误] 数据集目录不存在：{dataset_dir}")
        print("请准备 YOLO 格式数据集，或使用 --data 指定目录。")
        return 1

    yaml_path = dataset_dir / "data.yaml"
    # 若目录里没有 data.yaml，则自动生成
    if not yaml_path.exists():
        yaml_path = write_dataset_yaml(dataset_dir, yaml_path, CLASS_NAMES)
        print(f"[生成] 数据集配置: {yaml_path}")

    n_train = count_images(dataset_dir / "images" / "train") or \
        count_images(dataset_dir / "train" / "images")
    n_val = count_images(dataset_dir / "images" / "val") or \
        count_images(dataset_dir / "valid" / "images")
    print(f"[数据集] train={n_train} 张, val={n_val} 张")

    if n_train == 0:
        print("[错误] 训练集为空，请检查目录结构：")
        print("  dataset/images/train/*.jpg 与 dataset/labels/train/*.txt")
        return 1

    if args.dry_run:
        print("[dry-run] 数据集检查通过，未执行训练。")
        return 0

    # 延迟导入，未装 ultralytics 时给出友好提示
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[错误] 未安装 ultralytics，请先执行：pip install ultralytics")
        return 1

    model = YOLO(args.model)
    print(f"[训练] 开始微调：epochs={args.epochs}, imgsz={args.imgsz}, "
          f"batch={args.batch}, device={args.device or 'auto'}")
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device or None,
        project=str(ROOT / "runs"),
        name="pet_train",
        exist_ok=True,
    )

    # 把最佳权重复制到 models/，便于监测时直接使用
    best_src = Path(results.save_dir) / "weights" / "best.pt"
    best_dst = MODEL_DIR / "best.pt"
    if best_src.exists():
        import shutil
        shutil.copy2(best_src, best_dst)
        print(f"[完成] 最佳权重已保存: {best_dst}")
        print(f"[提示] 运行监测时指定环境变量：PET_MODEL=best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
