"""
视频源解析：从下拉框的"内置"条目（userData）映射到 data/videos/ 下的实际文件路径。
让 GUI 可以一键切换演示视频源，无需手填路径。
"""
from __future__ import annotations
from pathlib import Path

VIDEO_DIR = Path(__file__).resolve().parents[3] / "data" / "videos"

# 内置视频源 key -> 友好中文标签
BUILTIN_SOURCE_LABELS = {
    "pets_real": "🐱 内置·猫视频 (拼接)",
    "dog_real":  "🐶 内置·狗视频 (拼接)",
}

# key -> 实际文件名
BUILTIN_SOURCE_FILES = {
    "pets_real": "pets_real.mp4",
    "dog_real":  "dog_real.mp4",
}


def list_builtin_sources() -> list[str]:
    """返回 data/videos 下实际存在的内置源 key 列表。"""
    out = []
    for k, fn in BUILTIN_SOURCE_FILES.items():
        if (VIDEO_DIR / fn).exists():
            out.append(k)
    return out


def resolve_builtin_path(key: str) -> str:
    """根据 key 返回视频绝对路径；找不到抛 FileNotFoundError。"""
    if key not in BUILTIN_SOURCE_FILES:
        raise ValueError(f"未知内置源: {key}")
    p = VIDEO_DIR / BUILTIN_SOURCE_FILES[key]
    if not p.exists():
        raise FileNotFoundError(f"内置视频不存在: {p}")
    return str(p)
