"""
全局配置 / Configuration

集中管理项目路径、模型选择、行为识别阈值、报警规则等可调参数。
可通过环境变量覆盖，便于在不同机器上无需改代码即可调整。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------------- #
# 路径
# --------------------------------------------------------------------------- #
def project_root() -> Path:
    """src/pet_monitor/config.py -> 项目根目录 PetBehaviorMonitor/"""
    return Path(__file__).resolve().parents[2]


PROJECT_ROOT: Path = project_root()
DATA_DIR: Path = PROJECT_ROOT / "data"
DOCS_DIR: Path = PROJECT_ROOT / "docs"
MODEL_DIR: Path = DATA_DIR / "models"
VIDEO_DIR: Path = DATA_DIR / "videos"
SNAPSHOT_DIR: Path = DATA_DIR / "snapshots"
REPORT_DIR: Path = DATA_DIR / "reports"
DATASET_DIR: Path = DATA_DIR / "dataset"
DB_PATH: Path = DATA_DIR / "pet_monitor.db"
CONFIG_PATH: Path = DATA_DIR / "config.json"

for _d in (DATA_DIR, MODEL_DIR, VIDEO_DIR, SNAPSHOT_DIR, REPORT_DIR, DATASET_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# YOLO 检测配置
# --------------------------------------------------------------------------- #
@dataclass
class DetectorConfig:
    """YOLO 检测器配置。

    Attributes:
        model_name: ultralytics 模型名（自动下载），nano 最小最快。
        conf_threshold: 置信度阈值。
        iou_threshold: NMS IoU 阈值。
        device: 'cpu' / 'cuda:0'，空字符串表示自动。
        imgsz: 推理输入尺寸。
        pet_class_ids: COCO 中视为宠物的类别 id（猫=15、狗=16）。
    """
    model_name: str = os.environ.get("PET_MODEL", "yolov8n.pt")
    conf_threshold: float = float(os.environ.get("PET_CONF", "0.25"))
    iou_threshold: float = float(os.environ.get("PET_IOU", "0.45"))
    device: str = os.environ.get("PET_DEVICE", "")
    imgsz: int = 640
    pet_class_ids: tuple = (15, 16)  # cat, dog in COCO
    # 跟踪器：True 用 ultralytics 内置 ByteTrack（更专业、ID 更稳），
    # False 退回自写 IoU 贪心跟踪（detector.SimpleTracker，便于教学对比）
    use_bytetrack: bool = True


# --------------------------------------------------------------------------- #
# 行为识别配置（基于几何/运动启发式，见 core/behavior.py）
# --------------------------------------------------------------------------- #
@dataclass
class BehaviorConfig:
    """行为识别阈值。

    所有阈值都基于归一化的 bbox / 像素位移，可在实际场景中再校准。
    """
    # 帧间最大跟踪丢失容忍（用于简单 IoU 跟踪）
    track_max_missing: int = 15

    # 静止判定：bbox 中心在最近 N 帧内累计位移小于 S 像素
    resting_motion_px: float = 6.0
    resting_window: int = 30

    # 活跃判定：单位时间位移超过该值
    active_motion_px_per_frame: float = 8.0

    # 长时静止视为休息
    rest_min_duration_sec: float = 10.0

    # 长时无活动 + bbox 几乎不动 → 疑似异常（晕厥/抽搐）
    anomaly_min_duration_sec: float = 60.0
    anomaly_jitter_px: float = 1.5        # 静止但有微小抖动

    # 进食/饮水区 ROI 归一化坐标 (x1,y1,x2,y2)，可在 GUI 中调整
    food_roi: tuple = (0.05, 0.55, 0.45, 0.95)
    water_roi: tuple = (0.55, 0.55, 0.95, 0.95)

    # 在 ROI 内停留多久才判定进食/饮水
    consum_min_dwell_sec: float = 3.0

    # 光流辅助（Farneback）：用全局光流幅值辅助 active/resting 判定，
    # 减少 bbox 中心检测抖动导致的误判。None/False 关闭。
    enable_optical_flow: bool = True
    flow_downscale: int = 320          # 光流计算前缩放到该宽度，加速
    flow_active_threshold: float = 4.0  # 光流幅值高于此 → 倾向 active
    flow_resting_threshold: float = 1.5  # 光流幅值低于此 → 强化 resting


# --------------------------------------------------------------------------- #
# 报警配置
# --------------------------------------------------------------------------- #
@dataclass
class AlertConfig:
    """报警系统配置。"""
    enable_sound: bool = True
    cooldown_sec: float = 30.0       # 同一类型报警冷却时间
    enable_desktop_popup: bool = True
    log_to_db: bool = True


# --------------------------------------------------------------------------- #
# 数据库配置
# --------------------------------------------------------------------------- #
@dataclass
class DBConfig:
    path: Path = DB_PATH
    retention_days: int = 30         # 监控事件保留天数


# --------------------------------------------------------------------------- #
# UI 配置
# --------------------------------------------------------------------------- #
@dataclass
class UIConfig:
    window_title: str = "宠物行为识别监测系统 v1.0"
    window_size: tuple = (1280, 760)
    video_fps_default: int = 25


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #
@dataclass
class AppConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    alert: AlertConfig = field(default_factory=AlertConfig)
    db: DBConfig = field(default_factory=DBConfig)
    ui: UIConfig = field(default_factory=UIConfig)


CONFIG = AppConfig()


# --------------------------------------------------------------------------- #
# 配置持久化：跨会话保存 / 恢复可调参数
# --------------------------------------------------------------------------- #
def _serializable() -> dict:
    return {
        "detector": {
            "model_name": CONFIG.detector.model_name,
            "conf_threshold": CONFIG.detector.conf_threshold,
            "iou_threshold": CONFIG.detector.iou_threshold,
            "device": CONFIG.detector.device,
            "imgsz": CONFIG.detector.imgsz,
            "use_bytetrack": CONFIG.detector.use_bytetrack,
        },
        "behavior": {
            "resting_motion_px": CONFIG.behavior.resting_motion_px,
            "resting_window": CONFIG.behavior.resting_window,
            "active_motion_px_per_frame": CONFIG.behavior.active_motion_px_per_frame,
            "rest_min_duration_sec": CONFIG.behavior.rest_min_duration_sec,
            "anomaly_min_duration_sec": CONFIG.behavior.anomaly_min_duration_sec,
            "anomaly_jitter_px": CONFIG.behavior.anomaly_jitter_px,
            "consum_min_dwell_sec": CONFIG.behavior.consum_min_dwell_sec,
            "enable_optical_flow": CONFIG.behavior.enable_optical_flow,
            "flow_active_threshold": CONFIG.behavior.flow_active_threshold,
            "flow_resting_threshold": CONFIG.behavior.flow_resting_threshold,
            "food_roi": list(CONFIG.behavior.food_roi),
            "water_roi": list(CONFIG.behavior.water_roi),
        },
        "alert": {
            "enable_sound": CONFIG.alert.enable_sound,
            "cooldown_sec": CONFIG.alert.cooldown_sec,
            "enable_desktop_popup": CONFIG.alert.enable_desktop_popup,
        },
    }


def save_config(path: Path | None = None) -> Path:
    """把当前可调参数写入 JSON（默认 data/config.json），返回写入路径。"""
    import json

    p = Path(path) if path else CONFIG_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_serializable(), ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def load_config(path: Path | None = None) -> bool:
    """从 JSON 恢复可调参数；文件不存在或非法时返回 False 且不抛异常。"""
    import json

    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return False

    d = data.get("detector", {})
    for k in ("model_name", "conf_threshold", "iou_threshold", "imgsz"):
        if k in d and d[k] is not None:
            setattr(CONFIG.detector, k, d[k])
    if d.get("device") is not None:
        CONFIG.detector.device = d["device"]
    if "use_bytetrack" in d and d["use_bytetrack"] is not None:
        CONFIG.detector.use_bytetrack = bool(d["use_bytetrack"])

    b = data.get("behavior", {})
    for k in ("resting_motion_px", "resting_window", "active_motion_px_per_frame",
              "rest_min_duration_sec", "anomaly_min_duration_sec",
              "anomaly_jitter_px", "consum_min_dwell_sec",
              "flow_active_threshold", "flow_resting_threshold"):
        if k in b and b[k] is not None:
            setattr(CONFIG.behavior, k, b[k])
    if "enable_optical_flow" in b and b["enable_optical_flow"] is not None:
        CONFIG.behavior.enable_optical_flow = bool(b["enable_optical_flow"])
    if isinstance(b.get("food_roi"), list) and len(b["food_roi"]) == 4:
        CONFIG.behavior.food_roi = tuple(b["food_roi"])
    if isinstance(b.get("water_roi"), list) and len(b["water_roi"]) == 4:
        CONFIG.behavior.water_roi = tuple(b["water_roi"])

    a = data.get("alert", {})
    for k in ("enable_sound", "cooldown_sec", "enable_desktop_popup"):
        if k in a and a[k] is not None:
            setattr(CONFIG.alert, k, a[k])

    return True