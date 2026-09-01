# 宠物行为识别监测系统 · 项目说明

> **南京工程学院 · 计算机工程学院 / 人工智能学院 · 软件工程232**
> 软件工程项目训练（指导教师：**张冰**）
> 课题：**宠物行为识别监测系统**（自选课题）

---

## 目录

| 文件 | 说明 |
| --- | --- |
| [01_需求分析.md](docs/01_需求分析.md) | 需求分析 |
| [02_系统设计.md](docs/02_系统设计.md) | 系统设计（架构 / 数据库 / 接口） |
| [03_测试报告.md](docs/03_测试报告.md) | 测试报告 |
| [04_用户手册.md](docs/04_用户手册.md) | 用户手册 |
| [05_实训总结.md](docs/05_实训总结.md) | 实训总结 |
| [06_数据集资源.md](docs/06_数据集资源.md) | 训练/测试数据集资源清单（含真实宠物行为视频） |
| [07_项目简介.md](docs/07_项目简介.md) | 项目简介与代码仓库链接 |
| [main.py](main.py) | 程序入口（GUI / 报告 / 训练 / 测试） |
| [src/pet_monitor/](src/pet_monitor) | 源代码 |
| [scripts/generate_demo_video.py](scripts/generate_demo_video.py) | 演示视频生成脚本 |
| [scripts/train_model.py](scripts/train_model.py) | YOLOv8 微调训练脚本 |
| [tests/test_units.py](tests/test_units.py) | 基础单元测试 |
| [tests/test_features.py](tests/test_features.py) | 新增功能单元测试（统计 / 快照 / 报告 / 配置） |
| [requirements.txt](requirements.txt) | Python 依赖 |

---

## 快速开始

### 方式 A：PyCharm 打开即用（推荐）

项目内已自带虚拟环境 `venv/`（Python 3.9，依赖已全部装好），打开即可运行：

1. PyCharm → **Open** → 选中本项目文件夹（`PetBehaviorMonitor`）
2. 等待右下角索引完成，确认解释器已自动识别为 `venv\Scripts\python.exe`
   （若未自动识别：**File → Settings → Project → Python Interpreter → 齿轮 → Add Local Interpreter → Existing → 选 `venv\Scripts\python.exe`**）
3. 直接运行：顶部工具栏**运行按钮旁的下拉框**里已预置两个运行配置（文件在 `.run/` 目录）：
   - **宠物监测-演示模式(--demo)**：播放内置演示视频，无需摄像头，打开即见效果
   - **宠物监测-摄像头模式**：使用本机摄像头实时监测

> 若下拉框没显示配置：右键 `main.py` → **Run 'main'** 先跑一次即可；或手动在 Parameters 里填 `--demo`。

### 方式 B：双击运行 / 其他电脑一键初始化

- 本机：双击 `run.bat`（自动用 `venv` 里的 Python 启动）
- 换机器或环境损坏：双击 `setup.bat` 一键重建虚拟环境并装依赖

### 方式 C：命令行

```bash
# 激活项目内虚拟环境
venv\Scripts\activate            # Windows

# 启动（默认使用本机摄像头）
python main.py

# 使用内置演示视频（无需真实宠物）
python main.py --demo

# 生成行为分析报告（HTML + CSV）
python main.py --report

# 微调 YOLOv8 模型（需准备数据集）
python main.py --train --data path/to/dataset --epochs 50

# 运行单元测试
python main.py --tests

# 打包为 exe（可选）
pyinstaller pet_monitor.spec
```

> **提示**：首次运行会自动下载 `yolov8n.pt`（约 6 MB），保存到 `data/models/` 目录（本项目已预置，无需再下载）。

---

## 版本管理与 GitHub 提交

### 仓库信息

- 远程仓库：`https://github.com/asen-stk-bot/PetBehaviorMonitor`（**私有**）
- 认证方式：SSH（已配置部署密钥，走 443 端口，**无需输入密码**）

### 日常提交（改完代码后）

在 PyCharm 底部 **Terminal** 里依次执行：

```bash
git add -A
git commit -m "说明这次改了什么"
git push
```

> 也可用 PyCharm 图形界面：顶部菜单 **Git → Commit…** 勾选改动 → 提交并推送（Commit and Push）。

### 换电脑 / 重新克隆

```bash
# 1. 生成密钥（本机若已有可跳过）
ssh-keygen -t ed25519 -C "你的邮箱"

# 2. 把公钥（~/.ssh/id_ed25519.pub 内容）加到 GitHub 仓库的 Settings → Deploy keys

# 3. 走 443 端口克隆
git clone ssh://git@ssh.github.com:443/asen-stk-bot/PetBehaviorMonitor.git

# 4. 安装依赖
cd PetBehaviorMonitor && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt
```

> 若无法访问 GitHub：需开启 Watt Toolkit 等加速工具，或将 SSH 连接端口改为 `ssh.github.com:443`（本项目已配好）。

---

## 项目结构

```
PetBehaviorMonitor/
├── main.py                       # 入口（GUI / 报告 / 训练 / 测试）
├── requirements.txt
├── pet_monitor.spec              # PyInstaller 配置
├── run.bat / run.sh              # 启动脚本
├── data/                         # 数据/模型/数据库/快照/报告（运行时生成）
├── docs/                         # 文档
├── scripts/
│   ├── generate_demo_video.py    # 合成演示视频
│   └── train_model.py            # YOLOv8 微调训练
├── src/pet_monitor/
│   ├── config.py                 # 全局配置 + 持久化(save/load)
│   ├── core/
│   │   ├── detector.py           # YOLO 检测 + IoU 跟踪
│   │   ├── behavior.py           # 行为识别
│   │   ├── database.py           # SQLite 封装 + 统计分析 + CSV
│   │   ├── alert.py              # 报警系统
│   │   ├── snapshot.py           # 报警快照（非 ASCII 路径安全）
│   │   ├── report.py             # HTML 报告生成
│   │   └── monitor.py            # 监控引擎（粘合层）
│   └── ui/
│       ├── main_window.py        # PyQt5 主窗口（5 标签页 + 报告菜单）
│       ├── video_widget.py       # 视频显示组件
│       └── charts.py             # 纯 QPainter 图表组件
└── tests/
    ├── test_units.py             # 基础单元测试
    └── test_features.py          # 统计/快照/报告/配置测试
```

---

## 技术栈

| 层次 | 选型 |
| --- | --- |
| 编程语言 | Python 3.9+ |
| 视觉模型 | YOLOv8（ultralytics） |
| 视觉库 | OpenCV 4.x |
| GUI | PyQt5 |
| 数据库 | SQLite（标准库） |
| 打包 | PyInstaller |

---

## 核心功能

| 功能 | 说明 |
| --- | --- |
| 实时检测 | YOLOv8 检测猫/狗（COCO 15/16），IoU 跟踪分配稳定 track_id |
| 行为识别 | 5 类行为（休息/活动/进食/饮水/异常），几何+运动启发式 |
| 报警系统 | 异常/进食/饮水分级报警 + 蜂鸣 + 冷却去重 |
| 报警快照 | 关键报警自动截图存证（`data/snapshots/`） |
| 统计分析 | 按小时活动分布、每日汇总、行为时长占比 |
| 报告导出 | 一键生成自包含 HTML 报告（含图表）+ 事件 CSV |
| 可视化 | GUI 内置柱状图、快照画廊、事件/报警表格 |
| 配置持久化 | 阈值/ROI 改动自动保存，跨会话恢复 |
| 模型训练 | 提供 YOLOv8 微调脚本（`--train`），补齐训练环节 |

---

## 选题依据

任务书 10 个选题均为工业监控场景（皮带偏移、安全帽、烟火、异物、施工车辆、人脸、皮带撕裂、手势、异构视频源、边缘设备监控）。本项目自选 **"宠物行为识别监测系统"**，理由如下：

1. **技术栈契合**：完全采用任务书要求的 Python + OpenCV + YOLO + 数据库 技术栈；
2. **场景贴近生活**：相比工业监控，宠物场景对家庭/学校演示友好；
3. **算法可验证**：可使用公开猫狗数据集与 YOLOv8 预训练权重（猫=15、狗=16）直接验证；
4. **具备实用价值**：可独立发展为商用宠物监护产品。

---

## 成绩考核对照（任务书）

| 考核项 | 分值 | 本项目交付物 |
| --- | --- | --- |
| 平时考勤 / 表现 | 10% / 10% | （由教师评定） |
| 项目完成情况 | 40% | `src/` + `scripts/` + 可运行的 GUI 应用 |
| 报告 | 40% | `docs/` 全部 5 篇 |

---

## License

仅用于课程作业与教学交流。