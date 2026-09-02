"""
主窗口

布局（左侧视频 + 右侧面板，使用 QSplitter 弹性）：
    ┌──────────────────────────────────────────┬─────────────────────────┐
    │                                          │ ▶ 控制 / 阈值            │
    │            视频显示区域                  │ ─────                    │
    │            VideoWidget                   │ 当前状态 / FPS           │
    │                                          │ ─────                    │
    │                                          │ 最近事件（表格）         │
    │                                          │ ─────                    │
    │                                          │ 报警记录（表格）         │
    └──────────────────────────────────────────┴─────────────────────────┘
    [进度条 / 状态栏]

启动入口：
    from pet_monitor.ui.main_window import MainWindow
    MainWindow().show()
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QPixmap, QFont
from PyQt5.QtWidgets import (
    QAction, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
    QMainWindow, QMessageBox, QPushButton, QSizePolicy, QSpinBox,
    QSplitter, QStatusBar, QStyle, QSystemTrayIcon, QTabWidget, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget, QScrollArea,
)

from ..config import CONFIG, SNAPSHOT_DIR, load_config, save_config
from ..core.alert import AlertRecord, get_alert
from ..core.database import get_db
from ..core.monitor import FrameResult, Monitor
from ..core.report import export_csv, generate_report
from .charts import BarChart
from .video_widget import VideoWidget


log = logging.getLogger("pet_monitor.ui")


# --------------------------------------------------------------------------- #
# 状态更新载荷
# --------------------------------------------------------------------------- #
class _StateBridge(QWidget):
    """桥接后台线程的 FrameResult 到 UI 主线程。

    Monitor 线程通过 QTimer 触发 on_timer 拉取最新结果，
    UI 不直接被后台线程调用，保证线程安全。
    """

    frame_ready = pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._latest: FrameResult | None = None
        self._lock = False

    def submit(self, r: FrameResult) -> None:
        self._latest = r
        self.frame_ready.emit(r)


# --------------------------------------------------------------------------- #
# 主窗口
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(CONFIG.ui.window_title)
        self.resize(*CONFIG.ui.window_size)

        self._monitor: Monitor | None = None
        self._alert = get_alert()
        self._db = get_db()
        self._alert.set_popup_callback(self._on_alert_popup)

        # 系统托盘 + 桌面通知
        self._init_tray()

        self._bridge = _StateBridge()
        self._bridge.frame_ready.connect(self._on_frame)
        self.video.roi_drawn.connect(self._on_roi_drawn)

        self._build_ui()
        self._load_config()
        self._refresh_history()
        self._refresh_snapshots()

        self._tick = QTimer(self)
        self._tick.setInterval(2000)
        self._tick.timeout.connect(self._on_tick)
        self._tick.start()
        self._tick_count = 0

    def _on_tick(self) -> None:
        self._refresh_history()
        self._tick_count += 1
        # 快照画廊降低刷新频率，避免缩略图反复重建导致闪烁
        if self._tick_count % 3 == 0:
            self._refresh_snapshots()

    # ------------------------------------------------------------------ #
    # 界面构建
    # ------------------------------------------------------------------ #
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        # 左：视频
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.video = VideoWidget()
        ll.addWidget(self.video, 1)
        self.lbl_fps = QLabel("FPS: -")
        self.lbl_fps.setStyleSheet("color:#888; font-size:11px;")
        ll.addWidget(self.lbl_fps)
        splitter.addWidget(left)

        # 右：控制 / 状态 / 表格
        right = QTabWidget()
        splitter.addWidget(right)
        right.addTab(self._build_control_panel(), "控制")
        right.addTab(self._build_event_panel(), "事件")
        right.addTab(self._build_alert_panel(), "报警")
        right.addTab(self._build_stat_panel(), "统计")
        right.addTab(self._build_snapshot_panel(), "快照")

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("就绪")

        # 菜单
        m_file = self.menuBar().addMenu("文件(&F)")
        a_open = QAction("打开视频…", self)
        a_open.triggered.connect(self._action_open_video)
        m_file.addAction(a_open)
        m_file.addSeparator()
        a_exit = QAction("退出", self)
        a_exit.triggered.connect(self.close)
        m_file.addAction(a_exit)

        m_report = self.menuBar().addMenu("报告(&R)")
        a_html = QAction("生成 HTML 报告…", self)
        a_html.triggered.connect(self._export_report)
        m_report.addAction(a_html)
        a_csv = QAction("导出事件 CSV…", self)
        a_csv.triggered.connect(self._export_csv)
        m_report.addAction(a_csv)

        m_help = self.menuBar().addMenu("帮助(&H)")
        a_about = QAction("关于", self)
        a_about.triggered.connect(self._about)
        m_help.addAction(a_about)

    def _build_control_panel(self) -> QWidget:
        w = QWidget()
        g = QVBoxLayout(w)
        g.setContentsMargins(8, 8, 8, 8)

        gb_src = QGroupBox("视频源")
        gl = QGridLayout(gb_src)
        self.cmb_source = QComboBox()
        # 内置素材 + 摄像头/文件多种源，运行中可切换
        from ..core.video_source import list_builtin_sources, BUILTIN_SOURCE_LABELS
        builtin = list_builtin_sources()
        for key in builtin:
            self.cmb_source.addItem(BUILTIN_SOURCE_LABELS.get(key, key), userData=("builtin", key))
        self.cmb_source.addItem("默认摄像头 (0)", userData=("camera", 0))
        self.cmb_source.addItem("摄像头 1", userData=("camera", 1))
        self.cmb_source.addItem("摄像头 2", userData=("camera", 2))
        self.cmb_source.currentIndexChanged.connect(self._on_source_changed)
        self.ed_source_path = QLineEdit()
        self.ed_source_path.setPlaceholderText("或填写视频文件路径…")
        self.btn_browse = QPushButton("浏览…")
        self.btn_browse.clicked.connect(self._browse_video)
        gl.addWidget(QLabel("源类型"), 0, 0)
        gl.addWidget(self.cmb_source, 0, 1)
        gl.addWidget(QLabel("文件路径"), 1, 0)
        gl.addWidget(self.ed_source_path, 1, 1)
        gl.addWidget(self.btn_browse, 1, 2)
        g.addWidget(gb_src)

        gb_act = QGroupBox("操作")
        hl = QHBoxLayout(gb_act)
        self.btn_start = QPushButton("开始监控")
        self.btn_start.clicked.connect(self._start_monitor)
        self.btn_stop = QPushButton("停止监控")
        self.btn_stop.clicked.connect(self._stop_monitor)
        self.btn_stop.setEnabled(False)
        hl.addWidget(self.btn_start)
        hl.addWidget(self.btn_stop)
        g.addWidget(gb_act)

        gb_th = QGroupBox("行为识别阈值（实时生效）")
        tg = QGridLayout(gb_th)
        tg.addWidget(QLabel("检测置信度"), 0, 0)
        self.sp_conf = QDoubleSpinBox(); self.sp_conf.setRange(0.05, 0.95); self.sp_conf.setSingleStep(0.05)
        self.sp_conf.setValue(CONFIG.detector.conf_threshold)
        self.sp_conf.valueChanged.connect(lambda v: setattr(CONFIG.detector, "conf_threshold", v))
        tg.addWidget(self.sp_conf, 0, 1)

        tg.addWidget(QLabel("静止阈值(像素)"), 1, 0)
        self.sp_rest = QDoubleSpinBox(); self.sp_rest.setRange(0.5, 50.0); self.sp_rest.setSingleStep(0.5)
        self.sp_rest.setValue(CONFIG.behavior.resting_motion_px)
        self.sp_rest.valueChanged.connect(lambda v: setattr(CONFIG.behavior, "resting_motion_px", v))
        tg.addWidget(self.sp_rest, 1, 1)

        tg.addWidget(QLabel("活跃阈值(像素/帧)"), 2, 0)
        self.sp_act = QDoubleSpinBox(); self.sp_act.setRange(1.0, 100.0); self.sp_act.setSingleStep(1.0)
        self.sp_act.setValue(CONFIG.behavior.active_motion_px_per_frame)
        self.sp_act.valueChanged.connect(lambda v: setattr(CONFIG.behavior, "active_motion_px_per_frame", v))
        tg.addWidget(self.sp_act, 2, 1)

        tg.addWidget(QLabel("异常判定时长(秒)"), 3, 0)
        self.sp_ano = QSpinBox(); self.sp_ano.setRange(10, 600); self.sp_ano.setSingleStep(10)
        self.sp_ano.setValue(int(CONFIG.behavior.anomaly_min_duration_sec))
        self.sp_ano.valueChanged.connect(lambda v: setattr(CONFIG.behavior, "anomaly_min_duration_sec", float(v)))
        tg.addWidget(self.sp_ano, 3, 1)
        g.addWidget(gb_th)

        gb_roi = QGroupBox("进食/饮水 ROI（归一化 0-1，可直接编辑或下面点按钮在视频上拖拽）")
        rg = QGridLayout(gb_roi)
        rg.addWidget(QLabel("进食区:"), 0, 0)
        self.ed_food = QLineEdit(",".join(f"{x:.2f}" for x in CONFIG.behavior.food_roi))
        rg.addWidget(self.ed_food, 0, 1)
        rg.addWidget(QLabel("饮水区:"), 1, 0)
        self.ed_water = QLineEdit(",".join(f"{x:.2f}" for x in CONFIG.behavior.water_roi))
        rg.addWidget(self.ed_water, 1, 1)
        btn_apply_roi = QPushButton("应用 ROI（手动）")
        btn_apply_roi.clicked.connect(self._apply_roi)
        rg.addWidget(btn_apply_roi, 2, 0, 1, 2)

        # 视频上交互绘制按钮
        hl_draw = QHBoxLayout()
        self.btn_draw_food = QPushButton("✏  画食盆 (拖拽)")
        self.btn_draw_food.setCheckable(True)
        self.btn_draw_food.toggled.connect(lambda c: self._set_edit_roi("food" if c else "none"))
        self.btn_draw_water = QPushButton("✏  画水盆 (拖拽)")
        self.btn_draw_water.setCheckable(True)
        self.btn_draw_water.toggled.connect(lambda c: self._set_edit_roi("water" if c else "none"))
        hl_draw.addWidget(self.btn_draw_food)
        hl_draw.addWidget(self.btn_draw_water)
        hl_draw.addStretch(1)
        rg.addLayout(hl_draw, 3, 0, 1, 2)
        g.addWidget(gb_roi)

        g.addStretch(1)
        return w

    def _build_event_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.tbl_events = QTableWidget(0, 6)
        self.tbl_events.setHorizontalHeaderLabels(["时间", "ID", "物种", "行为", "置信度", "来源"])
        self.tbl_events.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_events.verticalHeader().setVisible(False)
        v.addWidget(self.tbl_events)
        h = QHBoxLayout()
        btn = QPushButton("刷新")
        btn.clicked.connect(self._refresh_history)
        h.addStretch(1)
        h.addWidget(btn)
        v.addLayout(h)
        return w

    def _build_alert_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.tbl_alerts = QTableWidget(0, 4)
        self.tbl_alerts.setHorizontalHeaderLabels(["时间", "级别", "消息", "已读"])
        self.tbl_alerts.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_alerts.verticalHeader().setVisible(False)
        v.addWidget(self.tbl_alerts)
        h = QHBoxLayout()
        btn_ref = QPushButton("刷新")
        btn_ref.clicked.connect(self._refresh_alerts)
        btn_clear = QPushButton("全部标记已读")
        btn_clear.clicked.connect(self._ack_all_alerts)
        h.addStretch(1); h.addWidget(btn_ref); h.addWidget(btn_clear)
        v.addLayout(h)
        return w

    def _build_stat_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        gb = QGroupBox("行为时长分布")
        gl = QVBoxLayout(gb)
        self.lbl_stats = QLabel("暂无数据")
        self.lbl_stats.setStyleSheet("font-family: Consolas; font-size: 12px; color:#555;")
        gl.addWidget(self.lbl_stats)
        self.chart = BarChart()
        gl.addWidget(self.chart)
        v.addWidget(gb)
        h = QHBoxLayout()
        btn = QPushButton("刷新统计")
        btn.clicked.connect(self._refresh_stats)
        btn_daily = QPushButton("生成每日汇总")
        btn_daily.clicked.connect(self._upsert_daily)
        h.addStretch(1); h.addWidget(btn_daily); h.addWidget(btn)
        v.addLayout(h)
        return w

    def _build_snapshot_panel(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.lbl_snap_count = QLabel("暂无快照")
        self.lbl_snap_count.setStyleSheet("color:#888; font-size:12px;")
        v.addWidget(self.lbl_snap_count)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.snap_grid = QGridLayout(container)
        self.snap_grid.setSpacing(8)
        scroll.setWidget(container)
        v.addWidget(scroll, 1)

        h = QHBoxLayout()
        btn_ref = QPushButton("刷新")
        btn_ref.clicked.connect(self._refresh_snapshots)
        btn_open = QPushButton("打开快照目录")
        btn_open.clicked.connect(self._open_snapshot_dir)
        h.addStretch(1); h.addWidget(btn_ref); h.addWidget(btn_open)
        v.addLayout(h)
        return w

    # ------------------------------------------------------------------ #
    # 操作
    # ------------------------------------------------------------------ #
    def _browse_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "", "Video (*.mp4 *.avi *.mov *.mkv);;All (*.*)",
        )
        if path:
            self.ed_source_path.setText(path)
            # 自动选回 "默认摄像头" 占位以提示当前以文件路径生效
            self.cmb_source.setCurrentIndex(0)

    def _on_source_changed(self, _idx: int) -> None:
        """切换视频源时若监控运行中，自动重启。"""
        if self._monitor and self._monitor.is_running():
            self.statusBar().showMessage("切换视频源中…", 2000)
            self._stop_monitor()
            self._start_monitor()

    def _resolve_source(self) -> int | str:
        path = self.ed_source_path.text().strip()
        if path:
            return path
        data = self.cmb_source.currentData()
        if data is None:
            return 0
        kind, key = data
        if kind == "builtin":
            from ..core.video_source import resolve_builtin_path
            return resolve_builtin_path(key)
        # camera
        return int(key)

    def _start_monitor(self) -> None:
        if self._monitor and self._monitor.is_running():
            return
        try:
            src = self._resolve_source()
            self._monitor = Monitor(source=src, on_result=self._bridge.submit)
            self._monitor.start()
            self.btn_start.setEnabled(False)
            self.btn_stop.setEnabled(True)
            self.statusBar().showMessage(f"监控中 · 源: {src}")
        except Exception as e:
            QMessageBox.critical(self, "启动失败", str(e))

    def _stop_monitor(self) -> None:
        if self._monitor:
            self._monitor.stop()
            self._monitor = None
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.statusBar().showMessage("已停止")

    def _apply_roi(self) -> None:
        try:
            f = tuple(float(x.strip()) for x in self.ed_food.text().split(","))
            w = tuple(float(x.strip()) for x in self.ed_water.text().split(","))
            assert len(f) == 4 and len(w) == 4
            CONFIG.behavior.food_roi = f
            CONFIG.behavior.water_roi = w
            self.video.set_roi(f, w)
            self.statusBar().showMessage("ROI 已更新", 3000)
        except Exception as e:
            QMessageBox.warning(self, "ROI 格式错误", str(e))

    def _set_edit_roi(self, kind: str) -> None:
        """开启/关闭视频上拖拽画 ROI 模式。"""
        # 互斥：开 food 时关 water，反之亦然
        if kind == "food":
            self.btn_draw_water.setChecked(False)
        elif kind == "water":
            self.btn_draw_food.setChecked(False)
        else:
            self.btn_draw_food.setChecked(False)
            self.btn_draw_water.setChecked(False)
        self.video.set_edit_kind(kind)
        if kind != "none":
            self.video.setFocus()
            self.statusBar().showMessage(f"在视频画面上拖拽绘制{kind} ROI（Esc 取消）", 5000)
        else:
            self.statusBar().showMessage("已退出 ROI 编辑", 2000)

    def _on_roi_drawn(self, kind: str, norm: tuple) -> None:
        """用户在视频上拖拽画完 ROI 后的回调。"""
        if kind == "food":
            CONFIG.behavior.food_roi = norm
            self.ed_food.setText(",".join(f"{x:.2f}" for x in norm))
        elif kind == "water":
            CONFIG.behavior.water_roi = norm
            self.ed_water.setText(",".join(f"{x:.2f}" for x in norm))
        self.statusBar().showMessage(f"{kind} ROI 已更新: {norm}", 4000)
        # 退出编辑态
        self._set_edit_roi("none")

    def _action_open_video(self) -> None:
        self._browse_video()

    def _about(self) -> None:
        QMessageBox.about(
            self, "关于",
            "<h3>宠物行为识别监测系统 v1.0</h3>"
            "<p>南京工程学院 · 软件工程232 · 软件工程项目训练</p>"
            "<p>指导教师：张冰</p>"
            "<p>技术栈：Python · YOLOv8 · OpenCV · PyQt5 · SQLite</p>",
        )

    # ------------------------------------------------------------------ #
    # 数据回调
    # ------------------------------------------------------------------ #
    def _on_frame(self, r: FrameResult) -> None:
        self.video.show_frame(r.frame)
        self.lbl_fps.setText(
            f"FPS: {r.fps:.1f} · 检测: {len(r.detections)} · 行为: "
            f"{','.join({b.behavior for b in r.behaviors}) or '-'}"
        )

    def _on_alert_popup(self, rec: AlertRecord) -> None:
        # 状态栏轻提示
        self.statusBar().showMessage(f"[{rec.level}] {rec.message}", 5000)
        # 系统托盘 + 桌面通知弹窗（即使主窗口最小化也能看到）
        if getattr(self, "_tray", None) is not None and QSystemTrayIcon.isSystemTrayAvailable():
            icon_kind = (
                QSystemTrayIcon.Warning if rec.level in ("warning", "anomaly")
                else QSystemTrayIcon.Information
            )
            self._tray.showMessage(
                f"宠物监控 · {rec.level}",
                rec.message,
                icon_kind,
                4000,
            )

    def _init_tray(self) -> None:
        """初始化系统托盘图标 + 双击恢复主窗口。"""
        if not QSystemTrayIcon.isSystemTrayAvailable():
            self._tray = None
            return
        # 没资源文件就用系统默认应用图标
        style_ico = self.style().standardIcon(QStyle.SP_ComputerIcon)
        self._tray = QSystemTrayIcon(style_ico, self)
        self._tray.setToolTip("宠物行为识别监测系统")
        from PyQt5.QtWidgets import QMenu
        menu = QMenu()
        act_show = menu.addAction("显示主窗口")
        act_show.triggered.connect(self._restore_from_tray)
        act_quit = menu.addAction("退出")
        act_quit.triggered.connect(self.close)
        self._tray.setContextMenu(menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _on_tray_activated(self, reason) -> None:
        if reason == QSystemTrayIcon.Trigger:  # 单击
            self._restore_from_tray()

    def _restore_from_tray(self) -> None:
        self.showNormal()
        self.raise_()
        self.activateWindow()

    def _refresh_history(self) -> None:
        try:
            rows = self._db.recent_events(100)
        except Exception:
            rows = []
        self.tbl_events.setRowCount(len(rows))
        for i, row in enumerate(rows[:200]):
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["ts"]))
            vals = [
                ts, str(row["track_id"]), row["species"], row["behavior"],
                f"{row['confidence']:.2f}", row["source"],
            ]
            for j, v in enumerate(vals):
                self.tbl_events.setItem(i, j, QTableWidgetItem(v))
        self._refresh_stats()

    def _refresh_alerts(self) -> None:
        try:
            rows = self._db.list_alerts(200)
        except Exception:
            rows = []
        self.tbl_alerts.setRowCount(len(rows))
        for i, row in enumerate(rows):
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["ts"]))
            self.tbl_alerts.setItem(i, 0, QTableWidgetItem(ts))
            self.tbl_alerts.setItem(i, 1, QTableWidgetItem(row["level"]))
            self.tbl_alerts.setItem(i, 2, QTableWidgetItem(row["message"]))
            self.tbl_alerts.setItem(i, 3, QTableWidgetItem("是" if row["acknowledged"] else "否"))

    def _ack_all_alerts(self) -> None:
        try:
            for row in self._db.list_alerts(500):
                if not row["acknowledged"]:
                    self._db.acknowledge_alert(row["id"])
        finally:
            self._refresh_alerts()

    def _refresh_stats(self) -> None:
        try:
            dist = self._db.behavior_distribution()
        except Exception:
            dist = {}
        from ..core.behavior import BEHAVIOR_LABELS_CN
        lines = [f"行为时长分布（共 {sum(dist.values()):.1f} 秒）"]
        for k, v in sorted(dist.items(), key=lambda x: -x[1]):
            cn = BEHAVIOR_LABELS_CN.get(k, k)
            lines.append(f"  {cn:<6} : {v:>8.1f} s")
        self.lbl_stats.setText("\n".join(lines))
        if hasattr(self, "chart"):
            self.chart.set_data(dist)

    def _upsert_daily(self) -> None:
        import datetime
        date_str = datetime.date.today().strftime("%Y-%m-%d")
        try:
            self._db.upsert_daily_stats(date_str)
            self.statusBar().showMessage(f"已生成 {date_str} 每日汇总", 3000)
        except Exception as e:
            QMessageBox.warning(self, "汇总失败", str(e))

    # ------------------------------------------------------------------ #
    # 快照画廊
    # ------------------------------------------------------------------ #
    def _refresh_snapshots(self) -> None:
        try:
            rows = self._db.list_snapshots(60)
        except Exception:
            rows = []
        # 清空旧缩略图
        while self.snap_grid.count():
            item = self.snap_grid.takeAt(0)
            wdg = item.widget()
            if wdg:
                wdg.deleteLater()

        self.lbl_snap_count.setText(f"共 {len(rows)} 张快照")
        cols = 3
        for i, s in enumerate(rows):
            lbl = QLabel()
            lbl.setFixedSize(150, 110)
            lbl.setScaledContents(True)
            lbl.setStyleSheet("background:#111; border:1px solid #444;")
            pix = QPixmap(s["path"])
            if not pix.isNull():
                lbl.setPixmap(pix)
                lbl.setToolTip(
                    f"{s['behavior']} #{s['track_id']} · "
                    f"{time.strftime('%m-%d %H:%M:%S', time.localtime(s['ts']))}"
                )
            else:
                lbl.setText("(已删除)")
            self.snap_grid.addWidget(lbl, i // cols, i % cols)

    def _open_snapshot_dir(self) -> None:
        import os
        import subprocess
        d = str(SNAPSHOT_DIR)
        try:
            if os.name == "nt":
                os.startfile(d)  # type: ignore[attr-defined]
            else:
                subprocess.Popen(["xdg-open", d])
        except Exception as e:
            QMessageBox.information(self, "快照目录", f"请手动打开：\n{d}")

    # ------------------------------------------------------------------ #
    # 报告导出
    # ------------------------------------------------------------------ #
    def _export_report(self) -> None:
        try:
            path = generate_report(self._db)
            self.statusBar().showMessage(f"报告已生成: {path}", 5000)
            QMessageBox.information(self, "报告已生成", f"HTML 报告已保存到：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _export_csv(self) -> None:
        try:
            path = export_csv(self._db)
            self.statusBar().showMessage(f"CSV 已导出: {path}", 5000)
            QMessageBox.information(self, "CSV 已导出", f"事件数据已保存到：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    # ------------------------------------------------------------------ #
    # 配置持久化
    # ------------------------------------------------------------------ #
    def _load_config(self) -> None:
        if load_config():
            # 把加载到的阈值回填到控件
            self.sp_conf.setValue(CONFIG.detector.conf_threshold)
            self.sp_rest.setValue(CONFIG.behavior.resting_motion_px)
            self.sp_act.setValue(CONFIG.behavior.active_motion_px_per_frame)
            self.sp_ano.setValue(int(CONFIG.behavior.anomaly_min_duration_sec))
            self.ed_food.setText(",".join(f"{x:.2f}" for x in CONFIG.behavior.food_roi))
            self.ed_water.setText(",".join(f"{x:.2f}" for x in CONFIG.behavior.water_roi))
            self.statusBar().showMessage("已加载保存的配置", 2000)

    def _save_config(self) -> None:
        try:
            save_config()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # 关闭
    # ------------------------------------------------------------------ #
    def closeEvent(self, e) -> None:
        self._save_config()
        self._stop_monitor()
        super().closeEvent(e)


def run_app() -> int:
    """供 main.py 调用的便捷入口。"""
    import sys
    from PyQt5.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(run_app())