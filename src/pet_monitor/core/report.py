"""
报告生成 / Report Generation

从数据库聚合数据，产出：
1. 自包含 HTML 报告（内联 CSS + SVG 图表，零外部依赖，双击即可打开）
   - 总览卡片：事件数 / 报警数 / 快照数 / 监测时长
   - 行为时长分布（横向条形图）
   - 分时活动图（24 小时柱状图）
   - 最近报警表格
   - 快照画廊（若有）
2. CSV 导出（由 Database.export_events_csv 提供）

用法：
    python main.py --report [--date YYYY-MM-DD]
"""
from __future__ import annotations

import html
import time
from pathlib import Path
from typing import Any

from ..config import CONFIG, REPORT_DIR
from .database import Database
from .behavior import BEHAVIOR_LABELS_CN


# 行为配色（BGR 场景无关，这里用网页 HEX）
BEHAVIOR_COLORS = {
    "resting":  "#5b8def",
    "active":   "#34c98e",
    "eating":   "#f2a13b",
    "drinking": "#42c6d6",
    "anomaly":  "#e5484d",
    "unknown":  "#9aa3b0",
}


def _fmt_ts(ts: float | None) -> str:
    if not ts:
        return "-"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))


def _fmt_duration(sec: float) -> str:
    sec = float(sec)
    if sec >= 3600:
        return f"{sec / 3600:.2f} 小时"
    if sec >= 60:
        return f"{sec / 60:.1f} 分钟"
    return f"{sec:.1f} 秒"


def _img_src(path: str) -> str:
    """把本地快照路径转为浏览器可识别的 file:// URL。

    Windows 反斜杠转正斜杠，并补 file:/// 前缀；空格转 %20。
    """
    p = str(path).replace("\\", "/")
    p = p.replace(" ", "%20")
    if not p.startswith("/"):
        p = "/" + p
    return "file://" + p


# --------------------------------------------------------------------------- #
# SVG 图表
# --------------------------------------------------------------------------- #
def _bar_chart_horizontal(dist: dict[str, float], width: int = 560, bar_h: int = 34) -> str:
    """横向条形图：行为 -> 时长（秒）。"""
    total = sum(dist.values()) or 1.0
    items = sorted(dist.items(), key=lambda kv: -kv[1])
    gap = 10
    height = bar_h * len(items) + gap * (len(items) - 1) + 20
    parts: list[str] = []
    for i, (k, v) in enumerate(items):
        y = i * (bar_h + gap) + 8
        pct = v / total
        bar_w = max(2, int(width * pct))
        color = BEHAVIOR_COLORS.get(k, "#9aa3b0")
        label = BEHAVIOR_LABELS_CN.get(k, k)
        parts.append(
            f'<text x="0" y="{y + 20}" font-size="13" fill="#333">{label}</text>'
        )
        parts.append(
            f'<rect x="90" y="{y}" width="{bar_w}" height="{bar_h - 6}" rx="5" '
            f'fill="{color}"></rect>'
        )
        parts.append(
            f'<text x="{90 + bar_w + 10}" y="{y + 20}" font-size="12" fill="#666">'
            f'{_fmt_duration(v)} ({pct * 100:.1f}%)</text>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">'
        + "".join(parts)
        + "</svg>"
    )


def _bar_chart_hourly(activity: list[dict[str, Any]]) -> str:
    """24 小时活动柱状图（采样点数）。"""
    width, height, pad = 760, 240, 34
    plot_h = height - pad * 2
    max_n = max((b["total"] for b in activity), default=0) or 1
    colors = BEHAVIOR_COLORS
    order = ["resting", "active", "eating", "drinking", "anomaly"]

    parts: list[str] = []
    bar_w = (width - pad * 2) / 24
    for h in range(24):
        b = activity[h]
        x = pad + h * bar_w
        # 堆叠：按 order 依次绘制各行为占比
        y_cursor = pad + plot_h
        for beh in order:
            cnt = b["behaviors"].get(beh, 0)
            if cnt <= 0:
                continue
            bh = cnt / max_n * plot_h
            y_cursor -= bh
            parts.append(
                f'<rect x="{x + 1}" y="{y_cursor:.1f}" width="{bar_w - 2:.1f}" '
                f'height="{bh:.1f}" fill="{colors.get(beh, "#ccc")}"></rect>'
            )
        # 底部小时刻度
        if h % 3 == 0:
            parts.append(
                f'<text x="{x + bar_w / 2}" y="{height - 6}" font-size="10" '
                f'text-anchor="middle" fill="#888">{h}</text>'
            )
    # 坐标轴
    parts.append(f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad + plot_h}" '
                 f'stroke="#ccc"/>')
    parts.append(f'<line x1="{pad}" y1="{pad + plot_h}" x2="{width - pad}" '
                 f'y2="{pad + plot_h}" stroke="#ccc"/>')

    # 图例
    lx = pad
    for beh in order:
        parts.append(
            f'<rect x="{lx}" y="8" width="12" height="12" fill="{colors.get(beh, "#ccc")}"></rect>'
        )
        parts.append(
            f'<text x="{lx + 16}" y="19" font-size="11" fill="#555">'
            f'{BEHAVIOR_LABELS_CN.get(beh, beh)}</text>'
        )
        lx += 16 + 40 + 12

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" '
        f'xmlns="http://www.w3.org/2000/svg" role="img">'
        + "".join(parts)
        + "</svg>"
    )


# --------------------------------------------------------------------------- #
# HTML 报告
# --------------------------------------------------------------------------- #
def _render_html(overview: dict[str, Any], activity: list[dict[str, Any]],
                 alerts: list[dict[str, Any]], snapshots: list[dict[str, Any]],
                 date_str: str) -> str:
    dist = overview.get("behavior_distribution", {})
    total_sec = overview.get("total_sec", 0.0)

    # 总览卡片
    cards = [
        ("监测事件数", f"{overview.get('total_events', 0):,}"),
        ("报警记录", f"{overview.get('total_alerts', 0):,}"),
        ("报警快照", f"{overview.get('total_snapshots', 0):,}"),
        ("累计监测时长", _fmt_duration(total_sec)),
    ]
    card_html = "".join(
        f'<div class="card"><div class="k">{k}</div><div class="v">{v}</div></div>'
        for k, v in cards
    )

    # 报警表
    alert_rows = "".join(
        f"<tr><td>{_fmt_ts(a['ts'])}</td>"
        f"<td><span class='lv {a['level']}'>{a['level']}</span></td>"
        f"<td>{html.escape(a['message'])}</td></tr>"
        for a in alerts[:50]
    ) or "<tr><td colspan='3' class='empty'>暂无报警记录</td></tr>"

    # 快照画廊
    if snapshots:
        thumbs = "".join(
            f'<figure><img src="{_img_src(s["path"])}" alt="snapshot">'
            f'<figcaption>{html.escape(BEHAVIOR_LABELS_CN.get(s["behavior"], s["behavior"] or "-"))} '
            f'#{s["track_id"]}<br>{_fmt_ts(s["ts"])}</figcaption></figure>'
            for s in snapshots[:12]
        )
    else:
        thumbs = "<p class='empty'>暂无报警快照（触发异常报警时自动截图）</p>"

    chart_dist = _bar_chart_horizontal(dist) if dist else "<p class='empty'>暂无行为数据</p>"
    chart_hour = _bar_chart_hourly(activity) if any(b["total"] for b in activity) else "<p class='empty'>当日无活动记录</p>"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>宠物行为监测报告 · {date_str}</title>
<style>
  :root {{ --bg:#f5f6f8; --card:#fff; --line:#e6e8ec; --txt:#222; --sub:#666; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;
         background:var(--bg); color:var(--txt); padding:24px; }}
  .wrap {{ max-width:960px; margin:0 auto; }}
  header h1 {{ font-size:22px; margin:0 0 4px; }}
  header .sub {{ color:var(--sub); font-size:13px; margin-bottom:20px; }}
  .cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
  .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px;
           padding:16px 18px; }}
  .card .k {{ color:var(--sub); font-size:12px; }}
  .card .v {{ font-size:24px; font-weight:700; margin-top:6px; }}
  section {{ background:var(--card); border:1px solid var(--line); border-radius:12px;
             padding:20px; margin-top:20px; }}
  h2 {{ font-size:16px; margin:0 0 16px; border-left:4px solid #5b8def; padding-left:10px; }}
  table {{ width:100%; border-collapse:collapse; font-size:13px; }}
  th, td {{ text-align:left; padding:8px 10px; border-bottom:1px solid var(--line); }}
  th {{ color:var(--sub); font-weight:600; }}
  .lv {{ padding:2px 8px; border-radius:10px; font-size:12px; color:#fff; }}
  .lv.info {{ background:#34c98e; }}
  .lv.warning {{ background:#f2a13b; }}
  .lv.critical {{ background:#e5484d; }}
  .empty {{ color:#999; text-align:center; padding:20px; }}
  .gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:12px; }}
  figure {{ margin:0; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:#000; }}
  figure img {{ width:100%; height:110px; object-fit:cover; display:block; }}
  figcaption {{ font-size:11px; color:var(--sub); padding:6px 8px; background:var(--card); }}
  footer {{ margin-top:24px; color:#999; font-size:12px; text-align:center; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>宠物行为识别监测系统 · 行为分析报告</h1>
    <div class="sub">南京工程学院 · 软件工程232 · 软件工程项目训练 &nbsp;|&nbsp; 统计日期：{date_str} &nbsp;|&nbsp; 生成时间：{_fmt_ts(time.time())}</div>
  </header>

  <div class="cards">{card_html}</div>

  <section>
    <h2>行为时长分布</h2>
    {chart_dist}
  </section>

  <section>
    <h2>分时活动（24 小时）</h2>
    {chart_hour}
  </section>

  <section>
    <h2>最近报警</h2>
    <table>
      <tr><th>时间</th><th>级别</th><th>消息</th></tr>
      {alert_rows}
    </table>
  </section>

  <section>
    <h2>报警快照</h2>
    <div class="gallery">{thumbs}</div>
  </section>

  <footer>由宠物行为识别监测系统自动生成</footer>
</div>
</body>
</html>
"""


def generate_report(
    db: Database | None = None,
    out_dir: Path | None = None,
    date_str: str | None = None,
) -> Path:
    """生成 HTML 报告，返回文件路径。"""
    import datetime

    db = db or Database()
    date_str = date_str or datetime.date.today().strftime("%Y-%m-%d")

    overview = db.stats_overview()
    activity = db.hourly_activity(date_str)
    alerts = db.list_alerts(limit=100)
    snapshots = db.list_snapshots(limit=60)

    out_dir = Path(out_dir) if out_dir else REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"report_{date_str}.html"
    out.write_text(
        _render_html(overview, activity, alerts, snapshots, date_str),
        encoding="utf-8",
    )
    return out


def export_csv(db: Database | None = None, out_dir: Path | None = None) -> Path:
    """导出事件 CSV，返回文件路径。"""
    import datetime

    db = db or Database()
    out_dir = Path(out_dir) if out_dir else REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"events_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    db.export_events_csv(out)
    return out


# --------------------------------------------------------------------------- #
# PDF 报告（基于 reportlab）
# --------------------------------------------------------------------------- #
def export_pdf(db: Database | None = None, out_dir: Path | None = None,
               date_str: str | None = None) -> Path:
    """导出 PDF 报告（封面+总览+行为分布+分时活动+报警+快照），返回文件路径。

    字体：使用 reportlab 内置的中文 CID 字体 STSong-Light，免安装系统字体。
    """
    import datetime
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak,
    )

    db = db or Database()
    date_str = date_str or datetime.date.today().strftime("%Y-%m-%d")
    out_dir = Path(out_dir) if out_dir else REPORT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"report_{date_str}.pdf"

    # 注册中文字体（reportlab 内置 CID，无需字体文件）
    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
    zh = "STSong-Light"

    overview = db.stats_overview()
    activity = db.hourly_activity(date_str)
    alerts = db.list_alerts(limit=50)
    snapshots = db.list_snapshots(limit=6)

    doc = SimpleDocTemplate(
        str(out), pagesize=A4,
        leftMargin=2 * cm, rightMargin=2 * cm,
        topMargin=2 * cm, bottomMargin=2 * cm,
        title="宠物行为监测报告",
    )

    # 样式
    base = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=base["Heading1"], fontName=zh, fontSize=20, spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=base["Heading2"], fontName=zh, fontSize=14, spaceAfter=6, textColor=colors.HexColor("#5b8def"))
    body = ParagraphStyle("body", parent=base["Normal"], fontName=zh, fontSize=10, leading=14)
    small = ParagraphStyle("small", parent=base["Normal"], fontName=zh, fontSize=8, textColor=colors.grey)
    big = ParagraphStyle("big", parent=base["Normal"], fontName=zh, fontSize=22, alignment=1, textColor=colors.HexColor("#222"))

    story = []

    # ----- 封面 -----
    story.append(Spacer(1, 4 * cm))
    story.append(Paragraph("宠物行为识别监测系统", h1))
    story.append(Paragraph("行为分析报告", h1))
    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph(f"统计日期：<b>{date_str}</b>", body))
    story.append(Paragraph(f"生成时间：{_fmt_ts(time.time())}", body))
    story.append(Paragraph("南京工程学院 · 软件工程232 · 软件工程项目训练", body))
    story.append(Paragraph("作者：程祥凯", body))
    story.append(PageBreak())

    # ----- 总览卡片 -----
    story.append(Paragraph("一、总览", h2))
    cards = [
        ["监测事件数", f"{overview.get('total_events', 0):,}"],
        ["报警记录", f"{overview.get('total_alerts', 0):,}"],
        ["报警快照", f"{overview.get('total_snapshots', 0):,}"],
        ["累计监测时长", _fmt_duration(overview.get("total_sec", 0.0))],
    ]
    card_tbl = Table(
        [[Paragraph(f"<b>{k}</b><br/><font size=18>{v}</font>", body) for k, v in cards]],
        colWidths=[3.6 * cm] * 4,
    )
    card_tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#e6e8ec")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e6e8ec")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafbfc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    story.append(card_tbl)
    story.append(Spacer(1, 0.6 * cm))

    # ----- 行为分布 -----
    dist = overview.get("behavior_distribution", {})
    story.append(Paragraph("二、行为时长分布", h2))
    if dist:
        total = sum(dist.values()) or 1.0
        rows = [["行为", "时长", "占比"]]
        for k, v in sorted(dist.items(), key=lambda kv: -kv[1]):
            rows.append([
                BEHAVIOR_LABELS_CN.get(k, k),
                _fmt_duration(v),
                f"{v / total * 100:.1f}%",
            ])
        tbl = Table(rows, colWidths=[4 * cm, 4 * cm, 3 * cm])
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), zh),
            ("FONTSIZE", (0, 0), (-1, -1), 10),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5b8def")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ddd")),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("暂无行为数据", body))
    story.append(Spacer(1, 0.6 * cm))

    # ----- 分时活动 -----
    story.append(Paragraph("三、分时活动（24 小时）", h2))
    if any(b["total"] for b in activity):
        # 用文本表格 + 简单条形字符
        rows = [["小时", "事件数"]]
        max_n = max((b["total"] for b in activity), default=1) or 1
        for h, b in enumerate(activity):
            n = b["total"]
            bar = "█" * int(20 * n / max_n) if max_n else ""
            rows.append([f"{h:02d}:00", f"{n:>4}  {bar}"])
        tbl = Table(rows, colWidths=[2 * cm, 12 * cm])
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), zh),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("FONTNAME", (1, 1), (1, -1), "Courier"),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5b8def")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#eee")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("当日无活动记录", body))
    story.append(Spacer(1, 0.6 * cm))

    # ----- 最近报警 -----
    story.append(Paragraph("四、最近报警（最多 50 条）", h2))
    if alerts:
        rows = [["时间", "级别", "消息"]]
        for a in alerts[:50]:
            rows.append([_fmt_ts(a["ts"]), a["level"], a["message"]])
        tbl = Table(rows, colWidths=[4 * cm, 2.2 * cm, 9 * cm])
        tbl.setStyle(TableStyle([
            ("FONTNAME", (0, 0), (-1, -1), zh),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#5b8def")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.2, colors.HexColor("#eee")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(tbl)
    else:
        story.append(Paragraph("暂无报警记录", body))
    story.append(Spacer(1, 0.6 * cm))

    # ----- 快照（嵌入图片） -----
    story.append(Paragraph("五、报警快照", h2))
    if snapshots:
        img_rows = [[]]
        for s in snapshots[:6]:
            p = Path(s["path"])
            if p.exists() and p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                try:
                    img_rows[0].append(
                        Image(str(p), width=4.8 * cm, height=3.2 * cm)
                    )
                except Exception:
                    pass
        if img_rows[0]:
            tbl = Table(img_rows, colWidths=[5 * cm] * len(img_rows[0]))
            tbl.setStyle(TableStyle([
                ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#ddd")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(tbl)
        else:
            story.append(Paragraph("快照文件无法读取", body))
    else:
        story.append(Paragraph("暂无报警快照（触发异常报警时自动截图）", body))

    story.append(Spacer(1, 1 * cm))
    story.append(Paragraph("由宠物行为识别监测系统自动生成", small))

    doc.build(story)
    return out


if __name__ == "__main__":
    db = Database()
    p = generate_report(db)
    print("报告已生成:", p)
