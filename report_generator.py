# -*- coding: utf-8 -*-
"""
PDF 分析报告生成模块（ReportLab）
==================================

把监测概况、异常识别、预测评估与预警结论汇总为一页式 PDF 报告。
中文字体使用 ReportLab 内置的 STSong-Light（CID 字体），
无需任何外部字体文件，跨平台可用。
"""

import io
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import (Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

import config
import warning as warning_mod

# 注册内置中文字体（无需字体文件）
pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))
FONT = "STSong-Light"

# 深海蓝配色
DARK_BLUE = colors.HexColor("#0D47A1")
LIGHT_BLUE = colors.HexColor("#E3F2FD")


def _styles() -> dict:
    return {
        "title": ParagraphStyle("title", fontName=FONT, fontSize=18,
                                textColor=DARK_BLUE, spaceAfter=4),
        "sub": ParagraphStyle("sub", fontName=FONT, fontSize=9,
                              textColor=colors.grey, spaceAfter=10),
        "h2": ParagraphStyle("h2", fontName=FONT, fontSize=13,
                             textColor=DARK_BLUE, spaceBefore=10, spaceAfter=4),
        "body": ParagraphStyle("body", fontName=FONT, fontSize=10,
                               leading=15),
        "small": ParagraphStyle("small", fontName=FONT, fontSize=8,
                                textColor=colors.grey, leading=11),
    }


def _table(df: pd.DataFrame, col_widths=None) -> Table:
    """DataFrame 转 PDF 表格（深色表头 + 隔行浅色）。"""
    data = [list(df.columns)] + [[str(v) for v in row] for row in df.values]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BLUE]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#90CAF9")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def generate_report(context: dict) -> bytes:
    """生成 PDF 报告，返回字节流。

    context 需要的键：
      report_date, day_stats(当日格点统计), risk(RiskAssessment),
      metrics_table(预测评估 DataFrame), forecast(未来预测 DataFrame),
      events(事件表 DataFrame), region_name
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=14 * mm)
    s = _styles()
    story = []

    # ---- 标题区 -------------------------------------------------------------
    story.append(Paragraph("渤海智瞳 · 海温异常监测及智能预警分析报告", s["title"]))
    story.append(Paragraph(
        f"报告日期：{context['report_date']} ｜ 数据源：{config.DATA_SOURCE_NAME} ｜ "
        f"研究区域：{config.REGION_NAME}（{config.LON_MIN:.0f}°E–{config.LON_MAX:.0f}°E, "
        f"{config.LAT_MIN:.0f}°N–{config.LAT_MAX:.0f}°N）", s["sub"]))

    # ---- 预警结论 -----------------------------------------------------------
    risk = context["risk"]
    story.append(Paragraph("一、智能预警结论", s["h2"]))
    story.append(Paragraph(
        f"<b>风险等级：{risk.level[0]}（{risk.level[1]}）</b>　"
        f"异常类型：{risk.risk_type}　风险位置：{risk.position}　"
        f"异常程度：{risk.degree}", s["body"]))
    for r in risk.reasons:
        story.append(Paragraph(f"· 判定依据：{r}", s["body"]))
    for g in risk.suggestions:
        story.append(Paragraph(f"· 应对建议：{g}", s["body"]))
    story.append(Paragraph(f"注意：{warning_mod.DISCLAIMER}", s["small"]))

    # ---- 当日监测概况 -------------------------------------------------------
    ds = context["day_stats"]
    story.append(Paragraph("二、当日监测概况", s["h2"]))
    overview = pd.DataFrame([{
        "有效海洋格点": ds["n_valid"],
        "显著偏暖格点": ds["n_warm"], "显著偏冷格点": ds["n_cold"],
        "平均距平(℃)": f"{ds['mean_anom']:+.2f}",
        "最大正距平(℃)": f"{ds['max_anom']:+.2f}",
        "最大负距平(℃)": f"{ds['min_anom']:+.2f}",
    }])
    story.append(_table(overview))

    # ---- 预测评估 -----------------------------------------------------------
    story.append(Paragraph("三、预测模型真实评估（测试集，按时间顺序切分）", s["h2"]))
    story.append(_table(context["metrics_table"]))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        "特征：前1天海温/前3天均值/前7天均值/日变化趋势/年周期；"
        "模型：持续性基线 vs 随机森林（各步长单独训练，无未来数据泄漏）。", s["small"]))

    # ---- 未来预测 -----------------------------------------------------------
    story.append(Paragraph("四、未来 1~3 天海温预测", s["h2"]))
    story.append(_table(context["forecast"].astype(str)))

    # ---- 历史事件 -----------------------------------------------------------
    story.append(Paragraph("五、海洋热浪 / 冷事件记录（最近 10 条）", s["h2"]))
    events = context["events"].head(10)
    if events.empty:
        story.append(Paragraph("研究时段内未识别到达到标准的事件。", s["body"]))
    else:
        story.append(_table(events.astype(str)))
    story.append(Paragraph(
        f"事件判据：超过季节第 90/10 百分位且连续 ≥{config.MHW_MIN_DURATION} 天"
        f"（间隔 ≤{config.MHW_MAX_GAP} 天合并，Hobday et al., 2016），"
        f"气候态基准期 {config.CLIM_LABEL}"
        f"{'（WMO 30 年标准）' if config.CLIM_IS_STANDARD else '（短基准期探索性识别）'}。",
        s["small"]))

    # ---- 免责与说明 ---------------------------------------------------------
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        f"说明：本报告所有数值均由真实数据与统计算法自动计算生成；"
        f"气候态基准期 {config.CLIM_LABEL}"
        f"{'（WMO 30 年标准气候态）' if config.CLIM_IS_STANDARD else '（短于 WMO 30 年标准，探索性识别）'}；"
        "预测结果为统计模型输出，"
        "供科研与教学参考，不作为生产决策唯一依据。", s["small"]))

    doc.build(story)
    return buf.getvalue()
