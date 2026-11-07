# -*- coding: utf-8 -*-
"""
海洋智能预警模块
================

把"异常识别"与"预测结果"融合成明确的风险等级。
全部规则为可解释的统计规则，不依赖大模型，规则文本在界面中公开。

风险等级（Ⅰ级最高；项目内部辅助提示规则，非官方海洋灾害预警标准）：
- Ⅰ级 红色（严重预警）：海洋热浪/冷事件持续 ≥10 天，或最大强度 ≥3℃
- Ⅱ级 橙色（警报）：处于海洋热浪/冷事件之中（未达Ⅰ级）
- Ⅲ级 黄色（预警）：≥30% 海域显著偏暖/偏冷（|z|≥2），或区域平均距平 ≥1.5℃
- Ⅳ级 蓝色（趋势关注）：存在显著异常格点、区域平均距平 ≥0.5℃，
  或预测未来 3 天内将突破季节第 90/10 百分位阈值。
  Ⅳ级只表示"出现值得留意的趋势"，不代表已发生统计显著异常或海洋热浪。
- 绿色（正常）：以上均不满足
"""

from dataclasses import dataclass, field

import pandas as pd

import anomaly_detection as ad
import config

# 等级定义：级别, 名称, 主题色
LEVEL_I = ("Ⅰ级", "严重预警", "#EF5350")
LEVEL_II = ("Ⅱ级", "警报", "#FF7043")
LEVEL_III = ("Ⅲ级", "预警", "#FFA726")
LEVEL_IV = ("Ⅳ级", "趋势关注", "#29B6F6")
LEVEL_OK = ("正常", "正常", "#66BB6A")

# 必须在界面与报告中展示的声明：本分级为项目自定义规则
DISCLAIMER = ("预警分级（Ⅰ～Ⅳ级）为本项目内部辅助提示规则，"
              "依据公开统计阈值自动生成，并非官方海洋灾害预警标准。")


@dataclass
class RiskAssessment:
    """一次风险评估结果。"""
    level: tuple                    # (级别, 名称, 颜色)
    risk_type: str                  # 异常类型：海洋热浪 / 显著偏暖 / …
    position: str                   # 风险位置（异常最显著的子区域）
    degree: str                     # 异常程度描述
    reasons: list = field(default_factory=list)    # 判定依据（含具体数值）
    suggestions: list = field(default_factory=list)  # 应对建议


SUGGESTIONS = {
    "海洋热浪": [
        "养殖提示：加强溶解氧监测，高温期减少投喂量",
        "生态提示：关注底栖生物与浮游生物群落变化",
        "建议对海参、贝类等高温敏感品种采取遮阳/换水措施",
    ],
    "海洋冷事件": [
        "养殖提示：注意低温冻害，提前做好保温与破冰准备",
        "关注海冰发展，避免养殖设施受损",
    ],
    "显著偏暖": [
        "关注后续发展，若持续升温可能演变为海洋热浪",
        "加强养殖海域水质与溶解氧监测",
    ],
    "显著偏冷": [
        "关注后续降温趋势，防范低温对养殖品种的影响",
    ],
    "偏暖趋势": [
        "当前为轻微偏暖，保持关注；若持续发展可能升级为更高级别预警",
        "建议留意未来 3 天预测与海温演变",
    ],
    "偏冷趋势": [
        "当前为轻微偏冷，保持关注",
        "建议留意未来 3 天预测与海温演变",
    ],
    "正常": ["海温处于历史同期正常范围，保持常规监测即可"],
}


def subregion_status(anom, day) -> pd.DataFrame:
    """计算各子区域某一天的风险状况（风险定位用）。"""
    day_anom = anom.sel(time=str(day))
    rows = []
    for name, box in config.SUB_REGIONS.items():
        lon_min, lon_max, lat_min, lat_max = box
        sub = day_anom.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
        z = sub["z_score"].values
        a = sub["anomaly"].values
        rows.append({
            "区域": name,
            "平均距平 (℃)": round(float(pd.Series(a.ravel()).mean()), 2),
            "显著偏暖格点": int((z >= config.ANOMALY_STD_THRESHOLD).sum()),
            "显著偏冷格点": int((z <= -config.ANOMALY_STD_THRESHOLD).sum()),
        })
    df = pd.DataFrame(rows)
    df["|平均距平|"] = df["平均距平 (℃)"].abs()
    return df.sort_values("|平均距平|", ascending=False).drop(columns="|平均距平|")


def forecast_risk(series: pd.Series, forecast: pd.DataFrame,
                  clim_years: tuple[int, int] | None = None) -> list[str]:
    """检查未来 1~3 天预测是否突破季节阈值，返回风险说明列表（可为空）。"""
    if forecast.empty:
        return []
    future_dates = pd.DatetimeIndex(forecast["预测日期"])
    stat = ad.seasonal_stat(series, years=clim_years or config.CLIM_YEARS)
    th = ad.thresholds_for_dates(stat, future_dates)
    preds = forecast["预测海温 (℃)"].values
    n_over = int((preds >= th["p90"].values).sum())
    n_under = int((preds <= th["p10"].values).sum())
    out = []
    if n_over > 0:
        out.append(f"预测未来 3 天内有 {n_over} 天将突破季节第 90 百分位，"
                   "存在发展为海洋热浪的风险")
    if n_under > 0:
        out.append(f"预测未来 3 天内有 {n_under} 天将跌破季节第 10 百分位，"
                   "存在发展为海洋冷事件的风险")
    return out


def assess(series: pd.Series, events: pd.DataFrame, day_stats: dict,
           forecast: pd.DataFrame, sub_status: pd.DataFrame,
           day, clim_years: tuple[int, int] | None = None) -> RiskAssessment:
    """按既定规则评估最新日期的风险等级。"""
    last = pd.Timestamp(day).date()
    mean_anom = day_stats["mean_anom"]
    n_valid = max(day_stats["n_valid"], 1)
    warm_ratio = day_stats["n_warm"] / n_valid
    cold_ratio = day_stats["n_cold"] / n_valid

    # 风险位置：平均距平绝对值最大的子区域（排除"整个渤海"本身）
    sub = sub_status[sub_status["区域"] != "整个渤海"]
    top = sub.iloc[sub["平均距平 (℃)"].abs().argmax()]
    position = f"{top['区域']}（平均距平 {top['平均距平 (℃)']:+.2f} ℃）"

    # 当前是否处于事件之中
    active = ad.event_on_date(events, last)
    mhw = active[active["类型"] == "海洋热浪"] if not active.empty else active
    mcs = active[active["类型"] == "海洋冷事件"] if not active.empty else active

    # 预测是否将突破阈值（未来 1~3 天）
    future_dates = pd.DatetimeIndex(forecast["预测日期"])
    stat = ad.seasonal_stat(series, years=clim_years or config.CLIM_YEARS)
    th = ad.thresholds_for_dates(stat, future_dates)
    preds = forecast["预测海温 (℃)"].values
    n_over_p90 = int((preds >= th["p90"].values).sum())
    n_under_p10 = int((preds <= th["p10"].values).sum())

    reasons, level, risk_type = [], LEVEL_OK, "正常"

    if not mhw.empty:
        ev = mhw.iloc[0]
        risk_type = "海洋热浪"
        reasons.append(f"海洋热浪进行中：{ev['开始日期']} 起已持续 {ev['持续天数']} 天")
        reasons.append(f"事件最大强度 {ev['最大强度(℃)']} ℃（相对气候态），"
                       f"峰值日 {ev['峰值日期']} 海温 {ev['期间最高海温(℃)']} ℃")
        if ev["持续天数"] >= 10 or ev["最大强度(℃)"] >= 3:
            level = LEVEL_I
            reasons.append("触发Ⅰ级规则：持续 ≥10 天或最大强度 ≥3℃")
        else:
            level = LEVEL_II
            reasons.append("触发Ⅱ级规则：处于海洋热浪事件中（Hobday 判据）")
    elif not mcs.empty:
        ev = mcs.iloc[0]
        risk_type = "海洋冷事件"
        reasons.append(f"海洋冷事件进行中：{ev['开始日期']} 起已持续 {ev['持续天数']} 天")
        reasons.append(f"事件最大强度 {ev['最大强度(℃)']} ℃")
        if ev["持续天数"] >= 10 or ev["最大强度(℃)"] <= -3:
            level = LEVEL_I
            reasons.append("触发Ⅰ级规则：持续 ≥10 天或最大强度 ≤-3℃")
        else:
            level = LEVEL_II
            reasons.append("触发Ⅱ级规则：处于海洋冷事件之中")
    elif warm_ratio >= 0.3 or mean_anom >= 1.5:
        level, risk_type = LEVEL_III, "显著偏暖"
        reasons.append(f"{warm_ratio:.0%} 海域标准化距平 z≥2（阈值 30%）"
                       if warm_ratio >= 0.3 else
                       f"区域平均距平 {mean_anom:+.2f} ℃（阈值 +1.5℃）")
    elif cold_ratio >= 0.3 or mean_anom <= -1.5:
        level, risk_type = LEVEL_III, "显著偏冷"
        reasons.append(f"{cold_ratio:.0%} 海域标准化距平 z≤-2（阈值 30%）"
                       if cold_ratio >= 0.3 else
                       f"区域平均距平 {mean_anom:+.2f} ℃（阈值 -1.5℃）")
    elif (day_stats["n_warm"] + day_stats["n_cold"] > 0
          or abs(mean_anom) >= 0.5 or n_over_p90 > 0 or n_under_p10 > 0):
        level = LEVEL_IV
        risk_type = "偏暖趋势" if mean_anom >= 0 else "偏冷趋势"
        if n_over_p90 > 0:
            reasons.append(f"预测未来 3 天内有 {n_over_p90} 天将突破季节第 90 百分位，"
                           "存在发展为海洋热浪的风险")
        if n_under_p10 > 0:
            reasons.append(f"预测未来 3 天内有 {n_under_p10} 天将跌破季节第 10 百分位")
        if day_stats["n_warm"] + day_stats["n_cold"] > 0:
            reasons.append(f"当日显著异常格点：偏暖 {day_stats['n_warm']} 个、"
                           f"偏冷 {day_stats['n_cold']} 个")
        if abs(mean_anom) >= 0.5:
            reasons.append(f"区域平均距平 {mean_anom:+.2f} ℃（关注阈值 0.5℃）")
    else:
        reasons.append(f"区域平均距平 {mean_anom:+.2f} ℃，"
                       "全海域标准化距平 |z|<2，无事件、无预测风险")

    degree = (f"区域平均距平 {mean_anom:+.2f} ℃；"
              f"最大单点距平 {day_stats['max_anom']:+.2f} ℃")
    return RiskAssessment(
        level=level, risk_type=risk_type, position=position, degree=degree,
        reasons=reasons, suggestions=SUGGESTIONS.get(risk_type, SUGGESTIONS["正常"]),
    )
