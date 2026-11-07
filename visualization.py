# -*- coding: utf-8 -*-
"""
可视化模块（Plotly）
====================

统一的深海蓝主题图表组件：
- 海温二维分布图
- 区域海温历史曲线（含历史同期基准）
- 距平分布图（第三阶段使用）
"""

import json
from functools import lru_cache

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

import config

# ---------------------------------------------------------------------------
# 海岸线与海域标注（Natural Earth 10m 真实海岸线，提取自渤海周边）
# ---------------------------------------------------------------------------
_COASTLINE_PATH = config.BASE_DIR / "assets" / "coastline_bohai.json"

# 主要海域名称标注位置（经纬度按各海湾几何中心人工设定）
SEA_LABELS = {
    "辽东湾": (121.35, 40.35),
    "渤海湾": (118.15, 38.75),
    "莱州湾": (119.55, 37.75),
    "渤海中部": (120.15, 38.95),
    "渤海海峡": (121.45, 38.15),
}


@lru_cache(maxsize=1)
def _load_coastline() -> list:
    """加载海岸线线段（带缓存）；文件缺失时返回空列表，地图照常可用。"""
    if not _COASTLINE_PATH.exists():
        return []
    with open(_COASTLINE_PATH, encoding="utf-8") as f:
        return json.load(f)["polylines"]


def _add_geography(fig: go.Figure, show_labels: bool = True) -> None:
    """在地图上叠加真实海岸线与海域名称标注。"""
    for seg in _load_coastline():
        fig.add_trace(go.Scatter(
            x=[p[0] for p in seg], y=[p[1] for p in seg],
            mode="lines", line=dict(color="rgba(232,241,250,0.75)", width=1.6),
            showlegend=False, hoverinfo="skip",
        ))
    if show_labels:
        for name, (lon, lat) in SEA_LABELS.items():
            fig.add_annotation(
                x=lon, y=lat, text=name, showarrow=False,
                font=dict(color="#B3E5FC", size=11),
                bgcolor="rgba(10,25,48,0.55)",
            )

# 深海蓝主题统一配色
BG_COLOR = "rgba(10, 25, 48, 0)"
PAPER_COLOR = "rgba(18, 41, 77, 0)"
FONT_COLOR = "#E8F1FA"
GRID_COLOR = "rgba(144, 202, 249, 0.15)"

# Plotly 图表全局配置：彻底隐藏英文工具栏
# （plotly.js 精简包不含中文语言包，工具栏提示无法翻译，故直接隐藏；
#   图表缩放/框选/悬停提示等交互不受影响，悬停文本全部为中文）
PLOTLY_CONFIG = {
    "displayModeBar": False,         # 隐藏英文工具栏按钮
    "displaylogo": False,            # 隐藏 Plotly 英文标识
    "responsive": True,
}


def _apply_dark_theme(fig: go.Figure, height: int = 480) -> go.Figure:
    """应用统一的深色主题样式。

    底部边距加大到 85：为横向图例（位于图表下方）预留空间，
    保证标题（顶部）与图例（底部）互不重叠。
    """
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=PAPER_COLOR,
        plot_bgcolor=BG_COLOR,
        font=dict(color=FONT_COLOR),
        margin=dict(l=10, r=10, t=50, b=85),
    )
    fig.update_xaxes(gridcolor=GRID_COLOR)
    fig.update_yaxes(gridcolor=GRID_COLOR)
    return fig


def sst_map(sst_2d: xr.DataArray, title: str = "渤海海表温度分布",
            zmin: float | None = None, zmax: float | None = None,
            colorscale: str = "RdYlBu_r", height: int = 520) -> go.Figure:
    """绘制单日期海温二维分布图。

    陆地区域为 NaN，显示为背景色；经纬度轴按实际比例缩放。
    """
    lon = sst_2d.lon.values
    lat = sst_2d.lat.values
    z = sst_2d.values.astype(float)
    if np.ndim(z) > 2:
        z = np.squeeze(z)

    if zmin is None or zmax is None:
        # 用 2%~98% 分位数固定色标范围，避免极端值压缩对比度
        finite = z[np.isfinite(z)]
        zmin = float(np.nanpercentile(finite, 2)) if finite.size else 0.0
        zmax = float(np.nanpercentile(finite, 98)) if finite.size else 1.0

    fig = go.Figure(go.Heatmap(
        x=lon, y=lat, z=z,
        zmin=zmin, zmax=zmax,
        colorscale=colorscale,
        colorbar=dict(title="℃", thickness=14),
        hovertemplate="经度 %{x:.2f}°E<br>纬度 %{y:.2f}°N<br>海温 %{z:.2f} ℃<extra></extra>",
    ))
    _add_geography(fig)                    # 叠加真实海岸线与海域名称
    _apply_dark_theme(fig, height)
    fig.update_layout(title=title)
    # 经纬度等比例显示（考虑纬度伸缩）
    fig.update_yaxes(scaleanchor="x", scaleratio=np.cos(np.deg2rad(lat.mean())),
                     title="纬度 (°N)")
    fig.update_xaxes(title="经度 (°E)")
    return fig


# 历史曲线时间窗口选项（"全部历史"使用月度均值降采样，仅用于展示，不改原始数据）
HIST_WINDOWS = {
    "最近 30 天": 30,
    "最近 1 年": 365,
    "最近 5 年": 365 * 5,
    "全部历史（月均）": None,
}


def sst_history_chart(series: pd.Series, clim: pd.DataFrame | None = None,
                      title: str = "区域海温历史变化", height: int = 400,
                      window: str = "最近 1 年") -> go.Figure:
    """区域海温时间序列，支持时间窗口切换与全历史月度降采样。

    window="全部历史（月均）" 时显示逐月均值（由原始逐日数据聚合，
    原始数据不变），避免 13000+ 个点挤在一张图上。
    """
    days = HIST_WINDOWS.get(window, 365)
    if days is not None:
        hist = series.iloc[-days:]
        line_name = "实际海温（逐日）"
    else:
        hist = series.resample("MS").mean()          # 月均，仅用于展示
        line_name = "实际海温（月均）"

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, mode="lines",
        name=line_name, line=dict(color="#29B6F6", width=1.6),
        hovertemplate="%{x|%Y-%m-%d}<br>海温 %{y:.2f} ℃<extra></extra>",
    ))
    if clim is not None:
        baseline = pd.Series(
            clim.loc[hist.index.day_of_year, "clim_mean"].values, index=hist.index
        )
        fig.add_trace(go.Scatter(
            x=baseline.index, y=baseline.values, mode="lines",
            name=f"历史同期基准 ({config.CLIM_LABEL})",
            line=dict(color="#FFA726", width=1.4, dash="dash"),
            hovertemplate="%{x|%Y-%m-%d}<br>基准 %{y:.2f} ℃<extra></extra>",
        ))
    _apply_dark_theme(fig, height)
    fig.update_layout(
        title=title,
        xaxis_title="日期", yaxis_title="海温 (℃)",
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
    )
    return fig


def prediction_chart(test_df: pd.DataFrame, title: str = "测试集：真实值 vs 预测值",
                     height: int = 420) -> go.Figure:
    """测试集真实值与各模型预测值对比曲线。"""
    fig = go.Figure()
    styles = {
        "真实值": dict(color="#FFFFFF", width=2.2),
        "持续性基线": dict(color="#90CAF9", width=1.4, dash="dot"),
        "随机森林": dict(color="#FFA726", width=1.6, dash="dash"),
    }
    for col in test_df.columns:
        fig.add_trace(go.Scatter(
            x=test_df.index, y=test_df[col], mode="lines", name=col,
            line=styles.get(col, dict(width=1.2)),
        ))
    _apply_dark_theme(fig, height)
    fig.update_layout(
        title=title, xaxis_title="日期", yaxis_title="海温 (℃)",
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
    )
    return fig


def feature_importance_chart(importance: pd.Series,
                             title: str = "随机森林特征重要性",
                             height: int = 320) -> go.Figure:
    """特征重要性横向条形图。"""
    labels = {
        "lag_1": "前 1 天海温", "roll_3": "前 3 天均值", "roll_7": "前 7 天均值",
        "diff_1": "日变化趋势", "doy_sin": "年周期(sin)", "doy_cos": "年周期(cos)",
    }
    s = importance.rename(labels).sort_values()
    fig = go.Figure(go.Bar(
        x=s.values, y=s.index, orientation="h",
        marker_color="#29B6F6",
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    _apply_dark_theme(fig, height)
    fig.update_layout(title=title, xaxis_title="相对重要性")
    return fig


def residual_chart(test_df: pd.DataFrame, model_col: str = "随机森林",
                   title: str = "测试集残差分布", height: int = 320) -> go.Figure:
    """预测残差（预测值 − 真实值）分布直方图，基线与随机森林叠加对比。

    理想模型应集中在 0 附近且比基线更窄；直方图如实展示，不做任何平滑修饰。
    """
    fig = go.Figure()
    for col, color in [("持续性基线", "#90CAF9"), (model_col, "#FFA726")]:
        resid = test_df[col] - test_df["真实值"]
        fig.add_trace(go.Histogram(
            x=resid, name=f"{col} 残差", opacity=0.65,
            marker_color=color, nbinsx=30,
        ))
    fig.update_layout(barmode="overlay")
    fig.add_vline(x=0, line=dict(color="#FFFFFF", width=1, dash="dash"))
    _apply_dark_theme(fig, height)
    fig.update_layout(
        title=title, yaxis_title="天数",
        xaxis_title="",               # 标题已含"残差"，避免与底部图例拥挤
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
    )
    return fig


def sst_trend_forecast_chart(series: pd.Series, clim: pd.DataFrame,
                             future: pd.DataFrame, days: int = 365,
                             title: str = "海温趋势与未来预测",
                             height: int = 420) -> go.Figure:
    """近一年海温 + 历史同期基准 + 未来 1~3 天预测（预测段虚线区分）。"""
    hist = series.iloc[-days:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist.index, y=hist.values, mode="lines", name="实际海温",
        line=dict(color="#29B6F6", width=1.6),
        hovertemplate="%{x|%Y-%m-%d}<br>海温 %{y:.2f} ℃<extra></extra>",
    ))
    if clim is not None:
        baseline = pd.Series(
            clim.loc[hist.index.day_of_year, "clim_mean"].values, index=hist.index)
        fig.add_trace(go.Scatter(
            x=baseline.index, y=baseline.values, mode="lines",
            name=f"历史同期基准 ({config.CLIM_LABEL})",
            line=dict(color="#FFA726", width=1.3, dash="dash"),
        ))
    # 预测段：从最后一天连接到未来预测点（虚线 + 标记，明确区别于实测）
    fx = [hist.index[-1]] + list(pd.DatetimeIndex(future["预测日期"]))
    fy = [hist.iloc[-1]] + list(future["预测海温 (℃)"])
    fig.add_trace(go.Scatter(
        x=fx, y=fy, mode="lines+markers", name="未来 1~3 天预测",
        line=dict(color="#EF5350", width=1.8, dash="dot"),
        marker=dict(size=7, symbol="diamond"),
        hovertemplate="%{x|%Y-%m-%d}<br>预测 %{y:.2f} ℃<extra></extra>",
    ))
    _apply_dark_theme(fig, height)
    fig.update_layout(
        title=title, xaxis_title="日期", yaxis_title="海温 (℃)",
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
    )
    return fig


def anomaly_series_chart(series: pd.Series, thresholds: pd.DataFrame,
                         events: pd.DataFrame | None = None,
                         title: str = "区域海温与季节性阈值",
                         height: int = 400,
                         window: str = "最近 1 年") -> go.Figure:
    """区域海温序列 + 第 90/10 百分位阈值带 + 海洋热浪事件高亮。

    window 支持 HIST_WINDOWS 中的选项；"全部历史（月均）" 时序列与阈值
    都按逐月展示（原始逐日数据不变）。
    """
    days = HIST_WINDOWS.get(window, 365)
    if days is not None:
        series = series.iloc[-days:]
        thresholds = thresholds.iloc[-days:]
    else:
        series = series.resample("MS").mean()
        thresholds = thresholds.resample("MS").mean()
    fig = go.Figure()
    # 阈值带（p10~p90 之间的正常范围）
    fig.add_trace(go.Scatter(
        x=series.index, y=thresholds["p90"], mode="lines",
        name="第 90 百分位（热浪阈值）", line=dict(color="#EF5350", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=series.index, y=thresholds["p10"], mode="lines",
        name="第 10 百分位（冷事件阈值）", line=dict(color="#42A5F5", width=1, dash="dot"),
        fill="tonexty", fillcolor="rgba(41, 182, 246, 0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=series.index, y=thresholds["clim_mean"], mode="lines",
        name="气候态均值", line=dict(color="#FFA726", width=1, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=series.index, y=series.values, mode="lines",
        name="实际海温", line=dict(color="#FFFFFF", width=1.6),
    ))
    # 高亮事件时段（只画与当前显示窗口有交集的，避免撑大坐标范围）
    win_start, win_end = series.index.min().date(), series.index.max().date()
    if events is not None and not events.empty:
        for _, ev in events.iterrows():
            if ev["结束日期"] < win_start or ev["开始日期"] > win_end:
                continue
            color = "rgba(239, 83, 80, 0.25)" if ev["类型"] == "海洋热浪" \
                else "rgba(66, 165, 245, 0.25)"
            fig.add_vrect(x0=str(ev["开始日期"]), x1=str(ev["结束日期"]),
                          fillcolor=color, line_width=0)
    fig.update_xaxes(range=[str(win_start), str(win_end)])
    _apply_dark_theme(fig, height)
    fig.update_layout(
        title=title, xaxis_title="日期", yaxis_title="海温 (℃)",
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
    )
    return fig


def anomaly_map(anom_2d: xr.DataArray, title: str = "海温距平分布",
                height: int = 520) -> go.Figure:
    """绘制海温距平二维分布图（红暖蓝冷，0 为中心）。"""
    z = anom_2d.values.astype(float)
    if np.ndim(z) > 2:
        z = np.squeeze(z)
    finite = z[np.isfinite(z)]
    zabs = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
    zabs = max(zabs, 0.5)   # 避免全场接近 0 时色标退化

    fig = go.Figure(go.Heatmap(
        x=anom_2d.lon.values, y=anom_2d.lat.values, z=z,
        zmin=-zabs, zmax=zabs, zmid=0,
        colorscale="RdBu_r",
        colorbar=dict(title="℃", thickness=14),
        hovertemplate="经度 %{x:.2f}°E<br>纬度 %{y:.2f}°N<br>距平 %{z:+.2f} ℃<extra></extra>",
    ))
    _add_geography(fig)                    # 叠加真实海岸线与海域名称
    _apply_dark_theme(fig, height)
    fig.update_layout(title=title)
    fig.update_yaxes(scaleanchor="x",
                     scaleratio=np.cos(np.deg2rad(float(anom_2d.lat.mean()))),
                     title="纬度 (°N)")
    fig.update_xaxes(title="经度 (°E)")
    return fig


def anomaly_status_map(z_2d: xr.DataArray,
                       threshold: float = config.ANOMALY_STD_THRESHOLD,
                       title: str = "异常状态分布", height: int = 520) -> go.Figure:
    """异常状态三分类地图：显著偏冷 / 正常 / 显著偏暖（基于标准化距平 z）。"""
    z = z_2d.values.astype(float)
    if np.ndim(z) > 2:
        z = np.squeeze(z)
    cat = np.where(z >= threshold, 1.0, np.where(z <= -threshold, -1.0, 0.0))
    cat[~np.isfinite(z)] = np.nan          # 陆地/缺测不显示

    fig = go.Figure(go.Heatmap(
        x=z_2d.lon.values, y=z_2d.lat.values, z=cat,
        zmin=-1, zmax=1,
        colorscale=[[0, "#1E88E5"], [0.5, "#26374A"], [1, "#E53935"]],
        colorbar=dict(title="状态", thickness=14,
                      tickvals=[-1, 0, 1],
                      ticktext=["显著偏冷", "正常", "显著偏暖"]),
        hovertemplate="经度 %{x:.2f}°E<br>纬度 %{y:.2f}°N<br>"
                      "z 分数 %{customdata:.2f}<extra></extra>",
        customdata=z,
    ))
    _add_geography(fig)
    _apply_dark_theme(fig, height)
    fig.update_layout(title=title + f"（|z| ≥ {threshold} 为显著）")
    fig.update_yaxes(scaleanchor="x",
                     scaleratio=np.cos(np.deg2rad(float(z_2d.lat.mean()))),
                     title="纬度 (°N)")
    fig.update_xaxes(title="经度 (°E)")
    return fig


def cell_trend_chart(cell_series: pd.Series, base_values: pd.Series | None = None,
                     title: str = "格点海温", height: int = 280) -> go.Figure:
    """单个格点的近期海温曲线（格点详情面板用，紧凑版）。"""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cell_series.index, y=cell_series.values, mode="lines",
        name="格点海温", line=dict(color="#29B6F6", width=1.8),
        hovertemplate="%{x|%Y-%m-%d}<br>海温 %{y:.2f} ℃<extra></extra>",
    ))
    if base_values is not None:
        fig.add_trace(go.Scatter(
            x=base_values.index, y=base_values.values, mode="lines",
            name=f"同期基准 ({config.CLIM_LABEL})",
            line=dict(color="#FFA726", width=1.2, dash="dash"),
        ))
    _apply_dark_theme(fig, height)
    fig.update_layout(
        title=title, xaxis_title="日期", yaxis_title="海温 (℃)",
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0),
    )
    return fig


def event_timeline_chart(events: pd.DataFrame, years: int = 3,
                         height: int = 320) -> go.Figure:
    """近期异常事件时间线（Plotly Timeline，真实日历日期轴）。

    海洋热浪红色、海洋冷事件蓝色；悬停显示事件详情。
    纵轴为事件（按开始日期排列），横轴为真实日历日期。
    """
    import plotly.express as px

    if events.empty:
        return go.Figure()
    df = events.copy()
    cutoff = pd.Timestamp.today().date() - pd.Timedelta(days=365 * years)
    df = df[df["结束日期"] >= cutoff]
    if df.empty:
        return go.Figure()
    df = df.sort_values("开始日期")
    # Timeline 需要真正的日期类型列；结束日期 +1 天让末日包含在条内
    df["start"] = pd.to_datetime(df["开始日期"])
    df["end"] = pd.to_datetime(df["结束日期"]) + pd.Timedelta(days=1)
    df["事件"] = (df["类型"] + " " + df["开始日期"].astype(str))
    # 持续天数使用事件表中已计算的列（不重算、不改原始值）

    fig = px.timeline(
        df, x_start="start", x_end="end", y="事件", color="类型",
        color_discrete_map={"海洋热浪": "#EF5350", "海洋冷事件": "#42A5F5"},
        custom_data=["持续天数", "最大强度(℃)", "峰值日期"],
    )
    fig.update_traces(
        hovertemplate="%{y}<br>%{customdata[0]} 天 ｜ 最大强度 %{customdata[1]:+.2f} ℃"
                      " ｜ 峰值 %{customdata[2]}<extra></extra>"
    )
    fig.update_yaxes(autorange="reversed", title="")
    _apply_dark_theme(fig, height)
    fig.update_layout(
        title=f"近 {years} 年异常事件时间线（红=海洋热浪，蓝=海洋冷事件）",
        xaxis_title="日期",
        legend=dict(orientation="h", yanchor="top", y=-0.14, x=0, title=""),
    )
    return fig
