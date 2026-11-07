# -*- coding: utf-8 -*-
"""
海温异常识别模块
================

两条技术路线，严格区分"普通海温异常"与"海洋热浪"：

1. **普通异常（格点级）**：
   每个格点相对其历史同期（逐"年日"气候态，11 天滑动窗口）的距平，
   用标准化距平 z = 距平 / 同期标准差，|z| ≥ 2 判为显著偏暖/偏冷。

2. **海洋热浪（MHW，区域级）**——遵循 Hobday et al. (2016) 定义：
   海温连续 ≥5 天高于"季节性第 90 百分位阈值"（基准期 11 天窗口汇集）；
   对称地，连续 ≥5 天低于第 10 百分位判为海洋冷事件（MCS）。

基准期：由 config.CLIM_YEARS 决定（当前为 1991–2020，WMO 标准 30 年气候态）。
缓存文件名包含基准期，更换基准期后自动重建。
"""

import warnings

import numpy as np
import pandas as pd
import xarray as xr

import config
import data_loader

CLIM_YEARS = config.CLIM_YEARS
# 缓存文件名包含基准期，基准期变化时自动失效重建
GRID_CLIM_PATH = config.PROCESSED_DATA_DIR / (
    f"grid_climatology_{CLIM_YEARS[0]}_{CLIM_YEARS[1]}.nc")
EVENTS_CSV = config.PROCESSED_DATA_DIR / (
    f"mhw_events_{CLIM_YEARS[0]}_{CLIM_YEARS[1]}.csv")


def _check_baseline_coverage(base, years: tuple[int, int]) -> None:
    """基准期数据覆盖检查：要求基准期内**每个年份都基本完整**。

    只数"出现了几个年份"是不够的——下载中途会出现多个残缺年份，
    若此时计算气候态并缓存，将产生永久性的错误基准。
    因此必须等到每个年份 ≥300 天才允许计算。
    """
    year_counts = (pd.Series(pd.DatetimeIndex(base.time.values).year)
                   .value_counts())
    incomplete = [y for y in range(years[0], years[1] + 1)
                  if int(year_counts.get(y, 0)) < 300]
    if incomplete:
        raise RuntimeError(
            f"气候态基准期 {years[0]}–{years[1]} 的数据尚不完整："
            f"{len(incomplete)} 个年份缺失或不足 300 天"
            f"（如 {incomplete[:3]}…{incomplete[-1] if len(incomplete) > 3 else ''}）。"
            "请等待 download_data.py 补全数据后再使用本功能。"
        )


# ---------------------------------------------------------------------------
# 网格级气候态
# ---------------------------------------------------------------------------
def compute_grid_climatology(sst: xr.DataArray,
                             window: int = config.CLIMATOLOGY_WINDOW,
                             years: tuple[int, int] = CLIM_YEARS) -> xr.Dataset:
    """计算每个格点逐"年日"的气候态统计量。

    返回 Dataset(doy=1..366, lat, lon)，变量：
    clim_mean / clim_std / clim_p90 / clim_p10
    """
    base = sst.where(
        (sst.time.dt.year >= years[0]) & (sst.time.dt.year <= years[1]),
        drop=True,
    )
    _check_baseline_coverage(base, years)
    doy_of_time = pd.DatetimeIndex(base.time.values).day_of_year.values
    data = base.values.astype(float)                 # (T, lat, lon)
    nlat, nlon = base.sizes["lat"], base.sizes["lon"]
    half = window // 2

    mean = np.full((366, nlat, nlon), np.nan)
    std = np.full_like(mean, np.nan)
    p90 = np.full_like(mean, np.nan)
    p10 = np.full_like(mean, np.nan)

    for doy in range(1, 367):
        # 跨年窗口：doy±half 按 366 取模
        win = {(doy - 1 + k) % 366 + 1 for k in range(-half, half + 1)}
        pool = data[np.isin(doy_of_time, list(win))]   # (N, lat, lon)
        valid = np.isfinite(pool)
        n = valid.sum(axis=0)
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean[doy - 1] = np.nanmean(pool, axis=0)
            std[doy - 1] = np.nanstd(pool, axis=0, ddof=1)
            p90[doy - 1] = np.where(n >= 10, np.nanpercentile(pool, 90, axis=0), np.nan)
            p10[doy - 1] = np.where(n >= 10, np.nanpercentile(pool, 10, axis=0), np.nan)

    clim = xr.Dataset(
        {
            "clim_mean": (("doy", "lat", "lon"), mean),
            "clim_std": (("doy", "lat", "lon"), std),
            "clim_p90": (("doy", "lat", "lon"), p90),
            "clim_p10": (("doy", "lat", "lon"), p10),
        },
        coords={"doy": np.arange(1, 367), "lat": base.lat, "lon": base.lon},
        attrs={"基准期": f"{years[0]}-{years[1]}", "窗口": f"{window} 天"},
    )
    return clim


def load_grid_climatology(sst: xr.DataArray) -> xr.Dataset:
    """带磁盘缓存的网格气候态：算一次存盘，之后直接读取。"""
    if GRID_CLIM_PATH.exists():
        return data_loader._open_from_memory(GRID_CLIM_PATH)
    clim = compute_grid_climatology(sst)
    data_loader._save_netcdf_safe(clim, GRID_CLIM_PATH)
    return clim


def _clim_at_dates(clim_var: xr.DataArray, times: pd.DatetimeIndex) -> np.ndarray:
    """按日期序列取气候态值：返回 (time, lat, lon) 数组。"""
    doy_idx = times.day_of_year.values - 1
    return clim_var.values[doy_idx]


def compute_anomaly(sst: xr.DataArray, clim: xr.Dataset) -> xr.Dataset:
    """计算逐日距平与标准化距平。

    返回 Dataset(time, lat, lon)：anomaly（℃）、z_score（无量纲）。
    """
    times = pd.DatetimeIndex(sst.time.values)
    mean = _clim_at_dates(clim["clim_mean"], times)
    std = _clim_at_dates(clim["clim_std"], times)
    std_safe = np.where(std < 0.1, np.nan, std)     # 避免稳定期除零
    anomaly = sst.values - mean
    z = anomaly / std_safe
    return xr.Dataset(
        {
            "anomaly": (("time", "lat", "lon"), anomaly),
            "z_score": (("time", "lat", "lon"), z),
        },
        coords={"time": sst.time, "lat": sst.lat, "lon": sst.lon},
    )


def grid_threshold_exceedance(sst_day: xr.DataArray, clim: xr.Dataset,
                              doy: int) -> tuple[int, int]:
    """统计单日达到网格级季节阈值的格点数。

    返回 (高于 p90 格点数, 低于 p10 格点数)。
    注意：这只是**单日阈值超越**计数，未做连续 ≥5 天的持续性判定，
    因此不能直接称为"海洋热浪格点"，界面须如实标注。
    """
    p90 = clim["clim_p90"].sel(doy=doy)
    p10 = clim["clim_p10"].sel(doy=doy)
    return int((sst_day > p90).sum()), int((sst_day < p10).sum())


def classify_day(anom_day: xr.Dataset, threshold: float = config.ANOMALY_STD_THRESHOLD
                 ) -> dict:
    """统计单日异常状况：显著偏暖/偏冷格点数、极值等。"""
    z = anom_day["z_score"].values
    anom = anom_day["anomaly"].values
    valid = np.isfinite(z)
    n_valid = int(valid.sum())
    return {
        "n_valid": n_valid,
        "n_warm": int(((z >= threshold) & valid).sum()),
        "n_cold": int(((z <= -threshold) & valid).sum()),
        "max_anom": float(np.nanmax(anom)) if np.isfinite(anom).any() else np.nan,
        "min_anom": float(np.nanmin(anom)) if np.isfinite(anom).any() else np.nan,
        "mean_anom": float(np.nanmean(anom)) if np.isfinite(anom).any() else np.nan,
        "mean_abs_z": float(np.nanmean(np.abs(z))) if n_valid else np.nan,
    }


# ---------------------------------------------------------------------------
# 海洋热浪 / 冷事件识别（Hobday et al. 2016，作用于区域平均序列）
# ---------------------------------------------------------------------------
def seasonal_stat(series: pd.Series,
                  window: int = config.CLIMATOLOGY_WINDOW,
                  years: tuple[int, int] = CLIM_YEARS,
                  smooth: int = 5) -> pd.DataFrame:
    """计算逐"年日"(1~366) 的气候态均值与第 90/10 百分位（索引为 doy）。

    11 天窗口汇集基准期样本，再做环形居中平滑，减小阶梯状跳跃。
    """
    base = series[(series.index.year >= years[0]) & (series.index.year <= years[1])]
    year_counts = base.index.year.value_counts()
    incomplete = [y for y in range(years[0], years[1] + 1)
                  if int(year_counts.get(y, 0)) < 300]
    if incomplete:
        raise RuntimeError(
            f"气候态基准期 {years[0]}–{years[1]} 的数据尚不完整："
            f"{len(incomplete)} 个年份缺失或不足 300 天。"
            "请等待 download_data.py 补全数据后再使用本功能。"
        )
    df = base.to_frame("sst")
    df["doy"] = df.index.day_of_year
    half = window // 2

    records = {}
    for doy in range(1, 367):
        win = {(doy - 1 + k) % 366 + 1 for k in range(-half, half + 1)}
        sample = df[df["doy"].isin(win)]["sst"]
        records[doy] = (sample.mean(), sample.quantile(0.9), sample.quantile(0.1))
    stat = pd.DataFrame.from_dict(records, orient="index",
                                  columns=["clim_mean", "p90", "p10"])
    # 环形居中平滑：首尾各补 smooth//2 天，滚动平均后取中间 366 天
    hs = smooth // 2
    if hs > 0:
        for col in stat.columns:
            v = stat[col].values
            ext = pd.Series(np.r_[v[-hs:], v, v[:hs]])
            stat[col] = ext.rolling(smooth, center=True,
                                    min_periods=1).mean().values[hs:-hs]
    stat.index = pd.Index(range(1, 367), name="doy")
    return stat


def thresholds_for_dates(stat: pd.DataFrame, dates) -> pd.DataFrame:
    """按任意日期序列查季节性阈值（用于历史序列和未来预测日期）。"""
    dates = pd.DatetimeIndex(dates)
    doy = dates.day_of_year.values - 1
    out = pd.DataFrame(index=dates)
    for col in stat.columns:
        out[col] = stat[col].values[doy]
    return out


def seasonal_thresholds(series: pd.Series, **kwargs) -> pd.DataFrame:
    """区域序列对应的季节性阈值（clim_mean / p90 / p10）。"""
    return thresholds_for_dates(seasonal_stat(series, **kwargs), series.index)


def detect_events(series: pd.Series, thresholds: pd.DataFrame,
                  kind: str = "heat",
                  min_duration: int = config.MHW_MIN_DURATION,
                  max_gap: int = config.MHW_MAX_GAP) -> pd.DataFrame:
    """识别连续超阈值事件（Hobday et al. 2016 完整规则）。

    kind="heat"：sst ≥ p90 为海洋热浪；kind="cold"：sst ≤ p10 为海洋冷事件。
    处理顺序（与文献一致）：
    1. 找出全部连续超阈值时段；
    2. **事件合并**：相邻时段间隔 ≤ max_gap 天时合并为同一事件
       （允许短暂回落不打断事件）；
    3. **持续天数过滤**：合并后总时长 ≥ min_duration 天才保留。
    返回事件表：开始/结束日期、持续天数、最大/平均/累计强度（相对气候态均值）。
    """
    if kind == "heat":
        exceed = series >= thresholds["p90"]
    else:
        exceed = series <= thresholds["p10"]
    intensity = series - thresholds["clim_mean"]

    # 第一步：收集连续超阈值时段 [start_idx, end_idx]
    runs = []
    in_event, start = False, None
    dates = series.index
    for i, flag in enumerate(exceed.fillna(False).values):
        if flag and not in_event:
            in_event, start = True, i
        elif not flag and in_event:
            runs.append((start, i - 1))
            in_event = False
    if in_event:
        runs.append((start, len(series) - 1))

    # 第二步：合并间隔 ≤ max_gap 天的相邻时段
    merged: list[list[int]] = []
    for s, e in runs:
        if merged and s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    # 第三步：按合并后的总时长过滤
    rows = []
    for s, e in merged:
        duration = e - s + 1
        if duration < min_duration:
            continue
        seg_int = intensity.iloc[s:e + 1]
        seg_sst = series.iloc[s:e + 1]
        rows.append({
            "类型": "海洋热浪" if kind == "heat" else "海洋冷事件",
            "开始日期": dates[s].date(), "结束日期": dates[e].date(),
            "持续天数": duration,
            "最大强度(℃)": round(float(seg_int.max() if kind == "heat" else seg_int.min()), 2),
            "平均强度(℃)": round(float(seg_int.mean()), 2),
            "累计强度(℃·天)": round(float(seg_int.sum()), 1),
            "峰值日期": dates[s + int(np.argmax(np.abs(seg_int.values)))].date(),
            "期间最高海温(℃)": round(float(seg_sst.max()), 2),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("开始日期", ascending=False).reset_index(drop=True)
    return df


def event_on_date(events: pd.DataFrame, day) -> pd.DataFrame:
    """查询某一天处于哪些事件之中。"""
    if events.empty:
        return events
    d = pd.Timestamp(day).date()
    hit = events[(events["开始日期"] <= d) & (events["结束日期"] >= d)]
    return hit


def detect_all_events(series: pd.Series,
                      min_duration: int = config.MHW_MIN_DURATION) -> pd.DataFrame:
    """一次性识别海洋热浪与海洋冷事件，并保存 CSV。"""
    thresholds = seasonal_thresholds(series)
    heat = detect_events(series, thresholds, "heat", min_duration)
    cold = detect_events(series, thresholds, "cold", min_duration)
    events = pd.concat([heat, cold], ignore_index=True)
    if not events.empty:
        events = events.sort_values("开始日期", ascending=False).reset_index(drop=True)
    events.to_csv(EVENTS_CSV, index=False, encoding="utf-8-sig")
    return events, thresholds
