# -*- coding: utf-8 -*-
"""
数据预处理与特征工程模块
========================

功能：
1. 提取指定海域的逐日平均海温时间序列（纬度余弦加权）；
2. 计算历史同期（气候态）基准：逐"年日"均值与标准差，11 天滑动窗口；
3. 生成机器学习特征（滞后、滑动均值、趋势、周期）；
4. 处理结果保存为 CSV。

基准期说明：气候态基准期由 config.CLIM_YEARS 决定（当前 1991–2020，
WMO 标准 30 年气候态）。若基准期被改短，界面与报告会自动提示
"短基准期探索性识别"。
"""

from datetime import date

import numpy as np
import pandas as pd
import xarray as xr

import config

# 气候态基准期（含首尾年份），统一从 config 读取（WMO 标准：1991–2020）
CLIM_START_YEAR = config.CLIM_START_YEAR
CLIM_END_YEAR = config.CLIM_END_YEAR


def region_timeseries(sst: xr.DataArray, box: tuple[float, float, float, float]) -> pd.Series:
    """提取指定区域 (lon_min, lon_max, lat_min, lat_max) 的逐日面积加权平均海温。

    使用 cos(lat) 权重，减小高纬格点面积偏小带来的偏差。
    """
    lon_min, lon_max, lat_min, lat_max = box
    sub = sst.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
    weights = np.cos(np.deg2rad(sub.lat))
    daily = sub.weighted(weights).mean(dim=("lat", "lon"), skipna=True)
    series = daily.to_series()
    series.index = pd.DatetimeIndex(series.index).normalize()
    series.name = "sst"
    return series


def compute_climatology(series: pd.Series, window: int = config.CLIMATOLOGY_WINDOW,
                        start_year: int = CLIM_START_YEAR,
                        end_year: int = CLIM_END_YEAR) -> pd.DataFrame:
    """计算逐"年日"(1~366)的气候态均值与标准差。

    采用 ±window//2 天的滑动窗口汇集样本（如 11 天窗口），
    仅使用基准期 [start_year, end_year] 内的数据，避免使用预测期信息。
    返回 DataFrame(index=day_of_year, columns=[clim_mean, clim_std])。
    """
    base = series[(series.index.year >= start_year) & (series.index.year <= end_year)]
    year_counts = base.index.year.value_counts()
    incomplete = [y for y in range(start_year, end_year + 1)
                  if int(year_counts.get(y, 0)) < 300]
    if incomplete:
        raise RuntimeError(
            f"气候态基准期 {start_year}–{end_year} 的数据尚不完整："
            f"{len(incomplete)} 个年份缺失或不足 300 天。"
            "请等待 download_data.py 补全数据后再使用本功能。"
        )
    df = base.to_frame("sst")
    df["doy"] = df.index.day_of_year

    half = window // 2
    records = {}
    for doy in range(1, 367):
        # 处理跨年日窗口（如 1 月 1 日的窗口包含 12 月末）
        doys = [(doy - 1 + k) % 366 + 1 for k in range(-half, half + 1)]
        sample = df[df["doy"].isin(doys)]["sst"]
        records[doy] = (sample.mean(), sample.std(ddof=1), len(sample))
    clim = pd.DataFrame.from_dict(
        records, orient="index", columns=["clim_mean", "clim_std", "n_samples"]
    )
    clim.index.name = "doy"
    return clim


def attach_baseline(series: pd.Series, clim: pd.DataFrame) -> pd.DataFrame:
    """把时间序列与气候态基准对齐，返回含 sst / 基准 / 距平 的 DataFrame。"""
    df = series.to_frame("sst")
    df["doy"] = df.index.day_of_year
    df = df.join(clim[["clim_mean", "clim_std"]], on="doy")
    df["anomaly"] = df["sst"] - df["clim_mean"]
    return df.drop(columns=["doy"])


def build_features(series: pd.Series, horizons: list[int] | None = None) -> pd.DataFrame:
    """由逐日海温序列构建机器学习特征表（供第四阶段预测模块使用）。

    特征（全部只使用当天及历史信息，避免未来数据泄漏）：
    - lag_1      : 前 1 天海温
    - roll_3     : 前 3 天均值（含当天）
    - roll_7     : 前 7 天均值（含当天）
    - diff_1     : 日变化量（当天 - 前 1 天，刻画趋势）
    - doy_sin/cos: 年日周期特征（正弦/余弦编码）
    """
    df = pd.DataFrame({"sst": series})
    df["lag_1"] = df["sst"].shift(1)
    df["roll_3"] = df["sst"].rolling(3).mean()
    df["roll_7"] = df["sst"].rolling(7).mean()
    df["diff_1"] = df["sst"].diff(1)
    doy = df.index.day_of_year
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def save_processed(df: pd.DataFrame | pd.Series, filename: str) -> str:
    """保存处理结果到 data/processed/，返回保存路径。"""
    out = config.PROCESSED_DATA_DIR / filename
    df.to_csv(out, encoding="utf-8-sig")
    return str(out)


def prepare_monitoring_data(sst: xr.DataArray) -> dict:
    """为"海洋环境监测中心"准备全套数据：各区域时间序列 + 气候态基准。

    返回 {区域名: {"series": Series, "clim": DataFrame}}，并保存整渤海 CSV。
    """
    result = {}
    for name, box in config.SUB_REGIONS.items():
        series = region_timeseries(sst, box)
        clim = compute_climatology(series)
        result[name] = {"series": series, "clim": clim}
        if name == "整个渤海":
            save_processed(attach_baseline(series, clim), "bohai_sst_daily.csv")
    return result
