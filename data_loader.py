# -*- coding: utf-8 -*-
"""
数据读取模块
============

负责读取 data/raw/ 下的逐日 OISST NetCDF 文件，合并为统一的时间序列数据集，
并提供海域掩膜等基础工具。

注意事项：
- 项目路径可能包含中文，netCDF4 底层 C 库无法直接打开中文路径，
  因此所有文件都先读成字节、再用 io.BytesIO 从内存打开。
- 单个裁剪后文件仅约 7KB，全部读入内存也没有压力。
"""

import re
from datetime import date
from pathlib import Path

import netCDF4
import numpy as np
import pandas as pd
import xarray as xr

import config

_FILE_PATTERN = re.compile(r"oisst_(\d{8})\.nc$")

# 合并缓存文件：把全部逐日文件合并成一个 NetCDF，避免每次都读几千个小文件
CONSOLIDATED_PATH = config.PROCESSED_DATA_DIR / "bohai_sst_grid.nc"


def list_raw_files(raw_dir: Path | None = None) -> list[tuple[date, Path]]:
    """列出原始目录下全部有效数据文件，返回 [(日期, 路径)]，按日期升序。"""
    raw_dir = raw_dir or config.RAW_DATA_DIR
    items = []
    for p in raw_dir.glob("oisst_*.nc"):
        m = _FILE_PATTERN.match(p.name)
        if m:
            items.append((date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8])), p))
    return sorted(items, key=lambda x: x[0])


def get_data_status() -> dict:
    """返回数据概况，用于界面展示与"无数据"判断。

    部署模式：即使没有逐日原始文件，只要存在合并缓存文件即视为有数据。
    """
    files = list_raw_files()
    if files:
        return {
            "n_files": len(files),
            "first_date": files[0][0],
            "last_date": files[-1][0],
        }
    if CONSOLIDATED_PATH.exists():
        ds = _open_from_memory(CONSOLIDATED_PATH)
        return {
            "n_files": int(ds.sizes["time"]),
            "first_date": pd.Timestamp(ds.time.values[0]).date(),
            "last_date": pd.Timestamp(ds.time.values[-1]).date(),
        }
    return {"n_files": 0, "first_date": None, "last_date": None}


def _open_from_memory(path: Path) -> xr.Dataset:
    """从内存打开 NetCDF（规避 netCDF4 不支持中文路径的问题）。

    netCDF4.Dataset 的 memory 参数可直接从字节串读取，
    再用 xarray 的 NetCDF4DataStore 包装成 Dataset。
    """
    nc = netCDF4.Dataset(path.name, memory=path.read_bytes())
    with xr.open_dataset(xr.backends.NetCDF4DataStore(nc)) as ds:
        return ds.load()


def load_sst(start: date | None = None, end: date | None = None,
             use_cache: bool = True) -> xr.DataArray:
    """加载并合并指定日期范围内的逐日海温，返回 (time, lat, lon) 的 DataArray。

    - 自动去除 zlev 等多余维度、统一时间坐标为日期；
    - 超出物理合理范围（-2 ~ 40 ℃）的值置为 NaN（海洋数据质量控制）；
    - 默认使用合并缓存：首次全量加载后保存为单个 NetCDF，之后秒开。
    """
    files = list_raw_files()
    if start:
        files = [(d, p) for d, p in files if d >= start]
    if end:
        files = [(d, p) for d, p in files if d <= end]
    if not files:
        # 部署模式：无逐日原始文件时，直接使用合并缓存文件
        if CONSOLIDATED_PATH.exists():
            return _open_from_memory(CONSOLIDATED_PATH)["sst"]
        raise FileNotFoundError(
            f"在 {config.RAW_DATA_DIR} 中没有找到可用数据文件。"
            "请先运行 python download_data.py 下载数据。"
        )

    # 合并缓存有效（覆盖范围与原始文件完全一致）时直接读取
    full_range = (start is None and end is None)
    if full_range and use_cache and CONSOLIDATED_PATH.exists():
        cached = _open_from_memory(CONSOLIDATED_PATH)["sst"]
        first_cached = pd.Timestamp(cached.time.values[0]).date()
        last_cached = pd.Timestamp(cached.time.values[-1]).date()
        # 首尾日期 + 天数三项都必须一致：下载进行中（范围相同但天数不足）也会触发重建
        if (first_cached <= files[0][0] and last_cached >= files[-1][0]
                and cached.sizes["time"] == len(files)):
            return cached

    arrays = []
    for day, path in files:
        ds = _open_from_memory(path)
        sst = ds["sst"]
        if "zlev" in sst.dims:
            sst = sst.isel(zlev=0, drop=True)
        sst = sst.squeeze(drop=True)
        # 以文件名日期为准（比文件内时间坐标更可靠）
        sst = sst.expand_dims(time=[np.datetime64(day)])
        arrays.append(sst)

    combined = xr.concat(arrays, dim="time").sortby("time")
    combined = combined.where((combined >= -2) & (combined <= 40))
    combined.name = "sst"
    combined.attrs.update(long_name="海表温度", units="°C",
                          source=config.DATA_SOURCE_NAME)

    if full_range and use_cache:
        _save_netcdf_safe(combined.to_dataset(), CONSOLIDATED_PATH)
    return combined


def _save_netcdf_safe(ds: xr.Dataset, dst: Path) -> None:
    """先写系统临时目录再移动（netCDF4 底层不支持中文路径）。"""
    import shutil
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        ds.to_netcdf(tmp_path)
        shutil.move(str(tmp_path), str(dst))
    finally:
        tmp_path.unlink(missing_ok=True)


def build_sea_mask(sst: xr.DataArray, min_valid_frac: float = 0.5) -> xr.DataArray:
    """构建海域掩膜：在研究时段内有效观测天数占比 ≥ min_valid_frac 的格点视为海洋。

    OISST 陆地区域本身为 NaN，该步骤同时可筛除常年缺测的格点。
    """
    valid_frac = sst.notnull().mean(dim="time")
    mask = valid_frac >= min_valid_frac
    mask.attrs.update(long_name="海域掩膜(True=海洋)")
    return mask


def apply_sea_mask(sst: xr.DataArray, mask: xr.DataArray) -> xr.DataArray:
    """应用海域掩膜，陆地及无效格点置为 NaN。"""
    return sst.where(mask)
