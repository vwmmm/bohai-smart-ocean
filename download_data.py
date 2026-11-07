# -*- coding: utf-8 -*-
"""
NOAA OISST v2.1 数据下载脚本
============================

数据流（自动三级回退）：
1. **首选 Azure 公共镜像**（noaacdr.blob.core.windows.net，微软 AI for Earth /
   Planetary Computer 镜像）：下载完整逐日全球文件（约 1.5MB），
   下载后立即在本地裁剪出渤海区域并删除全球文件，磁盘占用极小。
2. **次选 AWS S3 公共桶**（noaa-cdr-sea-surface-temp-optimum-interpolation-pds，
   NOAA Open Data 计划镜像）：与 Azure 内容相同。
3. **回退 NCEI THREDDS NCSS**：当镜像上某日期缺失（如最近几天同步延迟），
   改用 NCEI 的 NCSS 子集服务直接获取裁剪后的小文件（速度较慢）。

用法：
    python download_data.py                          # 默认 2020-01-01 至今
    python download_data.py --start 2025-01-01 --end 2025-12-31
    python download_data.py --threads 8

若本机无法访问外网，可在能上网的电脑上运行本脚本，
再把 data/raw/ 目录整体拷贝过来即可，主程序离线可用。
"""

import argparse
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests
import xarray as xr
from tqdm import tqdm

import config

AZURE_BASE = "https://noaacdr.blob.core.windows.net/sea-surface-temp-optimum-interpolation"
S3_BASE = "https://noaa-cdr-sea-surface-temp-optimum-interpolation-pds.s3.amazonaws.com"
NCEI_NCSS = "https://www.ncei.noaa.gov/thredds/ncss/grid/OisstBase/NetCDF/V2.1/AVHRR"


def _normalize_and_save(src: Path, dst: Path) -> None:
    """统一数据格式并保存到 dst。

    - S3 下载的是全球文件：裁剪出渤海区域；
    - NCSS 下载的已是裁剪后的小文件：原样读取即可；
    - 两种来源都统一去掉多余的 zlev 维度。

    注意：netCDF4 底层 C 库无法打开中文路径，因此先写入系统临时目录
    （纯 ASCII 路径），再用 shutil.move 搬到目标位置（Python 层面支持中文路径）。
    """
    ds = xr.open_dataset(src)
    sst = ds["sst"]
    if "zlev" in sst.dims:
        sst = sst.isel(zlev=0, drop=True)
    if float(sst.lon.max()) > 180:      # OISST 经度为 0~360，与区域定义一致无需转换
        pass
    sub = sst.sel(
        lon=slice(config.LON_MIN, config.LON_MAX),
        lat=slice(config.LAT_MIN, config.LAT_MAX),
    )
    sub.load()  # 读入内存后再关闭文件，避免 Windows 文件锁
    ds.close()
    with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        sub.to_netcdf(tmp_path)
        shutil.move(str(tmp_path), str(dst))
    finally:
        tmp_path.unlink(missing_ok=True)


def _download_mirror(day: date, dst: Path, base_url: str,
                     timeout: int = 120) -> bool:
    """从 Azure / AWS S3 镜像下载单日文件并裁剪，成功返回 True。"""
    yyyymm, yyyymmdd = day.strftime("%Y%m"), day.strftime("%Y%m%d")
    for suffix in ("", "_preliminary"):      # 正式文件优先，近实时文件兜底
        url = f"{base_url}/data/v2.1/avhrr/{yyyymm}/oisst-avhrr-v02r01.{yyyymmdd}{suffix}.nc"
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp.write(resp.content)
                tmp_path = Path(tmp.name)
            try:
                _normalize_and_save(tmp_path, dst)
            finally:
                tmp_path.unlink(missing_ok=True)
            return True
        except (requests.RequestException, OSError, ValueError):
            continue
    return False


def _download_ncss(day: date, dst: Path, timeout: int = 90) -> bool:
    """回退方案：NCEI NCSS 子集服务（慢，但只需传几 KB）。"""
    yyyymm, yyyymmdd = day.strftime("%Y%m"), day.strftime("%Y%m%d")
    url = (
        f"{NCEI_NCSS}/{yyyymm}/oisst-avhrr-v02r01.{yyyymmdd}.nc"
        f"?var=sst&north={config.LAT_MAX}&south={config.LAT_MIN}"
        f"&west={config.LON_MIN}&east={config.LON_MAX}"
        "&horizStride=1&accept=netcdf3"
    )
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code != 200 or not resp.content.startswith(b"CDF"):
            return False
        dst.write_bytes(resp.content)
        return True
    except requests.RequestException:
        return False


def download_one(day: date, out_dir: Path, retries: int = 2) -> tuple[date, str]:
    """下载单日数据：Azure → S3 → NCSS 三级回退。返回 (日期, 状态)。"""
    out_path = out_dir / f"oisst_{day.strftime('%Y%m%d')}.nc"
    if out_path.exists() and out_path.stat().st_size > 1000:
        return day, "已存在"

    if _download_mirror(day, out_path, AZURE_BASE):
        return day, "成功(Azure)"
    for attempt in range(retries):
        if _download_mirror(day, out_path, S3_BASE):
            return day, "成功(S3)"
        time.sleep(1)
    if _download_ncss(day, out_path):
        return day, "成功(NCSS)"
    return day, "失败或无数据"


def main() -> None:
    parser = argparse.ArgumentParser(description="下载渤海区域 NOAA OISST v2.1 逐日海温数据")
    parser.add_argument("--start", default="2020-01-01", help="起始日期 YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="结束日期 YYYY-MM-DD（默认今天）")
    parser.add_argument("--threads", type=int, default=8, help="并发下载线程数")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()

    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    out_dir = config.RAW_DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据源 : {config.DATA_SOURCE_NAME}")
    print(f"下载源 : Azure (Planetary Computer) → AWS S3 → NCEI NCSS 三级回退")
    print(f"区域   : {config.LON_MIN}E-{config.LON_MAX}E, "
          f"{config.LAT_MIN}N-{config.LAT_MAX}N")
    print(f"范围   : {start} ~ {end}，共 {len(days)} 天，{args.threads} 线程\n")

    stats: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futures = {pool.submit(download_one, d, out_dir): d for d in days}
        with tqdm(total=len(days), desc="下载进度", unit="day") as bar:
            for fut in as_completed(futures):
                _, status = fut.result()
                stats[status] = stats.get(status, 0) + 1
                bar.update(1)
                bar.set_postfix(Azure=stats.get("成功(Azure)", 0),
                                S3=stats.get("成功(S3)", 0),
                                NCSS=stats.get("成功(NCSS)", 0),
                                跳过=stats.get("已存在", 0))

    print("\n下载完成，统计：")
    for status, n in sorted(stats.items()):
        print(f"  {status}: {n} 天")
    print(f"文件保存于: {out_dir}")


if __name__ == "__main__":
    main()
