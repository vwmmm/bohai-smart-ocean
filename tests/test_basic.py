# -*- coding: utf-8 -*-
"""
基础测试：验证配置、预处理与特征工程的核心逻辑。

运行方式（项目根目录）：
    python -m pytest tests/ -v
或直接使用：
    python tests/test_basic.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config          # noqa: E402
import preprocess      # noqa: E402


def _synthetic_series() -> pd.Series:
    """构造 4 年合成海温序列（年周期 + 噪声），仅用于单元测试。"""
    idx = pd.date_range("2022-01-01", "2025-12-31", freq="D")
    doy = idx.day_of_year
    sst = 12 + 9 * np.sin(2 * np.pi * (doy - 200) / 365.25)
    rng = np.random.default_rng(0)
    return pd.Series(sst + rng.normal(0, 0.3, len(idx)), index=idx, name="sst")


def test_config_region_valid():
    assert config.LON_MIN < config.LON_MAX
    assert config.LAT_MIN < config.LAT_MAX
    assert "整个渤海" in config.SUB_REGIONS


def test_climatology_shape_and_range():
    series = _synthetic_series()
    clim = preprocess.compute_climatology(series, start_year=2022, end_year=2024)
    assert len(clim) == 366
    assert clim["clim_mean"].between(-2, 40).all()
    assert (clim["clim_std"] >= 0).all()


def test_attach_baseline_anomaly_near_zero():
    """合成数据的气候态应贴近真实年周期，近年距平绝对值应较小。"""
    series = _synthetic_series()
    clim = preprocess.compute_climatology(series, start_year=2022, end_year=2024)
    df = preprocess.attach_baseline(series["2025"], clim)
    assert df["anomaly"].abs().mean() < 1.0


def test_build_features_no_future_leak():
    """特征行 t 只能包含 t 及之前的信息：lag/rolling 首日必须为 NaN。"""
    series = _synthetic_series()
    feat = preprocess.build_features(series)
    assert np.isnan(feat["lag_1"].iloc[0])
    assert np.isnan(feat["roll_3"].iloc[1])
    assert np.isnan(feat["roll_7"].iloc[5])
    # 周期特征不应有缺失
    assert feat["doy_sin"].notnull().all()
    assert feat["doy_cos"].notnull().all()
    # 特征列齐全
    for col in ["lag_1", "roll_3", "roll_7", "diff_1", "doy_sin", "doy_cos"]:
        assert col in feat.columns


def test_mhw_detection_rules():
    """人工构造一段 6 天热浪 + 一段 3 天短热峰：只有 ≥5 天的应被识别。"""
    idx = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    doy = idx.day_of_year
    sst = 12 + 9 * np.sin(2 * np.pi * (doy - 200) / 365.25)
    series = pd.Series(sst, index=idx)
    # 2023 年 7 月：连续 6 天 +5℃（应识别）；2024 年 7 月：连续 3 天 +5℃（不应识别）
    series.loc["2023-07-10":"2023-07-15"] += 5.0
    series.loc["2024-07-10":"2024-07-12"] += 5.0

    import anomaly_detection as ad
    thresholds = ad.seasonal_thresholds(series, years=(2020, 2025))
    events = ad.detect_events(series, thresholds, "heat", min_duration=5)
    starts = [str(d) for d in events["开始日期"]]
    assert any(s.startswith("2023-07-1") for s in starts), f"未识别 6 天热浪: {starts}"
    assert not any(s.startswith("2024-07-1") for s in starts), f"误识别 3 天热峰: {starts}"


def test_no_event_on_plain_series():
    """纯周期序列不应产生任何事件（阈值基于其自身气候态）。"""
    idx = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    sst = 12 + 9 * np.sin(2 * np.pi * (idx.day_of_year - 200) / 365.25)
    series = pd.Series(sst, index=idx)
    import anomaly_detection as ad
    thresholds = ad.seasonal_thresholds(series, years=(2020, 2025))
    events = ad.detect_events(series, thresholds, "heat")
    assert len(events) == 0


def test_prediction_split_and_alignment():
    """预测模块：时间切分有序、目标对齐正确、无泄漏。"""
    series = _synthetic_series()
    import prediction
    X, y = prediction.build_supervised(series, horizon=1)
    # 目标对齐：y(t) 应等于 sst(t+1)
    t = X.index[10]
    assert abs(y.loc[t] - series.loc[t + pd.Timedelta(days=1)]) < 1e-9
    # 时间切分：训练集全部早于测试集
    X_tr, X_te, y_tr, y_te = prediction.time_split(X, y, test_days=60)
    assert X_tr.index.max() < X_te.index.min()
    assert len(X_te) == 60


def test_prediction_evaluate_runs():
    """小样本上完整跑通评估流程，指标为有限正数。"""
    series = _synthetic_series()
    import prediction
    results = prediction.evaluate(series, horizons=[1], test_days=60)
    m = results[1]["metrics"]
    assert (m["MAE (℃)"] > 0).all()
    assert (m["RMSE (℃)"] >= m["MAE (℃)"]).all()   # RMSE 恒不小于 MAE


def test_warning_levels():
    """预警规则：热浪期间应触发Ⅱ级以上，正常日应为正常/关注。"""
    import anomaly_detection as ad
    import warning
    idx = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    sst = 12 + 9 * np.sin(2 * np.pi * (idx.day_of_year - 200) / 365.25)
    series = pd.Series(sst, index=idx)
    series.loc["2023-07-10":"2023-07-20"] += 5.0     # 11 天强热浪 → Ⅰ级
    thresholds = ad.seasonal_thresholds(series, years=(2020, 2025))
    events = ad.detect_events(series, thresholds, "heat")

    stats = {"n_valid": 100, "n_warm": 0, "n_cold": 0,
             "max_anom": 5.0, "min_anom": 0.1, "mean_anom": 3.0, "mean_abs_z": 2.0}
    sub = pd.DataFrame({"区域": ["辽东湾"], "平均距平 (℃)": [4.0],
                        "显著偏暖格点": [10], "显著偏冷格点": [0]})
    forecast = pd.DataFrame({"预测日期": [pd.Timestamp("2023-07-21")],
                             "预测海温 (℃)": [26.0]})
    risk = warning.assess(series, events, stats, forecast, sub, "2023-07-15",
                          clim_years=(2020, 2025))
    assert risk.level[0] == "Ⅰ级", f"持续 11 天强热浪应触发Ⅰ级，实际 {risk.level}"
    assert risk.risk_type == "海洋热浪"


def test_pdf_report_generates():
    """PDF 报告能生成且非空。"""
    import report_generator
    import warning
    risk = warning.RiskAssessment(
        level=warning.LEVEL_OK, risk_type="正常",
        position="渤海中部（平均距平 +0.10 ℃）", degree="区域平均距平 +0.10 ℃",
        reasons=["测试依据"], suggestions=["测试建议"])
    context = {
        "report_date": "2026-11-03",
        "day_stats": {"n_valid": 162, "n_warm": 0, "n_cold": 0,
                      "mean_anom": 0.1, "max_anom": 0.5, "min_anom": -0.4},
        "risk": risk,
        "metrics_table": pd.DataFrame({"预测步长": ["未来第 1 天"], "模型": ["随机森林"],
                                       "MAE (℃)": [0.17], "RMSE (℃)": [0.21],
                                       "测试区间": ["2026-09-02 ~ 2026-10-31"]}),
        "forecast": pd.DataFrame({"预测日期": ["2026-11-02"], "预测海温 (℃)": [17.0],
                                  "预测步长": ["未来第 1 天"], "基准日": ["2026-11-01"]}),
        "events": pd.DataFrame({"类型": ["海洋热浪"], "开始日期": ["2026-08-24"],
                                "结束日期": ["2026-08-30"], "持续天数": [7],
                                "最大强度(℃)": [2.37]}),
        "region_name": "整个渤海",
    }
    pdf = report_generator.generate_report(context)
    assert len(pdf) > 3000
    assert pdf[:5] == b"%PDF-"


def test_mhw_gap_merging():
    """事件合并：两段超阈值间隔 ≤2 天应合并为一个事件；间隔 4 天不合并。"""
    import anomaly_detection as ad
    idx = pd.date_range("2020-01-01", "2025-12-31", freq="D")
    sst = 12 + 9 * np.sin(2 * np.pi * (idx.day_of_year - 200) / 365.25)
    series = pd.Series(sst, index=idx)
    # 2023 年 7 月：3 天热 + 间隔 2 天 + 3 天热 → 合并后 8 天，应识别
    series.loc["2023-07-10":"2023-07-12"] += 5.0
    series.loc["2023-07-15":"2023-07-17"] += 5.0
    # 2024 年 7 月：3 天热 + 间隔 4 天 + 3 天热 → 不合并，两段各 3 天 < 5，均不识别
    series.loc["2024-07-10":"2024-07-12"] += 5.0
    series.loc["2024-07-17":"2024-07-19"] += 5.0
    thresholds = ad.seasonal_thresholds(series, years=(2020, 2025))
    events = ad.detect_events(series, thresholds, "heat")
    starts = [str(d) for d in events["开始日期"]]
    assert any(s.startswith("2023-07-1") for s in starts), f"合并事件未识别: {starts}"
    assert not any(s.startswith("2024-07-1") for s in starts), f"间隔 4 天被误合并: {starts}"
    # 合并后事件持续天数应包含间隔（2023-07-10 ~ 2023-07-17 共 8 天）
    ev = events[events["开始日期"].astype(str).str.startswith("2023-07-1")].iloc[0]
    assert ev["持续天数"] == 8, f"合并后持续天数应为 8，实际 {ev['持续天数']}"


def test_rolling_backtest_no_leak():
    """滚动回测：各折测试区间互不重叠且严格递增，训练集均在测试集之前。"""
    import prediction
    series = _synthetic_series()
    bt, preds = prediction.rolling_backtest(series, horizon=1, n_folds=3, fold_days=60)
    assert len(bt) == 3
    assert len(preds) == 180
    intervals = []
    for _, row in bt.iterrows():
        a, b = str(row["测试区间"]).split(" ~ ")
        intervals.append((pd.Timestamp(a), pd.Timestamp(b)))
        # 训练集截止必须早于测试开始（行内已保证，此处验证区间有序不重叠）
    # 折次靠后的（第1折）测试区间应最新
    assert intervals[0][0] > intervals[1][0] > intervals[2][0]
    # 任意两折不重叠
    for i in range(len(intervals)):
        for j in range(i + 1, len(intervals)):
            assert intervals[i][1] < intervals[j][0] or intervals[j][1] < intervals[i][0]


def test_seasonal_breakdown():
    """分季节统计：覆盖四季、样本数守恒、指标为有限正数。"""
    import prediction
    idx = pd.date_range("2025-01-01", "2025-12-31", freq="D")
    preds = pd.DataFrame({
        "真实值": np.ones(len(idx)) * 10,
        "持续性基线": np.ones(len(idx)) * 10.5,
        "随机森林": np.ones(len(idx)) * 10.2,
        "折次": 1,
    }, index=idx)
    sb = prediction.seasonal_breakdown(preds)
    assert set(sb["季节"]) == {"春季", "夏季", "秋季", "冬季"}
    assert int(sb["样本数"].sum()) == 365
    row = sb[sb["季节"] == "夏季"].iloc[0]
    assert row["随机森林 MAE (℃)"] < row["基线 MAE (℃)"]
    assert row["改善率 (%)"] > 0


if __name__ == "__main__":
    for name, fn in sorted(
            {k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
        print(f"PASS  {name}")
    print("\n全部测试通过 ✅")
