# -*- coding: utf-8 -*-
"""
渤海智瞳 —— 基于海洋遥感数据与轻量化机器学习的渤海海温异常监测及智能预警系统
==============================================================================

Streamlit 应用入口。

运行方式：
    streamlit run app.py

模块进度：
- 第一阶段：项目骨架 + 首页 + 侧边栏导航 ✅
- 第二阶段：NOAA 数据接入 + 海洋环境监测中心 ✅
"""

from datetime import date

import pandas as pd
import streamlit as st

import config
import anomaly_detection
import data_loader
import prediction
import preprocess
import report_generator
import visualization
import warning

# ---------------------------------------------------------------------------
# 页面基本设置（必须是最先调用的 st 命令）
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",                       # 宽屏布局，适配 1920×1080 演示
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 自定义 CSS：深海蓝主题、蓝青渐变标题、数据卡片
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
/* 顶部渐变标题横幅（压缩高度，突出内容区） */
.main-header {
    background: linear-gradient(90deg, #0D47A1 0%, #0288D1 55%, #00ACC1 100%);
    padding: 0.65rem 1.4rem;
    border-radius: 10px;
    margin-bottom: 0.7rem;
}
.main-header h1 {
    color: #FFFFFF;
    font-size: 1.35rem;
    margin: 0;
    letter-spacing: 1px;
}
.main-header p {
    color: #B3E5FC;
    font-size: 0.8rem;
    margin: 0.15rem 0 0 0;
}
/* 数据卡片 */
.metric-card {
    background: linear-gradient(145deg, #12294D, #0D213F);
    border: 1px solid #1F4E79;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .label { color: #90CAF9; font-size: 0.85rem; }
.metric-card .value { color: #FFFFFF; font-size: 1.6rem; font-weight: 600; }
.metric-card .sub   { color: #78909C; font-size: 0.75rem; }
/* 模块占位提示框 */
.module-placeholder {
    background: rgba(2, 136, 209, 0.08);
    border: 1px dashed #29B6F6;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    color: #90CAF9;
}
/* 隐藏英文页脚（Made with Streamlit） */
footer {display: none !important;}
</style>
"""


# ---------------------------------------------------------------------------
# 数据加载（带缓存）：首次加载需要数秒，之后交互秒开
# ---------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def load_sst_dataset():
    """加载全部渤海海温数据并应用海域掩膜，返回 (sst, sea_mask)。"""
    sst = data_loader.load_sst()
    mask = data_loader.build_sea_mask(sst)
    return data_loader.apply_sea_mask(sst, mask), mask


@st.cache_data(ttl=600, show_spinner=False)
def load_region_data(_sst):
    """计算各区域时间序列与气候态基准（结果与数据内容绑定缓存）。"""
    return preprocess.prepare_monitoring_data(_sst)


@st.cache_data(ttl=600, show_spinner=False)
def run_prediction_eval(region_name: str):
    """对指定区域执行完整预测评估 + 滚动回测 + 分季节统计（按内容缓存）。"""
    sst, _ = load_sst_dataset()
    series = preprocess.region_timeseries(sst, config.SUB_REGIONS[region_name])
    results = prediction.evaluate(series)
    backtests, seasonal = {}, {}
    for h in config.PREDICT_HORIZONS:
        folds_df, preds_df = prediction.rolling_backtest(series, h)
        backtests[h] = {"folds": folds_df, "preds": preds_df}
        seasonal[h] = prediction.seasonal_breakdown(preds_df)
    future = prediction.predict_future(series)
    return series, results, backtests, seasonal, future


def render_prediction() -> None:
    """模块三：海温智能预测中心（真实模型、滚动回测与残差分析）。"""
    st.subheader("🤖 海温智能预测中心")
    if not require_data():
        return

    region_name = st.selectbox("选择预测区域", options=list(config.SUB_REGIONS.keys()))
    with st.spinner("正在训练模型、滚动回测（首次约 1 分钟，之后缓存秒开）…"):
        series, results, backtests, seasonal, future = run_prediction_eval(region_name)

    st.caption(
        f"预测对象：**{region_name}**逐日平均海温（{series.index[0].date()} ~ "
        f"{series.index[-1].date()}，共 {len(series)} 天）｜"
        "特征：前 1 天海温 / 前 3 天均值 / 前 7 天均值 / 日变化趋势 / 年周期 sin-cos｜"
        "训练集与测试集按时间顺序切分，**无未来数据泄漏**"
    )

    # ---- 模型对比总表 -------------------------------------------------------
    st.markdown("#### 📊 测试集真实评估结果")
    rows = []
    for h in config.PREDICT_HORIZONS:
        r = results[h]
        for model_name in r["metrics"].index:
            row = {
                "预测步长": f"未来第 {h} 天",
                "模型": model_name,
                "MAE (℃)": r["metrics"].loc[model_name, "MAE (℃)"],
                "RMSE (℃)": r["metrics"].loc[model_name, "RMSE (℃)"],
                "训练样本": r["n_train"], "测试样本": r["n_test"],
                "测试区间": f"{r['test_start']} ~ {r['test_end']}",
            }
            if model_name == "随机森林":
                row["相对基线 MAE 改善率"] = f"{r['improvement_pct']:+.1f}%"
            else:
                row["相对基线 MAE 改善率"] = "—"
            rows.append(row)
    metrics_table = pd.DataFrame(rows)

    # 标注每个步长的胜出者（如实，不预设随机森林更好）
    def _winner(h):
        m = results[h]["metrics"]
        return "随机森林" if m.loc["随机森林", "MAE (℃)"] < m.loc["持续性基线", "MAE (℃)"] \
            else "持续性基线"
    metrics_table["更优"] = metrics_table.apply(
        lambda r: "✅" if r["模型"] == _winner(int(r["预测步长"][4])) else "", axis=1)
    st.dataframe(metrics_table, width='stretch', hide_index=True)

    # ---- 对比曲线 + 特征重要性 + 残差分布 -----------------------------------
    h_sel = st.select_slider("查看预测步长", options=config.PREDICT_HORIZONS,
                             value=1, format_func=lambda h: f"未来第 {h} 天")
    r = results[h_sel]
    chart_col, imp_col = st.columns([3, 2])
    with chart_col:
        fig = visualization.prediction_chart(
            r["test"], title=f"测试集对比 · 未来第 {h_sel} 天"
            f"（训练 {r['train_start']}~{r['train_end']}，{r['n_train']} 天）")
        st.plotly_chart(fig, width='stretch', config=visualization.PLOTLY_CONFIG)
    with imp_col:
        fig_imp = visualization.feature_importance_chart(r["feature_importance"])
        st.plotly_chart(fig_imp, width='stretch', config=visualization.PLOTLY_CONFIG)
        fig_res = visualization.residual_chart(
            r["test"], title=f"残差分布 · 未来第 {h_sel} 天")
        st.plotly_chart(fig_res, width='stretch', config=visualization.PLOTLY_CONFIG)

    # ---- 滚动时间窗口回测（稳定性检验） --------------------------------------
    st.markdown("#### 🔁 滚动时间窗口回测（模型稳定性检验）")
    bt = backtests[h_sel]["folds"]
    st.dataframe(bt, width='stretch', hide_index=True)
    if not bt.empty:
        imp = bt["改善率 (%)"]
        st.caption(
            f"未来第 {h_sel} 天：{len(bt)} 折回测改善率均值 "
            f"**{imp.mean():+.1f}%**（最差 {imp.min():+.1f}%，最好 {imp.max():+.1f}%）。"
            "每一折只用该折测试区间之前的数据训练——若各折改善率均为正且稳定，"
            "说明模型优势不依赖某一段测试期。"
        )

    # ---- 分季节误差（基于回测 360 天，覆盖四季） ------------------------------
    st.markdown("#### 🍂 分季节预测误差（滚动回测汇总）")
    sb = seasonal[h_sel]
    if sb.empty:
        st.info("回测预测为空，无法分季节统计。")
    else:
        st.dataframe(sb, width='stretch', hide_index=True)
        valid = sb[sb["样本数"] > 0]
        if not valid.empty:
            worst = valid.loc[valid["改善率 (%)"].idxmin()]
            best = valid.loc[valid["改善率 (%)"].idxmax()]
            note_parts = [
                f"**优势**：{best['季节']}改善率最高（{best['改善率 (%)']:+.1f}%）",
                f"**不足**：{worst['季节']}改善率最低（{worst['改善率 (%)']:+.1f}%）"
                + ("，该季节随机森林**劣于基线**" if worst["改善率 (%)"] < 0 else ""),
            ]
            st.markdown("客观分析：" + "；".join(note_parts) + "。")
        st.caption(
            "分季节统计基于滚动回测的全部 6 折（共 360 天预测值），"
            "覆盖春夏秋冬四季；单一 60 天测试集只覆盖 1–2 个季节，不足以评估季节差异。"
            "冬季样本对应 12 月至次年 2 月。")

    # ---- 实验结论（根据真实结果自动生成） ------------------------------------
    st.markdown("#### 🧾 实验结论（根据真实结果自动生成）")
    for c in prediction.summarize_results(results, backtests, seasonal,
                                          horizon=h_sel):
        st.markdown(f"- {c}")

    # ---- 未来 1~3 天预测 ----------------------------------------------------
    st.markdown("#### 🔮 未来 1~3 天海温预测")
    st.dataframe(future, width='stretch', hide_index=True)
    st.caption(
        f"模型：随机森林回归（每个步长单独训练）｜训练数据截止：{series.index[-1].date()}｜"
        "预测基准日即最新数据日期，模型文件已保存至 `models/` 目录"
    )

    with st.expander("模型与评估方法说明"):
        st.markdown(
            """
- **持续性基线**：预测值 = 当天海温。海温短期变化缓慢，该基线很强，
  是衡量机器学习模型价值的底线。
- **随机森林回归**：300 棵树、`min_samples_leaf=2`、固定随机种子，
  参数事先设定，不针对测试集调参。
- **防泄漏**：特征只含当天及历史信息；训练/测试按时间顺序切分（最后 60 天为测试集）；
  每个步长单独训练直接多步模型，不做迭代递推；目标值 y(t)=sst(t+h)
  的对齐正确性有单元测试保障。
- **滚动回测**：6 折 × 每折 60 天（共 360 天，覆盖四季），每折仅用其之前的数据训练，
  检验模型优势的时间稳定性与季节差异。
- **残差分布**：预测值−真实值的直方图，理想情况集中在 0 附近且比基线窄。
- 所有数字均为真实计算结果，未做任何修饰；若某处基线更优，界面会如实标注。
"""
        )


@st.cache_data(ttl=600, show_spinner=False)
def load_anomaly_bundle(_sst):
    """异常识别数据包：网格气候态、逐日距平、区域事件列表。

    网格气候态有磁盘缓存，首次计算约 10 秒，之后秒开。
    """
    clim = anomaly_detection.load_grid_climatology(_sst)
    anom = anomaly_detection.compute_anomaly(_sst, clim)
    series = preprocess.region_timeseries(_sst, config.SUB_REGIONS["整个渤海"])
    events, thresholds = anomaly_detection.detect_all_events(series)
    return clim, anom, series, events, thresholds


def show_baseline_error(err: Exception) -> None:
    """基准期数据不足时的统一友好提示（功能降级而非整页崩溃）。

    附带实时下载进度，让用户明确知道系统在工作而不是故障。
    """
    files = data_loader.list_raw_files()
    progress = ""
    if files:
        span = (files[-1][0] - files[0][0]).days + 1
        progress = (f"\n\n📥 数据下载进度：**{len(files)} / {span} 天**"
                    f"（{len(files) / span:.0%}），下载完成后刷新页面即可自动恢复，"
                    "其余功能不受影响。")
    st.warning(f"⏳ **气候态基准期数据尚未就绪**\n\n{err}{progress}")


def build_metrics_table(results: dict) -> pd.DataFrame:
    """把 prediction.evaluate 的结果汇总为对比表（预测页与预警报告共用）。"""
    rows = []
    for h in config.PREDICT_HORIZONS:
        r = results[h]
        for model_name in r["metrics"].index:
            rows.append({
                "预测步长": f"未来第 {h} 天",
                "模型": model_name,
                "MAE (℃)": r["metrics"].loc[model_name, "MAE (℃)"],
                "RMSE (℃)": r["metrics"].loc[model_name, "RMSE (℃)"],
                "训练样本": r["n_train"], "测试样本": r["n_test"],
                "测试区间": f"{r['test_start']} ~ {r['test_end']}",
                "改善率": (f"{r['improvement_pct']:+.1f}%"
                           if model_name == "随机森林" else "—"),
            })
    return pd.DataFrame(rows)


def render_warning() -> None:
    """模块四：海洋智能预警中心（六段式布局：状态/事件/定位/预测/历史/报告）。"""
    st.subheader("⚠️ 海洋智能预警中心")
    if not require_data():
        return

    with st.spinner("正在汇总异常识别与预测结果…"):
        sst, _ = load_sst_dataset()
        try:
            clim, anom, series, events, thresholds = load_anomaly_bundle(sst)
        except RuntimeError as err:
            show_baseline_error(err)
            return
        _, results, _, _, future = run_prediction_eval("整个渤海")

    last_date = pd.DatetimeIndex(sst.time.values).date[-1]
    day_stats = anomaly_detection.classify_day(anom.isel(time=-1))
    sub_status = warning.subregion_status(anom, last_date)
    risk = warning.assess(series, events, day_stats, future, sub_status, last_date)

    # ---- 第一部分：当前综合风险状态 ------------------------------------------
    st.markdown("#### 一、当前综合风险状态")
    lv_name, lv_label, lv_color = risk.level
    st.markdown(
        f'<div style="background:{lv_color};border-radius:12px;padding:0.9rem 1.4rem;'
        f'margin-bottom:0.8rem">'
        f'<span style="font-size:1.35rem;font-weight:700;color:#fff">'
        f'{lv_name} · {lv_label}</span>'
        f'<span style="float:right;color:#fff">数据日期 {last_date}</span>'
        f'<div style="color:#fff;opacity:.92;margin-top:.2rem">'
        f'风险类型：{risk.risk_type}　｜　主要影响区域：{risk.position}</div></div>',
        unsafe_allow_html=True,
    )
    st.caption(f"⚠️ {warning.DISCLAIMER}（触发规则见页面底部）")

    # ---- 第二部分：当前异常事件 ----------------------------------------------
    st.markdown("#### 二、当前异常事件")
    active = anomaly_detection.event_on_date(events, last_date)
    if active.empty:
        st.success(f"截至 {last_date}，区域平均海温未处于海洋热浪/冷事件之中。")
    else:
        ev = active.iloc[0]
        c = st.columns(5)
        for col, (label, value, sub) in zip(c, [
            ("事件类型", ev["类型"], "Hobday 判据"),
            ("起始日期", str(ev["开始日期"]), "超过阈值首日"),
            ("持续时间", f"{ev['持续天数']} 天", "含 ≤2 天合并间隔"),
            ("最大强度", f"{ev['最大强度(℃)']:+.2f} ℃", "相对气候态均值"),
            ("峰值日期", str(ev["峰值日期"]), f"最高海温 {ev['期间最高海温(℃)']} ℃"),
        ]):
            col.markdown(
                f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value" style="font-size:1.1rem">{value}</div>'
                f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)
        st.caption("注意：事件基于**整个渤海区域平均海温**判定，"
                   "不代表所有网格都达到热浪标准；网格级情况见异常识别中心。")

    # ---- 第三部分：子区域风险定位 --------------------------------------------
    st.markdown("#### 三、子区域风险定位")
    st.dataframe(sub_status, width='stretch', hide_index=True)
    st.caption("按平均距平绝对值排序；显著格点阈值为标准化距平 |z| ≥ "
               f"{config.ANOMALY_STD_THRESHOLD}（单日格点级判据）。")

    # ---- 第四部分：未来风险预测 ----------------------------------------------
    st.markdown("#### 四、未来风险预测（1~3 天）")
    stat = anomaly_detection.seasonal_stat(series)
    th = anomaly_detection.thresholds_for_dates(
        stat, pd.DatetimeIndex(future["预测日期"]))
    fc = future.copy()
    fc["当日热浪阈值 p90 (℃)"] = th["p90"].round(2).values
    fc["超出阈值"] = ["是 ⚠️" if p >= t else "否"
                     for p, t in zip(fc["预测海温 (℃)"], th["p90"])]
    fc = fc[["预测日期", "预测海温 (℃)", "当日热浪阈值 p90 (℃)", "超出阈值", "预测步长"]]
    st.dataframe(fc, width='stretch', hide_index=True)
    rmse_h1 = results[1]["metrics"].loc["随机森林", "RMSE (℃)"]
    fc_risks = warning.forecast_risk(series, future)
    if fc_risks:
        for r in fc_risks:
            st.warning(r)
    else:
        st.success("未来 1~3 天预测均未突破季节第 90 百分位阈值。")
    st.caption(
        f"不确定性说明：随机森林为确定性点预测，参考测试集 RMSE ±{rmse_h1} ℃"
        "（未来第 1 天）评估置信范围；预测风险仅为统计提示，"
        "不构成确定性养殖处置指令。"
    )

    # ---- 第五部分：历史异常事件 ----------------------------------------------
    st.markdown("#### 五、历史异常事件")
    f1, f2 = st.columns(2)
    with f1:
        type_filter = st.multiselect(
            "事件类型筛选", ["海洋热浪", "海洋冷事件"],
            default=["海洋热浪", "海洋冷事件"])
    with f2:
        years_all = sorted({d.year for d in events["开始日期"]})
        year_range = st.slider("开始年份范围", min(years_all), max(years_all),
                               (max(years_all) - 4, max(years_all)))
    filtered = events[
        events["类型"].isin(type_filter)
        & events["开始日期"].apply(lambda d: year_range[0] <= d.year <= year_range[1])
    ]
    # 时间线与表格共用同一筛选结果，数量严格一致
    fig_tl = visualization.event_timeline_chart(
        filtered, years=year_range[1] - year_range[0] + 1)
    if fig_tl.data:
        st.plotly_chart(fig_tl, width='stretch', config=visualization.PLOTLY_CONFIG)
    st.caption(f"筛选结果：{len(filtered)} / {len(events)} 个事件"
               f"（判据：季节第 90/10 百分位 + 连续 ≥{config.MHW_MIN_DURATION} 天，"
               f"间隔 ≤{config.MHW_MAX_GAP} 天合并）")
    st.dataframe(filtered, width='stretch', hide_index=True, height=280)

    if not filtered.empty:
        options = [f"{r['开始日期']} {r['类型']}（{r['持续天数']} 天）"
                   for _, r in filtered.iterrows()]
        pick = st.selectbox("点击查看事件详情", options=options)
        ev = filtered.iloc[options.index(pick)]
        c1, c2, c3, c4 = st.columns(4)
        for col, (label, value) in zip(
                (c1, c2, c3, c4),
                [("事件类型", ev["类型"]),
                 ("持续天数", f"{ev['持续天数']} 天"),
                 ("最大强度", f"{ev['最大强度(℃)']:+.2f} ℃"),
                 ("期间最高海温", f"{ev['期间最高海温(℃)']} ℃")]):
            col.markdown(
                f'<div class="metric-card"><div class="label">{label}</div>'
                f'<div class="value" style="font-size:1.15rem">{value}</div>'
                f'<div class="sub">{ev["开始日期"]} ~ {ev["结束日期"]}</div></div>',
                unsafe_allow_html=True,
            )

    # ---- 第六部分：报告导出 ---------------------------------------------------
    st.markdown("#### 六、分析报告导出")
    st.caption("PDF 内容与当前页面使用同一组计算结果（同一 RiskAssessment 对象），数据一致。")
    context = {
        "report_date": date.today(),
        "day_stats": day_stats,
        "risk": risk,
        "metrics_table": build_metrics_table(results),
        "forecast": future,
        "events": events,
        "region_name": "整个渤海",
    }
    if st.button("生成 PDF 分析报告"):
        with st.spinner("正在生成 PDF…"):
            pdf_bytes = report_generator.generate_report(context)
        st.download_button(
            "⬇️ 下载 PDF 报告", pdf_bytes,
            file_name=f"渤海海温预警分析报告_{last_date}.pdf",
            mime="application/pdf",
        )

    # ---- 规则说明（折叠） -----------------------------------------------------
    with st.expander("预警规则说明（全部公开、可复现）"):
        st.markdown(
            r"""
| 等级 | 触发规则（满足其一） |
| --- | --- |
| Ⅰ级 严重预警 | 海洋热浪/冷事件持续 ≥10 天，或事件最大强度 ≥3℃ |
| Ⅱ级 警报 | 处于海洋热浪/冷事件之中（季节第 90/10 百分位 + 连续 ≥5 天） |
| Ⅲ级 预警 | ≥30% 海域标准化距平 \|z\|≥2，或区域平均距平 ≥1.5℃ |
| Ⅳ级 趋势关注 | 存在显著异常格点；或平均距平 ≥0.5℃；或预测未来 3 天突破第 90/10 百分位 |
| 正常 | 以上均不满足 |

概念区分：**趋势关注**（平均距平 ≥0.5℃）≠ **统计显著异常**（格点 \|z\|≥2）
≠ **海洋热浪事件**（季节第 90 百分位 + 连续 ≥5 天）。
空间尺度区分：格点级（单日）→ 子区域（平均距平）→ 整个渤海（区域事件）。
风险位置取平均距平绝对值最大的子区域；异常类型由事件识别结果或距平符号决定。
"""
        )
    st.caption(f"⚠️ {warning.DISCLAIMER}")


def render_header() -> None:
    """渲染顶部标题栏：系统名称、当前数据日期、数据来源。"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="main-header">
            <h1>🌊 渤海智瞳 · 海温异常监测及智能预警系统</h1>
            <p>2026 年辽宁省大学生智慧海洋科技创新大赛 · 海洋智能监测赛道</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # 顶部信息条：数据日期 / 数据来源 / 研究区域
    status = data_loader.get_data_status()
    date_text = (f"{status['last_date']}（共 {status['n_files']} 天）"
                 if status["n_files"] else "**暂无数据**")
    c1, c2, c3 = st.columns(3)
    c1.caption(f"📅 最新数据日期：{date_text}")
    c2.caption(f"🛰️ 数据来源：{config.DATA_SOURCE_NAME}")
    c3.caption(f"📍 研究区域：{config.REGION_NAME} "
               f"({config.LON_MIN:.0f}°E–{config.LON_MAX:.0f}°E, "
               f"{config.LAT_MIN:.0f}°N–{config.LAT_MAX:.0f}°N)")


def require_data() -> bool:
    """检查数据是否就绪；未就绪时给出明确提示并返回 False。"""
    status = data_loader.get_data_status()
    if status["n_files"] == 0:
        st.error("⚠️ 未检测到海温数据，无法展示该模块。")
        st.markdown(
            "请先下载数据（二选一）：\n"
            "1. 在项目目录运行：`python download_data.py`（自动下载 2020 年至今）\n"
            "2. 在能上网的电脑下载后，将 `data/raw/` 目录拷贝到本项目对应位置\n\n"
            f"数据源：[{config.DATA_SOURCE_NAME}]({config.DATA_SOURCE_URL})"
        )
        return False
    return True


def render_sidebar() -> str:
    """渲染左侧导航，返回当前选中的功能模块名称。

    侧边栏只放导航与身份信息；数据来源在顶部信息条展示，不在此重复。
    """
    with st.sidebar:
        st.markdown("## 🌊 渤海智瞳")
        st.markdown("**功能导航**")
        page = st.radio(
            label="选择功能模块",
            options=[
                "🏠 系统首页",
                "🗺️ 海洋环境监测中心",
                "🌡️ 海温异常识别中心",
                "🤖 海温智能预测中心",
                "⚠️ 海洋智能预警中心",
            ],
            label_visibility="collapsed",
            key="main_nav",                # 固定 key，保证重跑时组件身份稳定
        )
        st.divider()
        st.caption("开发团队：渤海智瞳")
        st.caption("版本：v1.0")
    return page


@st.cache_data(ttl=600, show_spinner=False)
def run_home_forecast():
    """首页驾驶舱轻量数据包：区域序列 + 气候态 + 未来预测 + 评估指标。"""
    sst, _ = load_sst_dataset()
    series = preprocess.region_timeseries(sst, config.SUB_REGIONS["整个渤海"])
    stat = anomaly_detection.seasonal_stat(series)   # doy 索引的气候态（含基准均值）
    results = prediction.evaluate(series)
    future = prediction.predict_future(series)
    return series, stat, results, future


def render_home() -> None:
    """系统首页：海洋智能监测驾驶舱。

    布局：核心指标 → 海洋地图（海温/距平切换）+ 趋势预测 →
    近期事件 + 核心成果 → 项目说明（折叠）。
    """
    status = data_loader.get_data_status()
    if status["n_files"] == 0:
        st.info("尚未接入数据。请先运行 `python download_data.py` 下载数据，"
                "之后此处将显示监测驾驶舱。")
        render_home_about()
        return

    with st.spinner("正在加载监测驾驶舱（首次约 30 秒，之后秒开）…"):
        sst, _ = load_sst_dataset()
        try:
            clim, anom, series, events, _ = load_anomaly_bundle(sst)
            _, stat, results, future = run_home_forecast()
            ready = True
        except RuntimeError as err:
            show_baseline_error(err)
            ready = False
    if not ready:
        return

    last_date = pd.DatetimeIndex(sst.time.values).date[-1]
    day_stats = anomaly_detection.classify_day(anom.isel(time=-1))
    sub_status = warning.subregion_status(anom, last_date)
    empty_fc = pd.DataFrame({"预测日期": [], "预测海温 (℃)": []})
    risk = warning.assess(series, events, day_stats, empty_fc, sub_status, last_date)
    day_sst = sst.isel(time=-1)

    # ---- 第一部分：核心指标 -------------------------------------------------
    lv_name, lv_label, lv_color = risk.level
    cols = st.columns(4)
    cards = [
        ("最新平均海温", f"{float(day_sst.mean()):.2f} ℃", str(last_date)),
        ("当前海温距平", f"{day_stats['mean_anom']:+.2f} ℃",
         f"基准 {config.CLIM_LABEL}"),
        ("显著异常格点", f"{day_stats['n_warm'] + day_stats['n_cold']} 个",
         f"偏暖 {day_stats['n_warm']} / 偏冷 {day_stats['n_cold']}（|z|≥2）"),
        ("综合风险状态", f"{lv_name} {lv_label}", risk.risk_type),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    # ---- 第二部分：海洋地图 + 海温趋势 --------------------------------------
    st.markdown("")
    map_col, trend_col = st.columns([3, 2])
    with map_col:
        view = st.radio("地图视图", ["海温分布", "海温距平"],
                        horizontal=True, key="home_map_view")
        if view == "海温分布":
            fig_map = visualization.sst_map(day_sst, title=f"渤海海表温度分布 · {last_date}")
        else:
            fig_map = visualization.anomaly_map(
                anom.isel(time=-1)["anomaly"], title=f"渤海海温距平分布 · {last_date}")
        st.plotly_chart(fig_map, width='stretch', config=visualization.PLOTLY_CONFIG)
    with trend_col:
        fig_trend = visualization.sst_trend_forecast_chart(
            series, stat, future,
            title="整个渤海：近一年海温 + 基准 + 预测")
        st.plotly_chart(fig_trend, width='stretch', config=visualization.PLOTLY_CONFIG)

    # ---- 第三部分：近期异常事件 + 项目核心成果 ------------------------------
    ev_col, res_col = st.columns([3, 2])
    with ev_col:
        st.markdown("#### 近期异常事件")
        if events.empty:
            st.info("研究时段内未识别到达到标准的事件。")
        else:
            recent = events.head(5)[["类型", "开始日期", "结束日期",
                                     "持续天数", "最大强度(℃)"]]
            st.dataframe(recent, width='stretch', hide_index=True)
            st.caption(f"共 {len(events)} 个事件（Hobday 判据，基准 {config.CLIM_LABEL}），"
                       "完整列表见异常识别中心。")
    with res_col:
        st.markdown("#### 项目核心成果")
        m1 = results[1]["metrics"]
        st.markdown(
            f"""
- **数据覆盖**：{status['n_files']} 天逐日海温（{status['first_date']} ~
  {status['last_date']}），NOAA OISST v2.1 官方数据
- **预测模型**：随机森林未来 1 天 MAE {m1.loc['随机森林', 'MAE (℃)']} ℃，
  较持续性基线改善 {results[1]['improvement_pct']:+.1f}%（测试集真实结果）
- **核心算法**：Hobday 海洋热浪识别（季节 p90 + 连续 5 天）、
  标准化距平、滚动回测验证、规则预警引擎
- **数据来源**：[{config.DATA_SOURCE_NAME}]({config.DATA_SOURCE_URL})
"""
        )

    # ---- 项目说明（折叠） ---------------------------------------------------
    with st.expander("📖 项目说明（简介 / 技术架构 / 创新点 / 开发路线）"):
        render_home_about()


def render_home_about() -> None:
    """项目说明内容：简介、技术架构、创新点、开发路线（供首页折叠区与无数据时展示）。"""
    st.markdown(
        "本系统面向**渤海海域**，基于 NOAA OISST v2.1 卫星遥感海表温度数据，"
        "构建集 **海温监测 → 异常识别 → 智能预测 → 风险预警** 于一体的"
        "轻量化智慧海洋平台，为海洋养殖防灾、海洋生态保护等场景提供数据支撑。"
    )
    st.markdown("**技术架构**")
    st.markdown(
        """
| 层次 | 技术选型 | 说明 |
| --- | --- | --- |
| 数据层 | NOAA OISST v2.1 (NetCDF) | 0.25° 分辨率逐日海表温度，Azure / AWS S3 / NCEI 三通道下载 |
| 处理层 | xarray / Pandas / NumPy | 数据读取、海域掩膜、气候态基准、特征工程 |
| 算法层 | Scikit-learn / 统计规则 | 标准化距平、Hobday 海洋热浪识别、随机森林预测 |
| 展示层 | Streamlit / Plotly | 深海蓝主题交互式 Web 监测平台 |
| 输出层 | ReportLab | PDF 分析报告导出 |
"""
    )
    st.markdown("**项目创新点**")
    st.markdown(
        """
1. **真实数据全链路**：13000+ 天 NOAA OISST 数据；三通道下载，断点续传，全部可复现。
2. **国际标准异常识别**：海洋热浪判定严格遵循 Hobday et al. (2016)
   （季节第 90 百分位 + 连续 ≥5 天 + 间隔 ≤2 天合并），WMO 30 年气候基准。
3. **诚实的机器学习**：基线对比、时间顺序切分、滚动回测、残差分析，
   所有指标为真实测试结果，模型不占优的折如实展示。
4. **规则透明的预警**：Ⅰ~Ⅳ级风险全部来自公开统计规则（项目内部规则，
   非官方标准），每条判定依据附带具体数值。
5. **轻量化部署**：纯 CPU、无数据库、无付费 API，普通笔记本即可离线演示。
"""
    )
    st.markdown("**开发路线（六阶段，已全部完成）**")
    st.markdown(
        "项目骨架 → 数据接入与可视化 → 异常识别 → 机器学习预测 → "
        "智能预警与报告 → UI 优化与测试，全部完成并持续增强中。"
    )


def render_monitoring() -> None:
    """模块一：海洋环境监测中心（第二阶段，真实数据）。"""
    st.subheader("🗺️ 海洋环境监测中心")
    if not require_data():
        return

    with st.spinner("正在加载海温数据（首次加载约需数秒）…"):
        sst, _ = load_sst_dataset()
        try:
            region_data = load_region_data(sst)
        except RuntimeError as err:
            # 基准期数据不足：降级为无基准线模式（地图与序列仍可用）
            show_baseline_error(err)
            region_data = {
                name: {"series": preprocess.region_timeseries(sst, box),
                       "clim": None}
                for name, box in config.SUB_REGIONS.items()
            }

    # ---- 顶部控制条：日期 + 区域 -------------------------------------------
    available_dates = list(pd.DatetimeIndex(sst.time.values).date)
    c1, c2 = st.columns([2, 2])
    with c1:
        selected_date = st.selectbox(
            "选择日期", options=list(reversed(available_dates)), index=0)
    with c2:
        region_name = st.selectbox("选择区域", options=list(config.SUB_REGIONS.keys()))

    day_sst = sst.sel(time=str(selected_date))
    clim = region_data["整个渤海"]["clim"]

    # ---- 核心指标卡片 -------------------------------------------------------
    mean_v = float(day_sst.mean())
    max_v = float(day_sst.max())
    min_v = float(day_sst.min())
    if clim is not None:
        baseline = float(clim.loc[pd.Timestamp(selected_date).day_of_year, "clim_mean"])
        delta = mean_v - baseline
        mean_sub = f"较历史同期 {delta:+.2f} ℃"
        baseline_card = (f"{baseline:.2f} ℃", f"{config.CLIM_LABEL} 气候态均值")
    else:
        mean_sub = "基准期数据下载中"
        baseline_card = ("—", "基准期数据下载中")
    cols = st.columns(4)
    cards = [
        ("平均海温", f"{mean_v:.2f} ℃", mean_sub),
        ("最高海温", f"{max_v:.2f} ℃", "海域内格点最大值"),
        ("最低海温", f"{min_v:.2f} ℃", "海域内格点最小值"),
        ("历史同期基准", baseline_card[0], baseline_card[1]),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(
            f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )

    # ---- 海温分布图 + 区域历史曲线 -----------------------------------------
    st.markdown("")
    map_col, chart_col = st.columns([3, 2])
    with map_col:
        fig_map = visualization.sst_map(
            day_sst, title=f"渤海海表温度分布 · {selected_date}")
        # on_select="rerun"：点击地图格点捕获坐标（Streamlit 原生能力，无额外依赖）
        map_event = st.plotly_chart(
            fig_map, width='stretch', config=visualization.PLOTLY_CONFIG,
            on_select="rerun", selection_mode="points", key="mon_map_click")
    with chart_col:
        window = st.radio("历史曲线时间范围", options=list(visualization.HIST_WINDOWS),
                          index=1, horizontal=True, key="mon_hist_window")
        rd = region_data[region_name]
        fig_ts = visualization.sst_history_chart(
            rd["series"], rd["clim"], title=f"{region_name}平均海温", window=window)
        st.plotly_chart(fig_ts, width='stretch', config=visualization.PLOTLY_CONFIG)

    # ---- 格点详情查询（点击地图或手动选择） ----------------------------------
    st.markdown("#### 🔍 格点详情查询")
    st.caption("点击地图中的海洋格点自动填充坐标，也可手动选择；"
               "展示该格点当前海温、历史同期基准、距平、阈值状态与近 30 天曲线。")
    lats = [float(v) for v in sst.lat.values]
    lons = [float(v) for v in sst.lon.values]

    # 从地图点击事件中提取坐标（若无点击则用默认值）
    clicked_lat = clicked_lon = None
    points = getattr(getattr(map_event, "selection", None), "points", []) or []
    if points:
        clicked_lon, clicked_lat = float(points[-1]["x"]), float(points[-1]["y"])

    def _nearest_idx(vals, target):
        return min(range(len(vals)), key=lambda i: abs(vals[i] - target))

    c1, c2 = st.columns(2)
    with c1:
        sel_lon = st.selectbox("经度格点 (°E)", lons,
                               index=_nearest_idx(lons, clicked_lon) if clicked_lon else 10,
                               key="cell_lon")
    with c2:
        sel_lat = st.selectbox("纬度格点 (°N)", lats,
                               index=_nearest_idx(lats, clicked_lat) if clicked_lat else 8,
                               key="cell_lat")

    cell = sst.sel(lon=sel_lon, lat=sel_lat)
    cur_v = float(cell.sel(time=str(selected_date)))
    if pd.isna(cur_v):
        st.warning("该格点为陆地或缺测（海域掩膜外），请另选海洋格点。")
    else:
        try:
            gclim = anomaly_detection.load_grid_climatology(sst)
        except RuntimeError as err:
            show_baseline_error(err)
            gclim = None
        if gclim is not None:
            doy = pd.Timestamp(selected_date).day_of_year
            base_v = float(gclim["clim_mean"].sel(lon=sel_lon, lat=sel_lat, doy=doy))
            std_v = float(gclim["clim_std"].sel(lon=sel_lon, lat=sel_lat, doy=doy))
            p90_v = float(gclim["clim_p90"].sel(lon=sel_lon, lat=sel_lat, doy=doy))
            anom_v = cur_v - base_v
            z_v = anom_v / std_v if std_v >= 0.1 else 0.0
            if abs(z_v) >= config.ANOMALY_STD_THRESHOLD:
                status = "显著偏暖" if z_v > 0 else "显著偏冷"
            elif cur_v >= p90_v:
                status = "达到热浪阈值（未含持续性判定）"
            else:
                status = "正常"
            c = st.columns(5)
            for col, (label, value, sub) in zip(c, [
                ("当前海温", f"{cur_v:.2f} ℃", str(selected_date)),
                ("同期基准", f"{base_v:.2f} ℃", f"{config.CLIM_LABEL} 气候态"),
                ("海温距平", f"{anom_v:+.2f} ℃", "当前 − 基准"),
                ("z 分数", f"{z_v:+.2f}", f"|z|≥{config.ANOMALY_STD_THRESHOLD} 为显著"),
                ("阈值状态", status, f"当日 p90 = {p90_v:.2f} ℃"),
            ]):
                col.markdown(
                    f'<div class="metric-card"><div class="label">{label}</div>'
                    f'<div class="value" style="font-size:1.1rem">{value}</div>'
                    f'<div class="sub">{sub}</div></div>', unsafe_allow_html=True)
            # 近 30 天格点曲线
            cell_30d = cell.sel(time=slice(
                str(pd.Timestamp(selected_date) - pd.Timedelta(days=29)),
                str(selected_date))).to_series()
            base_30d = pd.Series(
                [float(gclim["clim_mean"].sel(lon=sel_lon, lat=sel_lat, doy=d))
                 for d in cell_30d.index.day_of_year],
                index=cell_30d.index)
            st.plotly_chart(
                visualization.cell_trend_chart(
                    cell_30d, base_30d,
                    title=f"格点 ({sel_lon:.2f}°E, {sel_lat:.2f}°N) 近 30 天海温"),
                width='stretch', config=visualization.PLOTLY_CONFIG)
        else:
            st.info(f"该格点当前海温 {cur_v:.2f} ℃（基准数据就绪后显示距平与阈值状态）。")

    # ---- 数据来源与更新说明 -------------------------------------------------
    with st.expander("数据来源与处理说明"):
        st.markdown(
            f"""
- **数据源**：{config.DATA_SOURCE_NAME}（0.25° 分辨率逐日海表温度）
- **下载渠道**：AWS S3 NOAA 公开数据桶（`noaa-cdr-sea-surface-temp-optimum-interpolation-pds`），
  本地裁剪至渤海区域
- **数据范围**：{status_text()}
- **海域掩膜**：有效观测天数占比 ≥ 50% 的格点视为海洋，陆地已筛除
- **质量控制**：物理合理范围 -2 ~ 40 ℃，超范围值剔除
- **历史同期基准**：{config.CLIM_LABEL} 逐"年日"气候态（11 天滑动窗口）{_clim_note()}
"""
        )


def _clim_note() -> str:
    """基准期说明文字：达到 30 年标准 / 短基准期警示。"""
    if config.CLIM_IS_STANDARD:
        return "，达到 WMO 推荐的 30 年标准气候态"
    return ("，基准期短于 WMO 推荐的 30 年，属短基准期探索性识别，"
            "事件频率与阈值稳定性需谨慎解读")


def status_text() -> str:
    """数据覆盖范围的简要文字说明。"""
    s = data_loader.get_data_status()
    return f"{s['first_date']} 至 {s['last_date']}，共 {s['n_files']} 天" if s["n_files"] else "无"


def render_anomaly() -> None:
    """模块二：海温异常识别中心（第三阶段，真实算法）。"""
    st.subheader("🌡️ 海温异常识别中心")
    if not require_data():
        return

    with st.spinner("正在计算气候态与距平（首次约 10 秒）…"):
        sst, _ = load_sst_dataset()
        try:
            clim, anom, series, events, thresholds = load_anomaly_bundle(sst)
        except RuntimeError as err:
            show_baseline_error(err)
            return

    # ---- 日期选择 -----------------------------------------------------------
    available_dates = list(pd.DatetimeIndex(sst.time.values).date)
    selected_date = st.selectbox(
        "选择日期", options=list(reversed(available_dates)), index=0,
        key="anomaly_date")
    day_anom = anom.sel(time=str(selected_date))
    day_sst_grid = sst.sel(time=str(selected_date))
    stats = anomaly_detection.classify_day(day_anom)
    active = anomaly_detection.event_on_date(events, selected_date)

    # ---- 状态横幅：事件 > 显著格点 > 正常 -----------------------------------
    if not active.empty:
        ev = active.iloc[0]
        st.error(
            f"⚠️ **{ev['类型']}进行中**：{ev['开始日期']} 起已持续至该日"
            f"（共 {ev['持续天数']} 天），最大强度 {ev['最大强度(℃)']} ℃，"
            f"峰值日期 {ev['峰值日期']}。"
            "判定依据：海温连续 ≥5 天超过季节第 90 百分位阈值（Hobday et al., 2016）。"
            if ev["类型"] == "海洋热浪" else
            f"❄️ **{ev['类型']}进行中**：{ev['开始日期']} 起（共 {ev['持续天数']} 天），"
            f"最大强度 {ev['最大强度(℃)']} ℃。"
        )
    elif stats["n_warm"] + stats["n_cold"] > 0:
        st.warning(
            f"该日存在显著距平格点（标准化距平 |z| ≥ {config.ANOMALY_STD_THRESHOLD}）："
            f"偏暖 {stats['n_warm']} 个、偏冷 {stats['n_cold']} 个，"
            "但未达到海洋热浪/冷事件的持续时间标准。"
        )
    elif abs(stats["mean_anom"]) >= 0.5:
        # 与预警中心Ⅳ级"趋势关注"规则一致：有趋势但未达统计显著异常
        direction = "偏暖" if stats["mean_anom"] > 0 else "偏冷"
        st.info(
            f"该日无统计显著异常（所有格点 |z| < 2），"
            f"但区域平均距平 {stats['mean_anom']:+.2f} ℃ 达到趋势关注阈值（0.5 ℃），"
            f"对应预警中心 **Ⅳ级·趋势关注（{direction}趋势）**。"
            "趋势关注 ≠ 显著异常 ≠ 海洋热浪。"
        )
    else:
        st.success(
            f"该日全海域正常：所有格点标准化距平 |z| < 2，"
            f"区域平均距平 {stats['mean_anom']:+.2f} ℃ 未达趋势关注阈值（0.5 ℃）。"
        )

    # ---- 指标卡片 -----------------------------------------------------------
    n_p90, n_p10 = anomaly_detection.grid_threshold_exceedance(
        day_sst_grid, clim, pd.Timestamp(selected_date).day_of_year)
    cols = st.columns(5)
    cards = [
        ("显著偏暖格点", f"{stats['n_warm']} 个", f"z ≥ {config.ANOMALY_STD_THRESHOLD}"),
        ("显著偏冷格点", f"{stats['n_cold']} 个", f"z ≤ -{config.ANOMALY_STD_THRESHOLD}"),
        ("达热浪阈值格点", f"{n_p90} 个", "当日超 p90，未含持续性判定"),
        ("最大正距平", f"{stats['max_anom']:+.2f} ℃", "相对历史同期"),
        ("最大负距平", f"{stats['min_anom']:+.2f} ℃", "相对历史同期"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(
            f'<div class="metric-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div>'
            f'<div class="sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )

    # ---- 距平分布图 + 阈值序列图 -------------------------------------------
    st.markdown("")
    map_col, chart_col = st.columns([3, 2])
    with map_col:
        map_view = st.radio("地图视图", ["距平分布", "异常状态"],
                            horizontal=True, key="anomaly_map_view")
        if map_view == "距平分布":
            fig_map = visualization.anomaly_map(
                day_anom["anomaly"], title=f"海温距平分布 · {selected_date}")
        else:
            fig_map = visualization.anomaly_status_map(
                day_anom["z_score"], title=f"异常状态分布 · {selected_date}")
        st.plotly_chart(fig_map, width='stretch', config=visualization.PLOTLY_CONFIG)
    with chart_col:
        window = st.radio("时间范围", options=list(visualization.HIST_WINDOWS),
                          index=1, horizontal=True, key="anom_hist_window")
        fig_ts = visualization.anomaly_series_chart(
            series, thresholds, events,
            title="整个渤海：海温与季节性阈值", window=window)
        st.plotly_chart(fig_ts, width='stretch', config=visualization.PLOTLY_CONFIG)

    # ---- 历史事件列表 -------------------------------------------------------
    st.markdown("#### 📋 海洋热浪 / 冷事件历史记录")
    if events.empty:
        st.info("研究时段内未识别到达到标准的事件。")
    else:
        n_heat = int((events["类型"] == "海洋热浪").sum())
        n_cold = int((events["类型"] == "海洋冷事件").sum())
        st.caption(
            f"共识别 {len(events)} 个事件（海洋热浪 {n_heat} 个、海洋冷事件 {n_cold} 个），"
            f"识别规则：超过季节第 90/10 百分位且连续 ≥{config.MHW_MIN_DURATION} 天"
            f"（间隔 ≤{config.MHW_MAX_GAP} 天合并），"
            f"基准期 {config.CLIM_LABEL}{_clim_note()}。"
            f"完整记录已保存至 `data/processed/{anomaly_detection.EVENTS_CSV.name}`。"
        )
        st.dataframe(events, width='stretch', hide_index=True)
        st.download_button(
            "⬇️ 下载事件记录 (CSV)",
            events.to_csv(index=False, encoding="utf-8-sig"),
            file_name="mhw_events.csv", mime="text/csv",
        )

    # ---- 方法说明 -----------------------------------------------------------
    with st.expander("异常识别方法说明"):
        st.markdown(
            f"""
**1. 普通海温异常（格点级）**
- 气候态：每个格点逐"年日"的均值与标准差（11 天滑动窗口，基准期 {config.CLIM_LABEL}）
- 距平 = 当日海温 − 同期气候态均值；标准化距平 z = 距平 / 同期标准差
- 判据：|z| ≥ {config.ANOMALY_STD_THRESHOLD} 视为显著偏暖 / 偏冷
- 注意：格点级判据只看单日统计显著性，**不做持续性判定**，不等同于海洋热浪

**2. 海洋热浪（MHW，区域平均序列级，Hobday et al. 2016）**
- 阈值：区域平均海温的季节性**第 90 百分位**（11 天窗口 + 5 天环形平滑）
- 判据：连续 **≥{config.MHW_MIN_DURATION} 天**超过阈值；
  相邻超阈值时段间隔 **≤{config.MHW_MAX_GAP} 天合并**为同一事件
- 对称规则：连续 ≥{config.MHW_MIN_DURATION} 天低于第 10 百分位记为海洋冷事件
- 事件作用于**区域平均海温序列**，与上面的格点级 z 分数是两个层级，不可混淆
- 基准期 {config.CLIM_LABEL}{_clim_note()}
"""
        )

    # ---- 指标口径说明（统一各页面结论的依据） ---------------------------------
    with st.expander("指标口径说明（空间/时间尺度与阈值）"):
        st.markdown(
            fr"""
| 指标 | 空间尺度 | 时间尺度 | 计算方法 | 阈值/判据 |
| --- | --- | --- | --- | --- |
| 海温距平 | 格点 / 区域 | 单日 | 当日海温 − 同期气候态均值 | — |
| 标准化距平 z | 格点 | 单日 | 距平 ÷ 同期标准差 | \|z\| ≥ {config.ANOMALY_STD_THRESHOLD} 为显著 |
| 显著偏暖/偏冷格点 | 格点计数 | 单日 | z 超阈值格点数量 | 同上 |
| 达热浪阈值格点 | 格点计数 | 单日 | 海温 ≥ 当日季节 p90 | 未含持续性判定 |
| 海洋热浪/冷事件 | 整个渤海（区域平均） | 连续 ≥{config.MHW_MIN_DURATION} 天 | 季节 p90/p10 + 持续性 + ≤{config.MHW_MAX_GAP} 天合并 | Hobday et al. 2016 |
| 预测风险 | 整个渤海 | 未来 1~3 天 | 随机森林预测 vs 季节 p90/p10 | 提示性，非确定性指令 |
| 综合预警等级 | 整个渤海 | 综合当日 + 预测 | 规则引擎（预警中心） | Ⅰ~Ⅳ级，项目内部规则 |

所有页面（首页/监测/异常/预测/预警）使用同一数据包与同一气候态基准，
同一日期同一区域的结论在各页面保持一致。
"""
        )


def main() -> None:
    render_header()
    page = render_sidebar()

    if page == "🏠 系统首页":
        render_home()
    elif page == "🗺️ 海洋环境监测中心":
        render_monitoring()
    elif page == "🌡️ 海温异常识别中心":
        render_anomaly()
    elif page == "🤖 海温智能预测中心":
        render_prediction()
    elif page == "⚠️ 海洋智能预警中心":
        render_warning()


if __name__ == "__main__":
    main()
