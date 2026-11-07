# 渤海智瞳 · 海温异常监测及智能预警系统

> 2026 年辽宁省大学生智慧海洋科技创新大赛 · 海洋智能监测赛道参赛作品

基于 **NOAA OISST v2.1** 卫星遥感海表温度数据与轻量化机器学习，
实现渤海海域海温 **监测 → 异常识别 → 智能预测 → 风险预警** 全流程。

## 环境要求

- Python 3.11+（本项目在 Windows + Python 3.13 上开发验证）
- 无需数据库、无需付费 API、无需特殊硬件

## 快速开始

```bash
# 1. 创建虚拟环境（推荐）
python -m venv .venv
# Windows 激活：
.venv\Scripts\activate
# macOS / Linux 激活：
# source .venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 启动系统
streamlit run app.py
```

浏览器将自动打开 `http://localhost:8501`。

## 数据下载（首次使用必做）

```bash
# 下载 2020-01-01 至今的渤海区域 OISST 数据（约 2500 个文件，共 ~20MB，约 30~60 分钟）
python download_data.py

# 自定义范围 / 并发
python download_data.py --start 2023-01-01 --end 2025-12-31 --threads 8
```

- 下载渠道：优先 **AWS S3 NOAA 公开数据桶**（全球文件下载后本地裁剪，速度快）；
  日期缺失时自动回退 **NCEI THREDDS NCSS 子集服务**。
- 支持断点续下：已存在的日期自动跳过，可随时中断后重跑。
- 若本机无法访问外网：在能上网的电脑运行后，把 `data/raw/` 目录整体拷贝过来即可。

## 目录结构

```
bohai-smart-ocean/
├── app.py                  # Streamlit 系统入口（五大页面）
├── config.py               # 全局配置（区域 / 数据源 / 阈值）
├── requirements.txt        # 依赖清单
├── run.bat                 # Windows 一键启动
├── download_data.py        # NOAA OISST 数据下载（S3 + NCSS 双通道）
├── data_loader.py          # NetCDF 读取、合并、海域掩膜、缓存
├── preprocess.py           # 区域时间序列、气候态基准、特征工程
├── anomaly_detection.py    # 距平分析、海洋热浪/冷事件识别
├── prediction.py           # 持续性基线 vs 随机森林短期预测
├── warning.py              # Ⅰ~Ⅳ级规则预警引擎
├── visualization.py        # Plotly 深色主题图表组件
├── report_generator.py     # PDF 分析报告导出
├── data/
│   ├── raw/                # 逐日 NetCDF（13000+ 个文件）
│   └── processed/          # 合并缓存、事件记录、时间序列 CSV
├── assets/
│   └── coastline_bohai.json # 渤海海岸线（Natural Earth 10m 提取）
├── models/                 # 训练好的随机森林模型
└── tests/                  # 基础测试（12 项，可无 pytest 运行）
```

## 数据来源

- **NOAA OISST v2.1**（Optimum Interpolation Sea Surface Temperature）
- 分辨率：0.25° × 0.25°，逐日
- 官方地址：<https://www.ncei.noaa.gov/products/optimum-interpolation-sst>
- 研究区域：渤海（117°E–122°E，37°N–41°N）

## 开发进度

- [x] 第一阶段：项目骨架、首页与导航
- [x] 第二阶段：数据下载、读取、预处理与海温可视化
- [x] 第三阶段：海温异常识别
- [x] 第四阶段：机器学习预测与基线对比
- [x] 第五阶段：智能预警与报告导出
- [x] 第六阶段：UI 优化、测试与演示


## 运行测试

```bash
python tests/test_basic.py      # 12 项测试：配置/气候态/特征无泄漏/事件规则/预测对齐/预警分级/PDF
```

## 常见问题

- **页面报 `module 'xxx' has no attribute 'yyy'` 或界面与代码不一致**：
  说明运行中的服务是在代码修改前启动的。Streamlit 页面重跑不会重新导入
  已加载的模块——**关闭服务重新启动即可**（Ctrl+C 后重新运行 run.bat）。
  若端口被占用：`netstat -ano | findstr 8501` 找到 PID 后 `taskkill /PID <pid> /F`。
- **首次打开首页需要约 30 秒**：正在加载 13000+ 天数据并训练预测模型，
  之后所有交互秒开（多级缓存）。答辩前请先打开一次预热。

## 已知技术要点

- **中文路径兼容**：netCDF4 底层 C 库无法读写含中文的路径，项目内所有
  NetCDF 读写均通过"内存打开 / 临时文件搬运"规避（见 `data_loader.py`、
  `download_data.py`）。**建议项目放在纯英文路径下**，可彻底避免此类问题。
- **气候态基准期**：1991–2020（WMO 标准 30 年气候态），由 `config.py` 的
  `CLIM_YEARS` 统一配置；若改为短于 30 年的基准期，界面与 PDF 会自动
  标注"短基准期探索性识别"。基准期数据不足时系统会明确降级提示，
  不会用残缺数据计算阈值。
- **海岸线数据**：`assets/coastline_bohai.json` 提取自 Natural Earth 10m
  coastline（一次性离线提取，运行时不依赖网络与 pyshp）。
- **离线演示**：全部数据与资产本地化，断网可完整演示（仅侧栏数据来源
  链接与增量下载需要网络）。
