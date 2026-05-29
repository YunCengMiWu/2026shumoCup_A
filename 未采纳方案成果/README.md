# 电工杯A题：绿电制氨园区调度优化 — 代码方案 v2

## 项目简介

本项目针对2026年电工杯数学建模竞赛A题"绿电制氨园区调度优化"，基于PuLP构建MILP/LP优化模型，求解Q1~Q5五个子问题。模型涵盖风光出力分析、碱性与PEM电解槽制氢调度、合成氨生产优化、离网储能配置以及参数敏感性分析，所有结果以CSV/PNG/XLSX形式输出至`results/`目录，供论文直接引用。

代码按四波（Wave1~4）推进开发：基础组件搭建 → 单问题建模求解 → 图表生成 → 汇总整合与文档定稿。

## 目录结构

```
code_solution_v2/
├── README.md                    # 本文件
├── run_all.py                   # 一键运行入口（Q1→Q5顺序执行）
├── models/                      # Q1~Q5 优化模型
│   ├── run_q1.py                # Q1 典型日绿电指标计算
│   ├── q1_calculation.py        # Q1 典型日指标计算逻辑
│   ├── q2_milp.py               # Q2 离散制氨调节 MILP 模型 + 入口
│   ├── q3_lp.py                 # Q3 连续制氨调节 LP 模型 + 入口
│   ├── q3_clustering.py         # Q3 K-means 聚类（6特征典型场景提取）
│   ├── clustering.py            # 通用聚类工具
│   ├── q4_storage.py            # Q4 离网储能配置优化（MILP）
│   ├── q5_scanner.py            # Q5 参数α敏感性分析
│   └── power_balance.py         # 功率平衡计算
├── utils/                       # 工具模块
│   ├── constants.py             # 全局常量（BIG_M=1000、设备额定参数等）
│   ├── data_loader.py           # 附件1~8数据读取与预处理
│   ├── indicators.py            # 绿电指标与成本计算
│   ├── economics.py             # 经济性计算
│   └── excel_exporter.py        # 结果汇总导出至 全部结果.xlsx
├── visualization/               # 可视化
│   ├── charts_q1.py             # Q1 多指标折线图 / 功率平衡堆叠图
│   ├── charts_q2.py             # Q2 甘特图 / 箱线图 / 成本分布
│   ├── charts_q3.py             # Q3 聚类饼图 / 雷达图 / 散点图 / 箱线图
│   ├── charts_q4.py             # Q4 SOC热力图 / 成本散点 / Q4 vs Q3对比
│   ├── charts_q5.py             # Q5 敏感性曲线 / 策略对比
│   ├── charts_cluster.py        # Q3 聚类分析图表
│   ├── chinese_font.py          # 中文字体配置（SimHei 自动回退）
│   └── plotting.py              # 统一绘图工具（样式、颜色主题）
├── data/                        # 原始数据附件
│   ├── 附件1.xlsx ~ 附件8.xlsx
└── results/                     # 输出结果（脚本自动生成，详见下文）
    ├── q1_multi_metric.png
    ├── q1_power_balance.png
    ├── q1_sankey.html
    ├── q2_results.csv
    ├── q2_gantt.png
    ├── q2_boxplot.png
    ├── q2_cost_distribution.png
    ├── q3_results.csv
    ├── q3_cluster_labels.csv
    ├── q3_pie.png
    ├── q3_radar.png
    ├── q3_scatter.png
    ├── q3_boxplot.png
    ├── q3_cluster_report.txt
    ├── q4_results.csv
    ├── q4_soc_heatmap.png
    ├── q4_cost_scatter.png
    ├── q4_vs_q3_comparison.png
    ├── q5_sensitivity.csv
    ├── q5_sensitivity_curves.png
    ├── q5_policy_comparison.png
    └── 全部结果.xlsx
```

## 依赖安装

Python 3.9及以上版本。依赖包安装命令：

```
pip install pulp pandas numpy matplotlib openpyxl scikit-learn
```

| 包名 | 用途 |
|------|------|
| `pulp` | MILP/LP 线性规划建模与求解 |
| `pandas` | 时序数据处理与CSV读写 |
| `numpy` | 数值计算 |
| `matplotlib` | 图表绘制（中文标注） |
| `openpyxl` | Excel结果汇总导出 |
| `scikit-learn` | K-means聚类（Q3典型场景提取） |

## 快速开始

```bash
cd code_solution_v2
python run_all.py
```

脚本按Q1至Q5顺序依次执行各子问题的建模、求解与图表输出，结果文件自动写入`results/`目录。全程约2~5分钟（Q2 MILP为耗时瓶颈，24场景×5产量档位共120次求解）。

单独运行某一问题：

```bash
python models/run_q1.py            # 问题一：典型日绿电指标计算
python models/q2_milp.py           # 问题二：离散制氨调节MILP
python models/q3_lp.py             # 问题三：连续制氨调节LP
python models/run_q4.py            # 问题四：离网储能配置MILP
python models/run_q5.py            # 问题五：参数α敏感性分析
```

## 各问题说明

### Q1 — 典型日绿电指标计算

基于风电（40 MW）与光伏（64 MW）全年8760小时出力数据，选取典型日计算绿电自消纳率η_self、绿电占比η_green、并网上网率η_grid三项核心指标，并绘制Sankey能流图。

### Q2 — 离散制氨调节优化（MILP）

在Q1典型日负荷基础上，引入碱性与PEM电解槽离散档位调节约束（启停0-1变量），以日运行成本最小为目标建立混合整数线性规划模型，求解最优制氢调度方案。

### Q3 — 连续制氨调节优化（LP）

将Q2的离散档位约束松弛为连续功率调节，同时采用K-means聚类提取多典型场景，以运行成本最小为目标建立线性规划模型，分析连续调节相对离散调节的经济性提升。

### Q4 — 离网储能配置优化（MILP）

在园区离网运行场景下，以年化储能投资成本与年运行成本之和（TAC）最小为目标，优化电池储能容量E与功率P的配置方案，约束储能SOC初值=0.5E且充放电效率0.90。

### Q5 — 参数α敏感性分析

以权重系数α合成多目标函数：f = min(购电成本 − α×售电收入 + RE发电成本 + 运维成本 + 折旧)。α∈[1,4]以0.1为步长共31个取值点，逐点求解Q3连续调度模型，生成成本-消纳率敏感性曲线与α=1 vs α=4策略对比图，分析α对调度方案经济性与绿电消纳的权衡影响。

## 核心公式速查

| 指标 | 公式 | 物理含义 |
|------|------|---------|
| 绿电自消纳率 η_self | (E_total − E_sell − E_buy) / E_RE | 本地消纳绿电占比（需>60%） |
| 绿电占比 η_green | (E_RE − E_sell) / E_total | 总用电中绿电占比（需>30%） |
| 并网上网率 η_grid | E_sell / E_RE | 绿电外送比例（需<20%） |
| 功率平衡 | P_wind + P_pv + P_buy = P_base + P_H2 + P_NH3 + P_sell + P_ch − P_dis | 各时刻园区功率守恒 |
| 氢气平衡 | λ_ALK · P_ALK + λ_PEM · P_PEM = 200 · M_NH3 | 制氢量 = 合成氨耗氢量 |

## 结果文件说明

所有输出文件位于`results/`目录，按问题分组。以下为完整清单及说明。

**CSV数据文件：**

- `q2_results.csv` — Q2 MILP 24场景×5产量 120行结果（吨氨成本、购售电量）
- `q3_results.csv` — Q3 LP 24场景连续调节结果（最小运行成本模式）
- `q3_cluster_labels.csv` — Q3 K-means聚类标签（k=2，两簇6/18分布）
- `q4_results.csv` — Q4 离网储能优化 24场景结果（E/P容量、TAC成本）
- `q5_sensitivity.csv` — Q5 α∈[1,4]参数扫描 31行敏感性结果
- `全部结果.xlsx` — 汇总Excel，5个Sheet（Q1指标/Q2/Q3/Q4/Q5）

**图表PNG文件：**

- `q1_multi_metric.png` — Q1 多指标折线图（用电/新能源/网购/上网/吨氨成本，24h）
- `q1_power_balance.png` — Q1 典型日功率平衡堆叠面积图
- `q2_gantt.png` — Q2 设备启停甘特图（ALK/PEM/NH3二值状态）
- `q2_boxplot.png` — Q2 24场景吨氨成本箱线图
- `q2_cost_distribution.png` — Q2 成本分布密度曲线
- `q3_pie.png` — Q3 聚类簇占比饼图
- `q3_radar.png` — Q3 两簇6指标特征雷达图
- `q3_scatter.png` — Q3 PCA降维两簇散点图
- `q3_boxplot.png` — Q3 两簇关键指标对比箱线图
- `q4_soc_heatmap.png` — Q4 24场景SOC热力图（24h×24场景）
- `q4_cost_scatter.png` — Q4 TAC成本散点图（按储能配置着色）
- `q4_vs_q3_comparison.png` — Q4离网 vs Q3并网 成本对比柱状图
- `q5_sensitivity_curves.png` — Q5 α敏感性三曲线（吨氨成本/η_self/η_green vs α）
- `q5_policy_comparison.png` — Q5 α=1 vs α=4 多指标对比柱状图

**其他文件：**

- `q1_sankey.html` — Q1 交互式桑基能流图（HTML，浏览器打开）
- `q3_cluster_report.txt` — Q3 聚类分析详细报告（轮廓系数/簇特征）

## 论文图表引用

下表建立结果文件与论文 Figure/Table 编号的对应关系，供写作时直接调用。

| 结果文件 | 论文引用建议 | 说明 |
|----------|-------------|------|
| `q1_sankey.html` | Figure 2 | 典型日能流桑基图 |
| `q1_multi_metric.png` | Figure 3 | 典型日多指标折线图 |
| `q1_power_balance.png` | Figure 4 | 功率平衡堆叠图 |
| `q2_results.csv` | Table 1 | 24场景MILP求解汇总 |
| `q2_gantt.png` | Figure 5 | 设备启停甘特图 |
| `q2_boxplot.png` | Figure 6 | 吨氨成本箱线图 |
| `q2_cost_distribution.png` | Figure 7 | 成本分布曲线 |
| `q3_results.csv` | Table 2 | 24场景LP求解汇总 |
| `q3_pie.png` / `q3_radar.png` / `q3_scatter.png` / `q3_boxplot.png` | Figure 8a-d | 聚类分析四图 |
| `q4_results.csv` | Table 3 | 离网储能优化结果 |
| `q4_soc_heatmap.png` | Figure 9 | SOC热力图 |
| `q4_vs_q3_comparison.png` | Figure 10 | 离网vs并网成本对比 |
| `q5_sensitivity_curves.png` | Figure 11 | α敏感性曲线 |
| `q5_sensitivity.csv` | Table 4 | 参数扫描数值结果 |
| `q5_policy_comparison.png` | Figure 12 | 政策建议对比 |

> Figure/Table 编号为建议值，请根据论文实际章节顺序调整。

## 常见问题（FAQ）

**Q: 图表中的中文显示为方框（□□）怎么办？**
A: 需要安装SimHei（黑体）字体。下载SimHei.ttf后放入`C:\Windows\Fonts\`，重启Python即可。也可修改`visualization/chinese_font.py`中的字体名称。

**Q: 运行时报 `ModuleNotFoundError: No module named 'pulp'`？**
A: 执行 `pip install pulp pandas numpy matplotlib openpyxl scikit-learn` 安装全部依赖。

**Q: `run_all.py` 运行需要多长时间？**
A: 总计约30-60秒。Q2 MILP（24场景×5产量）最耗时约6秒，Q4离网优化约13秒，其余子问题均在2秒以内。图表生成各1-3秒。

**Q: Q4全部场景显示Infeasible（不可行）？**
A: 检查`utils/constants.py`中三个关键参数已设置：`SOC_INITIAL_FRAC = 0.5`（储能初始SOC）、`MIN_LOAD_RATIO = 0.10`（设备最低负荷10%）、`BIG_M = 1000`。

**Q: Excel打开CSV文件中文乱码？**
A: 所有CSV均使用`utf-8-sig`编码（带BOM标记），直接双击打开应正常。如仍乱码，请使用`results/全部结果.xlsx`（已格式化的汇总Excel）。

**Q: 如何修改模型参数（如BIG_M、风/光容量）重新求解？**
A: 修改`utils/constants.py`中对应常量值，然后运行对应模型的`run_qX.py`脚本或重新运行`python run_all.py`。

**Q: 如何单独运行某个子问题的可视化？**
A: 直接执行对应图表脚本：`python visualization/charts_q1.py`（Q1图表）至`charts_q5.py`（Q5图表）。
