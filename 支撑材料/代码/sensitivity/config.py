"""
sensitivity/config.py — OAT参数灵敏度分析配置模块

定义待分析参数列表、扰动范围、扰动点数，以及典型场景选择。
供 Q1-Q4 所有灵敏度分析脚本统一导入。

依赖: 无外部依赖，可独立导入。
"""

import os
import numpy as np

# =============================================================================
# 路径配置 — 通过 sys.path 访问根目录模块
# =============================================================================

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), 'figures')

# =============================================================================
# OAT 扰动配置
# =============================================================================

PERTURBATION_RANGE = 0.30   # ±30% 基线附近
PERTURBATION_POINTS = 5     # 5 个等距扰动点 (含基线)

# =============================================================================
# 待分析参数列表
#
# 每个参数包含:
#   name       — 根目录 config.py 中的常量名 (字符串, 用于 getattr/setattr)
#   label      — 中文标签 (用于图表标题/轴标签)
#   baseline   — 根目录 config.py 中的基线值
#   unit       — 单位 (用于图表标注)
#   range      — [下限, 上限] 合理物理/经济范围, 扰动可能超出
# =============================================================================

PARAMETERS = [
    {
        "name": "green_penalty_self_use",
        "label": "自发自用率惩罚$\\lambda_1$",
        "baseline": 0.5,
        "unit": "¥/kWh",
        "range": [0.2, 1.0],
    },
    {
        "name": "green_penalty_green_rate",
        "label": "绿电比例惩罚$\\lambda_2$",
        "baseline": 0.3,
        "unit": "¥/kWh",
        "range": [0.1, 0.6],
    },
    {
        "name": "green_penalty_feed_rate",
        "label": "上网比例惩罚$\\lambda_3$",
        "baseline": 1.5,  # aligned with config.py GREEN_PENALTY_FEED_RATE (赵昊天2026 δ=1.5)
        "unit": "¥/kWh",
        "range": [0.5, 2.5],
    },
    {
        "name": "nh3_load_min",
        "label": "NH$_3$最低负荷率",
        "baseline": 0.30,
        "unit": "ratio",
        "range": [0.10, 0.50],
    },
    {
        "name": "tou_price_scale",
        "label": "分时电价系数",
        "baseline": 1.0,
        "unit": "scale",
        "range": [0.7, 1.3],
    },
    {
        "name": "wind_lcoe",
        "label": "风电LCOE",
        "baseline": 0.15,
        "unit": "¥/kWh",
        "range": [0.075, 0.225],
    },
    {
        "name": "solar_lcoe",
        "label": "光伏LCOE",
        "baseline": 0.12,
        "unit": "¥/kWh",
        "range": [0.06, 0.18],
    },
    {
        "name": "storage_investment",
        "label": "储能投资成本",
        "baseline": 1000,
        "unit": "¥/kWh",
        "range": [500, 1500],
    },
    {
        "name": "alk_h2_per_mw",
        "label": "ALK制氢效率$\\eta_{alk}$",
        "baseline": 14,
        "unit": "kg/(h·MW)",
        "range": [11.9, 16.1],
    },
    {
        "name": "pem_h2_per_mw",
        "label": "PEM制氢效率$\\eta_{pem}$",
        "baseline": 16,
        "unit": "kg/(h·MW)",
        "range": [13.6, 18.4],
    },
    {
        "name": "scale_factor",
        "label": "产能规模系数",
        "baseline": 2.0,
        "unit": "scale",
        "range": [1.5, 2.5],
    },
    {
        "name": "alkel_om",
        "label": "ALK运维费",
        "baseline": 0.10,
        "unit": "¥/kWh",
        "range": [0.05, 0.15],
    },
    {
        "name": "pemel_om",
        "label": "PEM运维费",
        "baseline": 0.15,
        "unit": "¥/kWh",
        "range": [0.075, 0.225],
    },
    {
        "name": "alkel_rated_scale",
        "label": "ALK电解槽容量缩放",
        "baseline": 1.0,
        "unit": "scale",
        "range": [0.5, 1.5],
    },
    {
        "name": "pemel_rated_scale",
        "label": "PEM电解槽容量缩放",
        "baseline": 1.0,
        "unit": "scale",
        "range": [0.5, 1.5],
    },
    {
        "name": "nh3_om",
        "label": "NH3运维费",
        "baseline": 0.002,
        "unit": "¥/kWh",
        "range": [0.001, 0.005],
    },
    {
        "name": "grid_feedin_price",
        "label": "上网电价",
        "baseline": 0.3779,
        "unit": "¥/kWh",
        "range": [0.20, 0.60],
    },
]

# =============================================================================
# OAT 扰动采样
# =============================================================================

def generate_oat_points(param_def: dict) -> list:
    """为指定参数生成 OAT 扰动采样点列表 (不含基线值)。

    在 [range_min, range_max] 内等间距生成 PERTURBATION_POINTS 个点，
    剔除与 baseline 重合的点（容忍 1e-10）。

    Args:
        param_def: PARAMETERS 列表中的一项字典，含 range, baseline

    Returns:
        float 列表，长度 ≤ PERTURBATION_POINTS - 1
    """
    rmin, rmax = param_def['range']
    n = PERTURBATION_POINTS
    baseline = param_def['baseline']
    points = np.linspace(rmin, rmax, n).tolist()
    # 剔除基线值（浮点容差）
    points = [float(p) for p in points if abs(p - baseline) > 1e-10]
    return points


# 参数名 → 索引 快速查找
PARAM_INDEX = {p["name"]: i for i, p in enumerate(PARAMETERS)}

# =============================================================================
# 典型场景选择
#
# 从 6×4=24 个风-光组合中选取 8 个覆盖不同象限的典型场景:
#  (0,0) — 最低风电 + 最低光伏 (最不利)
#  (0,2) — 最低风电 + 中等光伏
#  (1,1) — 低风电   + 低光伏
#  (2,0) — 中等风电 + 最低光伏
#  (2,3) — 中等风电 + 最高光伏
#  (3,1) — 较高风电 + 低光伏
#  (4,2) — 高风电   + 中等光伏
#  (5,3) — 最高风电 + 最高光伏 (最有利)
# =============================================================================

TYPICAL_SCENARIOS = [
    (0, 0), (0, 2), (1, 1), (2, 0),
    (2, 3), (3, 1), (4, 2), (5, 3),
]

# 场景总数 (来自根 config.py, 此处硬编码以减少导入依赖)
N_WIND_SCENARIOS = 6
N_SOLAR_SCENARIOS = 4
N_SCENARIOS = N_WIND_SCENARIOS * N_SOLAR_SCENARIOS  # 24

# =============================================================================
# 指标名称映射
# =============================================================================

METRIC_LABELS = {
    "ton_ammonia_cost": "吨氨成本 (¥/ton)",
    "self_use_rate": "自发自用率",
    "green_rate": "绿电比例",
    "grid_feed_rate": "上网比例",
    "total_cost": "总成本 (¥/day)",
    "grid_buy_cost": "购电费 (¥/day)",
    "grid_sell_revenue": "售电收入 (¥/day)",
    "wind_curtailment": "弃风率",
    "solar_curtailment": "弃光率",
    "total_curtailment": "总弃电率",
}
