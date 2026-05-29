"""
config.py — 电工杯A题 全局参数配置与数据加载模块

加载全部8个附件Excel文件，定义系统常量，提供数据访问接口。
"""

import os
import numpy as np
import pandas as pd

# =============================================================================
# 路径配置
# =============================================================================

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '2026年电工杯竞赛赛题', 'A题'
)

def _path(filename: str) -> str:
    """构造附件完整路径"""
    return os.path.join(DATA_DIR, filename)

# =============================================================================
# 核心设备容量 (MW) — 基准产能 36 ton/day
# =============================================================================

WIND_RATED    = 40   # MW  风电装机
SOLAR_RATED   = 64   # MW  光伏装机
LOAD_PEAK     = 6    # MW  常规用电负荷峰值
ALKEL_RATED   = 10   # MW  碱性电解槽 (36t/d 基准)
PEMEL_RATED   = 10   # MW  PEM电解槽 (36t/d 基准)
NH3_RATED     = 0.75 # MW  合成氨装置 (36t/d 基准)

# =============================================================================
# 设备台数与单体参数 (v2风格：拆分建模，拒绝聚合)
# =============================================================================

# --- 设备台数 ---
N_ALK = 2          # 碱性电解槽 (ALK) 台数 (10 MW/台)
N_PEM = 2          # PEM电解槽台数 (10 MW/台)
N_NH3 = 2          # 合成氨装置台数 (0.75 MW/台, 满足72t/d产能)

# --- 单体额定功率 (MW/台) ---
ALK_UNIT_POWER = 10    # 碱性电解槽
PEM_UNIT_POWER = 10    # PEM电解槽
NH3_UNIT_POWER = 0.75  # 合成氨装置

# --- 系统总功率 (MW) ---
ALK_SYS_POWER = N_ALK * ALK_UNIT_POWER   # 20 MW (2台×10)
PEM_SYS_POWER = N_PEM * PEM_UNIT_POWER   # 20 MW (2台×10)
NH3_SYS_POWER = N_NH3 * NH3_UNIT_POWER   # 1.5 MW (2台×0.75)

# --- 制氢速率 (已折算效率, kgH2 / (h·MW)) ---
# ALK: 50 kWh/kg ÷ 0.70 eff = 71.43 kWh/kg → 1000/71.43 ≈ 14 kg/(h·MW)
# PEM: 50 kWh/kg ÷ 0.80 eff = 62.50 kWh/kg → 1000/62.50 = 16 kg/(h·MW)
ALK_H2_PER_MW = 14   # 碱性电解槽制氢速率
PEM_H2_PER_MW = 16   # PEM电解槽制氢速率

# --- 氢气-氨转换 ---
H2_PER_TON_NH3 = 200     # kg H2 / ton NH3 (附件6)
NH3_POWER_RATE = 0.002   # ton NH3 / (kW·h) (1.5t/h ÷ 0.75MW ÷ 1000)

# =============================================================================
# 学术约束参数 (基于郑勇2025、赵昊天2025)
# =============================================================================

# --- NH3爬坡率 (效率变化≤20%/h, 即 |ΔP| ≤ 0.20 × P_rated) ---
NH3_RAMP_RATE = 0.20       # 爬坡率上限, 占额定功率比例
# 72t/d 场景下: 0.20 × 1.5 MW = 0.30 MW/h
# 36t/d 场景下: 0.20 × 0.75 MW = 0.15 MW/h
NH3_LOAD_MIN  = 0.10          # 最低负荷率 (统一10%, 严格遵循赛题要求)
NH3_LOAD_MAX  = 1.10          # 最高负荷率 (郑勇: 110%)

# --- 最小启停时间 (赵昊天2025框架) ---
MIN_RUN_HOURS  = 4    # 最小连续运行时间 (h)
MIN_STOP_HOURS = 2    # 最小连续停机时间 (h)

# =============================================================================
# MILP建模参数
# =============================================================================

BIG_M = 1000           # 大M常数 (购售电互斥约束, 原104不足以覆盖发电峰值)

# --- 优化模型通用参数 ---
MIN_LOAD_RATIO = 0.10   # 设备最小负荷率 (10%, 问题描述)
DELTA_T = 1.0            # 调度时间步长 (小时)
HOURS_PER_DAY = 24       # 每日调度时段数

# =============================================================================
# 产能参数 (基准 36 ton/day)
# =============================================================================

NH3_BASELINE_CAPACITY = 36       # ton/day
PRODUCTION_LEVELS     = [72, 63, 54, 45, 36]  # ton/day

ALKEL_H2_RATE      = 140   # kgH2/h
PEMEL_H2_RATE      = 160   # kgH2/h
NH3_RATE           = 1.5   # ton/h (at 36t/day)
NH3_H2_CONSUMPTION = 0.2   # kgH2 per kgNH3
NH3_ELEC_CONSUMPTION = 0.5 # kWh per kgNH3

ALKEL_ELEC_PER_KG  = 50    # kWh/kgH2 (未考虑效率)
PEMEL_ELEC_PER_KG  = 50    # kWh/kgH2 (未考虑效率)

# =============================================================================
# 效率参数
# =============================================================================

ALKEL_EFF = 0.70
PEMEL_EFF = 0.80

# =============================================================================
# 成本参数
# =============================================================================

# 度电成本 (yuan/kWh)
WIND_LCOE  = 0.15
SOLAR_LCOE = 0.12

# 运维系数 (yuan/kWh)
ALKEL_OM = 0.10
PEMEL_OM = 0.15

# 电储能
STORAGE_INVESTMENT = 1000   # yuan/kWh
STORAGE_OM         = 0.01   # yuan/kWh
STORAGE_LIFETIME   = 15     # years

# 合成氨装置
NH3_INVESTMENT = 60000      # yuan/kgH2
NH3_OM         = 0.002      # yuan/kWh
NH3_LIFETIME   = 30         # years

# 设备寿命 (years)
WIND_LIFETIME   = 25
SOLAR_LIFETIME  = 25
ALKEL_LIFETIME  = 30
PEMEL_LIFETIME  = 30

# =============================================================================
# 储能参数
# =============================================================================

STORAGE_CHARGE_EFF    = 0.90
STORAGE_DISCHARGE_EFF = 0.90
STORAGE_SELF_DISCHARGE = 0.002  # 0.2%

# =============================================================================
# 电网价格
# =============================================================================

GRID_FEEDIN_PRICE = 0.3779   # yuan/kWh (风电+光伏均为此价)

# 分时电价查找表 (hour_index -> price)
_TOU_PRICE_MAP = {}
# 低谷: 23, 0-6
for h in [23] + list(range(0, 7)):
    _TOU_PRICE_MAP[h] = 0.3424
# 平段: 7-9, 15-17, 21-22
for h in list(range(7, 10)) + list(range(15, 18)) + list(range(21, 23)):
    _TOU_PRICE_MAP[h] = 0.6074
# 高峰: 10-14, 18-20
for h in list(range(10, 15)) + list(range(18, 21)):
    _TOU_PRICE_MAP[h] = 0.8024


def get_price(hour: int) -> float:
    """返回指定小时的分时电价 (yuan/kWh)

    Args:
        hour: 0-23 整数小时索引

    Returns:
        分时电价 (yuan/kWh)
    """
    return _TOU_PRICE_MAP[hour % 24]

# =============================================================================
# 场景索引
# =============================================================================

N_WIND_SCENARIOS  = 6
N_SOLAR_SCENARIOS = 4
N_SCENARIOS       = N_WIND_SCENARIOS * N_SOLAR_SCENARIOS  # 24

SCENARIO_INDEX = [(w, s) for w in range(N_WIND_SCENARIOS)
                       for s in range(N_SOLAR_SCENARIOS)]
# [(0,0),(0,1),(0,2),(0,3), (1,0),...,(5,3)]

# =============================================================================
# 数据加载 (模块级, 导入时执行)
# =============================================================================

def load_load_profile() -> pd.DataFrame:
    """加载附件1: 园区常规用电负荷标幺功率曲线

    Returns:
        DataFrame with columns ['时段', '标幺功率']
    """
    df = pd.read_excel(_path('附件1：园区典型日常规电负荷标幺功率曲线.xlsx'))
    df.columns = ['时段', '标幺功率']
    return df

def load_typical_wind_solar() -> pd.DataFrame:
    """加载附件2: 典型日风电+光伏标幺功率

    Returns:
        DataFrame with columns ['时段', '风电标幺值', '光伏标幺值']
    """
    df = pd.read_excel(_path('附件2：典型日风电、光伏标幺功率表.xlsx'))
    df.columns = ['时段', '风电标幺值', '光伏标幺值']
    return df

def load_wind_scenarios() -> pd.DataFrame:
    """加载附件3: 6种风电场景标幺功率

    Returns:
        DataFrame with columns ['时段', '场景1', ..., '场景6']
    """
    df = pd.read_excel(_path('附件3：园区6种场景的风电标幺功率表.xlsx'))
    cols = ['时段'] + [f'场景{i}' for i in range(1, 7)]
    df.columns = cols
    return df

def load_solar_scenarios() -> pd.DataFrame:
    """加载附件4: 4种光伏场景标幺功率

    Returns:
        DataFrame with columns ['时段', '场景1', ..., '场景4']
    """
    df = pd.read_excel(_path('附件4：园区4种场景的光伏标幺功率表.xlsx'))
    cols = ['时段'] + [f'场景{i}' for i in range(1, 5)]
    df.columns = cols
    return df

# 预加载数据 (模块级缓存, 只读一次Excel)
_df_load     = load_load_profile()
_df_typical  = load_typical_wind_solar()
_df_wind_sc  = load_wind_scenarios()
_df_solar_sc = load_solar_scenarios()

# =============================================================================
# 便捷数据访问函数
# =============================================================================

def get_load_profile() -> np.ndarray:
    """返回24小时常规电负荷实际功率 (MW)

    Returns:
        np.array shape (24,), 单位 MW
    """
    pu = _df_load['标幺功率'].values.astype(float)
    return pu * LOAD_PEAK


def get_typical_wind() -> np.ndarray:
    """返回24小时典型日风电实际功率 (MW)

    Returns:
        np.array shape (24,), 单位 MW
    """
    pu = _df_typical['风电标幺值'].values.astype(float)
    return pu * WIND_RATED


def get_typical_solar() -> np.ndarray:
    """返回24小时典型日光伏实际功率 (MW)

    Returns:
        np.array shape (24,), 单位 MW
    """
    pu = _df_typical['光伏标幺值'].values.astype(float)
    return pu * SOLAR_RATED


def get_wind_profile(scenario_id: int) -> np.ndarray:
    """返回指定风电场景的24小时实际功率 (MW)

    Args:
        scenario_id: 0-5 (对应风电场景1-6)

    Returns:
        np.array shape (24,), 单位 MW
    """
    col = f'场景{scenario_id + 1}'
    pu = _df_wind_sc[col].values.astype(float)
    return pu * WIND_RATED


def get_solar_profile(scenario_id: int) -> np.ndarray:
    """返回指定光伏场景的24小时实际功率 (MW)

    Args:
        scenario_id: 0-3 (对应光伏场景1-4)

    Returns:
        np.array shape (24,), 单位 MW
    """
    col = f'场景{scenario_id + 1}'
    pu = _df_solar_sc[col].values.astype(float)
    return pu * SOLAR_RATED


def get_scenario_wind_solar(scenario_id: int) -> tuple:
    """返回指定组合场景的风光24小时实际功率 (MW)

    Args:
        scenario_id: 0-23 (24种风光组合场景)

    Returns:
        (wind_array, solar_array), 各为 (24,) np.array, 单位 MW
    """
    w, s = SCENARIO_INDEX[scenario_id]
    return get_wind_profile(w), get_solar_profile(s)


# =============================================================================
# 绿电指标惩罚系数 (¥/kWh)
# 依据: 综合调研报告 Section 4.4 + 全国碳市场碳价(~80¥/tCO₂) + 绿证均价(~7.76¥/MWh)
# slack变量单位为MWh, utils.py中已×1000转换为kWh
# =============================================================================

GREEN_PENALTY_SELF_USE = 0.5   # ¥/kWh 自发自用率缺口 (依据: 网购电价 0.34-0.80, 取中值)
GREEN_PENALTY_GREEN_RATE = 0.3 # ¥/kWh 绿电比例缺口 (依据: 绿证+碳价 ≈ 0.07 + 绿电溢价 ≈ 0.3)
GREEN_PENALTY_FEED_RATE = 1.5  # ¥/kWh 上网比例超标 (依据: 赵昊天2026 δ=1.5)


# =============================================================================
# 自检
# =============================================================================

if __name__ == '__main__':
    print(f"WIND_RATED  = {WIND_RATED}")
    print(f"ALKEL_RATED = {ALKEL_RATED}")
    print(f"get_price(10) = {get_price(10)}")
    print(f"Grid feed-in  = {GRID_FEEDIN_PRICE}")
    print(f"Scenarios     = {N_SCENARIOS}")
    print(f"Production    = {PRODUCTION_LEVELS}")
    print(f"load_profile  = {get_load_profile()[:3]}...")
    print(f"wind_sc0      = {get_wind_profile(0)[:3]}...")
    print(f"solar_sc0     = {get_solar_profile(0)[:3]}...")
    print(f"scenario(0) w = {get_scenario_wind_solar(0)[0][:3]}...")
    print(f"scenario(0) s = {get_scenario_wind_solar(0)[1][:3]}...")
