"""
utils.py — 电工杯A题 共享计算函数库

提供功率平衡校验、绿电指标计算、成本核算、场景遍历等基础工具，
供 q1.py ~ q5.py 五个问题脚本共同调用。

依赖: numpy, config
"""

import numpy as np
from config import (
    WIND_LCOE, SOLAR_LCOE,
    STORAGE_INVESTMENT, STORAGE_LIFETIME,
    NH3_INVESTMENT, NH3_LIFETIME,
    WIND_LIFETIME, SOLAR_LIFETIME,
    ALKEL_LIFETIME, PEMEL_LIFETIME,
    GRID_FEEDIN_PRICE,
    STORAGE_OM, NH3_OM, ALKEL_OM, PEMEL_OM,
    SCENARIO_INDEX,
    PRODUCTION_LEVELS,
    get_price,
)

from pulp import LpVariable, lpSum

# =============================================================================
# 功率平衡
# =============================================================================

def power_balance(P_wind: float, P_solar: float, P_buy: float,
                  P_sell: float, P_el: float, P_nh3: float,
                  P_load: float) -> float:
    """计算某一时刻的净功率不平衡量 (MW)。

    功率平衡方程:
        P_wind + P_solar + P_buy = P_sell + P_el + P_nh3 + P_load

    Args:
        P_wind:  风电发电功率 (MW)
        P_solar: 光伏发电功率 (MW)
        P_buy:   网购电功率 (MW)
        P_sell:  上网电功率 (MW)
        P_el:    电解槽用电功率 (MW, 碱性+PEM合计)
        P_nh3:   合成氨装置用电功率 (MW)
        P_load:  常规电负荷功率 (MW)

    Returns:
        净功率不平衡量 (MW)。理想情况 ~0，正值表示供给过剩，负值表示供给不足。
    """
    supply = P_wind + P_solar + P_buy
    demand = P_sell + P_el + P_nh3 + P_load
    return supply - demand


# =============================================================================
# 绿电指标计算
# =============================================================================

def calc_green_indicators(total_gen: float, total_load: float,
                          grid_buy: float, grid_sell: float) -> dict:
    """计算三项绿电直连项目核心指标。

    依据《关于有序推动绿电直连发展有关事项的通知》公式:
        - 新能源自发自用电量占比 = (总用电量 - 上网电量 - 网购电量) / 新能源发电量
        - 总用电量绿电比例         = (新能源发电量 - 上网电量) / 总用电量
        - 新能源上网电量比例       = 上网电量 / 新能源发电量

    Args:
        total_gen:  新能源总发电量 (风电+光伏, kWh 或 MWh)
        total_load: 园区总用电量 (含电解槽+合成氨+常规负荷, kWh 或 MWh)
        grid_buy:   网购电量 (kWh 或 MWh)
        grid_sell:  上网电量 (kWh 或 MWh)

    Returns:
        dict with keys:
            'self_use_rate':   新能源自发自用电量占比 (> 0.60 合格)
            'green_rate':      总用电量绿电比例 (> 0.30 合格)
            'grid_feed_rate':  新能源上网电量比例 (< 0.20 合格)
        若分母为 0，对应指标返回 0。
    """
    self_use_rate = ((total_load - grid_sell - grid_buy) / total_gen
                     if total_gen > 0 else 0.0)
    green_rate = ((total_gen - grid_sell) / total_load
                  if total_load > 0 else 0.0)
    grid_feed_rate = (grid_sell / total_gen
                      if total_gen > 0 else 0.0)

    return {
        'self_use_rate': self_use_rate,
        'green_rate': green_rate,
        'grid_feed_rate': grid_feed_rate,
    }


# =============================================================================
# 绿电指标合规检查
# =============================================================================

# 绿电直连项目指标阈值 (来自赛题附件 / 政策文件)
SELF_USE_THRESHOLD = 0.60   # 自发自用率 > 60%
GREEN_THRESHOLD    = 0.30   # 绿电比例 > 30%
FEED_THRESHOLD     = 0.20   # 上网比例 < 20%


def check_green_requirements(indicators: dict) -> tuple:
    """检查三项绿电指标是否满足政策要求。

    Args:
        indicators: calc_green_indicators() 返回的字典，
                    包含 'self_use_rate', 'green_rate', 'grid_feed_rate'

    Returns:
        (pass_self_use, pass_green, pass_feed) 三个布尔值。
        - pass_self_use: self_use_rate > 0.60
        - pass_green:    green_rate > 0.30
        - pass_feed:     grid_feed_rate < 0.20
    """
    pass_self_use = indicators['self_use_rate'] > SELF_USE_THRESHOLD
    pass_green    = indicators['green_rate'] > GREEN_THRESHOLD
    pass_feed     = indicators['grid_feed_rate'] < FEED_THRESHOLD
    return (pass_self_use, pass_green, pass_feed)


# =============================================================================
# 场景分类
# =============================================================================

def classify_scenario(indicators: dict) -> str:
    """根据绿电指标对场景进行分类。

    Args:
        indicators: calc_green_indicators() 返回的字典

    Returns:
        - "全满足":   三项指标全部合格
        - "部分满足": 至少一项合格但不全合格
        - "全不满足": 三项指标全不合格
    """
    p1, p2, p3 = check_green_requirements(indicators)
    if p1 and p2 and p3:
        return "全满足"
    elif not p1 and not p2 and not p3:
        return "全不满足"
    else:
        return "部分满足"


# =============================================================================
# 吨氨成本计算
# =============================================================================

def calc_ton_ammonia_cost(nh3_output_tons: float,
                          grid_buy_kwh,
                          grid_sell_kwh: float = 0.0,
                          grid_buy_cost: float = 0.0,
                          grid_sell_revenue: float = 0.0,
                          wind_gen_kwh: float = 0.0,
                          solar_gen_kwh: float = 0.0,
                          equipment_depreciation: float = 0.0,
                          el_om_cost: float = 0.0,
                          nh3_om_cost: float = 0.0,
                          p_alk_mw=None,
                          p_pem_mw=None,
                          p_nh3_mw=None,
                          **kwargs,
                          ) -> float:
    """计算吨氨生产成本 (yuan/ton NH3)。

    总成本构成:
        购电费 + 风电度电成本 + 光伏度电成本
        + 电解槽运维费(ALK/PEM独立) + 合成氨运维费 + 设备折旧
        - 售电收入

    吨氨成本 = 总成本 / 氨产量

    Args:
        nh3_output_tons:      合成氨日产量 (tons)
        grid_buy_kwh:         网购电量。可为标量 (总 kWh) 或
                              array-like (24 小时逐时 kWh)。
                              array 时自动按分时电价累加购电费。
        grid_sell_kwh:        上网电量 (总 kWh)
        grid_buy_cost:        购电费 (yuan)。grid_buy_kwh 为标量时使用；
                              array 时忽略此参数（内部重算）。
        grid_sell_revenue:    售电收入 (yuan) = grid_sell_kwh × 上网电价
        wind_gen_kwh:         风电总发电量 (kWh)
        solar_gen_kwh:        光伏总发电量 (kWh)
        equipment_depreciation: 日设备折旧费 (yuan)
        el_om_cost:           [v1兼容] 电解槽运维费合计 (yuan)
        nh3_om_cost:          [v1兼容] 合成氨装置运维费 (yuan)
        p_alk_mw:             [v2] 24h ALK功率数组 (MW), 自动计算OM
        p_pem_mw:             [v2] 24h PEM功率数组 (MW), 自动计算OM
        p_nh3_mw:             [v2] 24h NH3功率数组 (MW), 自动计算OM

    Returns:
        吨氨成本 (yuan/ton NH3)。若 nh3_output_tons <= 0 返回 inf。
    """
    if nh3_output_tons <= 0:
        return float('inf')

    # 若提供逐时刻功率数组, 从数组计算拆分运维费 (v2)
    if p_alk_mw is not None and p_pem_mw is not None and p_nh3_mw is not None:
        el_om_cost = (np.sum(p_alk_mw) * 1000 * ALKEL_OM
                      + np.sum(p_pem_mw) * 1000 * PEMEL_OM)
        nh3_om_cost = np.sum(p_nh3_mw) * 1000 * NH3_OM

    # 网购电成本: 若传入逐时数组则按分时电价计算，否则用标量
    if isinstance(grid_buy_kwh, (list, np.ndarray)):
        grid_buy_arr = np.asarray(grid_buy_kwh, dtype=float)
        grid_buy_cost = float(sum(
            grid_buy_arr[t] * get_price(t)
            for t in range(len(grid_buy_arr))
        ))
    else:
        grid_buy_cost = float(grid_buy_cost)

    # 风光度电成本
    wind_cost  = float(wind_gen_kwh)  * WIND_LCOE
    solar_cost = float(solar_gen_kwh) * SOLAR_LCOE

    total_cost = (grid_buy_cost + wind_cost + solar_cost
                  + float(el_om_cost) + float(nh3_om_cost)
                  + float(equipment_depreciation)
                  - float(grid_sell_revenue))

    return total_cost / float(nh3_output_tons)


# =============================================================================
# 设备折旧计算
# =============================================================================

def calc_equipment_depreciation(investment_yuan: float,
                                lifetime_years: float,
                                annual_production_tons: float
                                ) -> float:
    """按直线折旧法计算每吨氨的设备折旧费 (yuan/ton)。

    公式:
        年折旧额 = 投资额 / 使用寿命
        吨氨折旧 = 年折旧额 / 年产量

    Args:
        investment_yuan:        设备总投资 (yuan)
        lifetime_years:         设备设计寿命 (年)
        annual_production_tons: 年氨产量 (tons)

    Returns:
        每吨氨折旧费 (yuan/ton)。若 annual_production_tons <= 0 返回 0。
    """
    if annual_production_tons <= 0:
        return 0.0
    annual_depreciation = investment_yuan / lifetime_years
    return annual_depreciation / annual_production_tons


# =============================================================================
# 场景遍历
# =============================================================================

def generate_all_scenarios() -> list:
    """生成全部 24 种风光出力组合场景索引。

    风电 6 种场景（索引 0-5）× 光伏 4 种场景（索引 0-3）→ 24 种组合。

    Returns:
        list of (wind_id, solar_id) tuples, 共 24 个。
        顺序: 风电 0 配光伏 0,1,2,3 → 风电 1 配光伏 0,1,2,3 → ...
    """
    return list(SCENARIO_INDEX)


# =============================================================================
# 格式化与换算工具
# =============================================================================

def format_hour(h: int) -> str:
    """将整型小时数格式化为 "HH:00" 字符串。

    Args:
        h: 小时 (0-23)

    Returns:
        格式化的时间字符串，如 0 → "00:00", 8 → "08:00", 23 → "23:00"

    Raises:
        ValueError: 若 h 不在 [0, 23] 范围内。
    """
    if not (0 <= h <= 23):
        raise ValueError(f"小时必须在 0-23 之间，收到: {h}")
    return f"{h:02d}:00"


def calc_annual_from_daily(daily_value: float,
                           days_per_scenario: int = 15,
                           n_scenarios: int = 24) -> float:
    """将日指标按场景数外推为年指标。

    假设每年按 24 种风光场景各运行 days_per_scenario 天。

    Args:
        daily_value:        某一场景下的日指标值（如日成本、日产量等）
        days_per_scenario:  每种场景的年均运行天数，默认 15 天
        n_scenarios:        场景总数，默认 24

    Returns:
        年化指标值 = daily_value × days_per_scenario × n_scenarios
    """
    return daily_value * days_per_scenario * n_scenarios


# =============================================================================
# 日折旧总额计算
# =============================================================================

def calc_total_depreciation(daily_output_tons: float, annual_operating_days: int = 360) -> float:
    """计算合成氨装置日折旧总额 (yuan)。
    
    Args:
        daily_output_tons: 日氨产量 (tons)
        annual_operating_days: 年运行天数, 默认 360
    Returns:
        日折旧费 (yuan), 按日产量比例分摊
    """
    annual_production = daily_output_tons * annual_operating_days
    dep_per_ton = calc_equipment_depreciation(NH3_INVESTMENT, NH3_LIFETIME, annual_production)
    return dep_per_ton * daily_output_tons


# =============================================================================
# 绿电指标 + 爬坡率惩罚 (MILP辅助函数)
# =============================================================================

def add_green_penalty_to_problem(prob, total_load_expr, total_gen,
                                  sell_sum, buy_sum, config_module):
    """为MILP问题添加绿电指标不达标的线性惩罚项。

    向问题中引入3个松弛变量 (>=0)，分别对应三项绿电指标的缺口：
      - Slack_SelfUse:  自发自用率 < 60% 的缺口
      - Slack_GreenRate: 绿电比例 < 30% 的缺口
      - Slack_FeedRate:  上网比例 > 20% 的超标量

    约束:
      Slack_SelfUse   >= 0.60 * total_gen - (total_load - sell_sum - buy_sum)
      Slack_GreenRate >= 0.30 * total_load - (total_gen - sell_sum)
      Slack_FeedRate  >= sell_sum - 0.20 * total_gen

    Args:
        prob:            LpProblem 实例
        total_load_expr: LpAffineExpression, 园区总用电量 (MW·h)
        total_gen:       float, 新能源总发电量 (MW·h, 固定)
        sell_sum:        LpAffineExpression, 上网电量之和 (MW·h)
        buy_sum:         LpAffineExpression, 网购电量之和 (MW·h)
        config_module:   config 模块 (提供惩罚系数)

    Returns:
        LpAffineExpression: 绿电惩罚项 (元), 可直接加入目标函数
    """
    s_self = LpVariable("Slack_SelfUse", lowBound=0)
    s_green = LpVariable("Slack_GreenRate", lowBound=0)
    s_feed = LpVariable("Slack_FeedRate", lowBound=0)

    prob += s_self >= 0.60 * total_gen - (total_load_expr - sell_sum - buy_sum), \
        "GreenSlack_SelfUse"
    prob += s_green >= 0.30 * total_load_expr - (total_gen - sell_sum), \
        "GreenSlack_GreenRate"
    prob += s_feed >= sell_sum - 0.20 * total_gen, \
        "GreenSlack_FeedRate"

    # 惩罚 = Σ(系数 ¥/kWh × 松弛量 MWh × 1000 kWh/MWh)
    penalty = 1000 * (config_module.GREEN_PENALTY_SELF_USE * s_self
                      + config_module.GREEN_PENALTY_GREEN_RATE * s_green
                      + config_module.GREEN_PENALTY_FEED_RATE * s_feed)
    return penalty


def add_ramp_penalty_to_problem(prob, p_vars, n_hours, ramp_coeff):
    """为MILP问题添加设备功率爬坡率的L1惩罚项。

    通过辅助变量将 |p_t - p_{t-1}| 线性化:
      p_t - p_{t-1} = delta_up[t] - delta_down[t]
      delta_up[t], delta_down[t] >= 0

    惩罚项 = ramp_coeff * sum(delta_up[t] + delta_down[t])

    Args:
        prob:       LpProblem 实例
        p_vars:     list[LpVariable], 长度 n_hours, 各时段设备功率
        n_hours:    int, 调度时段数 (通常 24)
        ramp_coeff: float, 爬坡惩罚系数 (¥/MW)

    Returns:
        tuple (ramp_expr, delta_up_vars, delta_down_vars):
            - ramp_expr:       LpAffineExpression, 爬坡惩罚项
            - delta_up_vars:   list[LpVariable], 向上爬坡辅助变量
            - delta_down_vars: list[LpVariable], 向下爬坡辅助变量
    """
    delta_up = []
    delta_down = []
    ramp_terms = []

    for t in range(1, n_hours):
        du = LpVariable(f"RampUp_{t}", lowBound=0)
        dd = LpVariable(f"RampDown_{t}", lowBound=0)
        delta_up.append(du)
        delta_down.append(dd)

        prob += p_vars[t] - p_vars[t-1] == du - dd, f"RampDef_{t}"
        ramp_terms.append(du + dd)

    ramp_expr = ramp_coeff * lpSum(ramp_terms)
    return ramp_expr, delta_up, delta_down


# =============================================================================
# H2平衡 / NH3爬坡 / 最小启停约束 (学术论文集成)
# =============================================================================

def build_h2_balance_constraint(prob, p_alk_list, p_pem_list, p_nh3_list, n_hours):
    """逐时+逐日氢气质量平衡: 产H2 >= 耗H2(逐时), 总产H2 == 总耗H2(逐日).

    逐时: p_alk*η_alk + p_pem*η_pem >= p_nh3 * 400  (允许小时级缓冲)
    逐日: Σ产H2 == Σ耗H2                       (禁止超额产氢作为假负载)
    """
    import config as _cfg  # read at call-time for sensitivity monkey-patching
    H2_DEMAND_PER_MW = 200 * 0.002 * 1000
    for t in range(n_hours):
        prob += (p_alk_list[t] * _cfg.ALK_H2_PER_MW + p_pem_list[t] * _cfg.PEM_H2_PER_MW
                 >= p_nh3_list[t] * H2_DEMAND_PER_MW), f"H2Bal_{t}"
    # 逐日等式: 禁止利用电解槽作为假负载消耗过剩风电
    from pulp import lpSum
    prob += (lpSum(p_alk_list) * _cfg.ALK_H2_PER_MW + lpSum(p_pem_list) * _cfg.PEM_H2_PER_MW
             == lpSum(p_nh3_list) * H2_DEMAND_PER_MW), "H2Bal_Daily"


def build_nh3_ramp_constraint(prob, p_nh3_list, n_hours, nh3_rated_mw, ramp_rate=0.20):
    """NH3爬坡率硬约束. 效率变化≤20%/h.

    |P_nh3[t] - P_nh3[t-1]| ≤ ramp_rate × nh3_rated_mw

    72t/d: 0.20 × 1.5 MW = 0.30 MW/h
    36t/d: 0.20 × 0.75 MW = 0.15 MW/h
    """
    ramp_limit = ramp_rate * nh3_rated_mw
    for t in range(1, n_hours):
        delta = p_nh3_list[t] - p_nh3_list[t-1]
        prob += delta >= -ramp_limit, f"NH3RampDn_{t}"
        prob += delta <=  ramp_limit, f"NH3RampUp_{t}"


def build_min_up_down_constraint(prob, z_list, n_hours, min_up=4, min_down=2):
    """最小启停时间约束. 赵昊天2025框架. z[t]=1运行, z[t]=0停机."""
    for t in range(1, n_hours):
        for k in range(t + 1, min(t + min_up, n_hours)):
            prob += z_list[k] >= z_list[t] - z_list[t-1], f"MinUp_{t}_{k}"
        for k in range(t + 1, min(t + min_down, n_hours)):
            prob += 1 - z_list[k] >= z_list[t-1] - z_list[t], f"MinDn_{t}_{k}"


# =============================================================================
# =============================================================================
# 自检
# =============================================================================
if __name__ == '__main__':
    # --- 测试绿电指标计算 ---
    print("=" * 60)
    print("1. 绿电指标计算测试")
    print("=" * 60)
    # total_gen=1000, total_load=800, grid_buy=200, grid_sell=100
    r = calc_green_indicators(1000, 800, 200, 100)
    print(f"   输入: total_gen=1000, total_load=800, grid_buy=200, grid_sell=100")
    print(f"   结果: {r}")
    # 手工验算:
    #   self_use_rate = (800 - 100 - 200) / 1000 = 500/1000 = 0.50
    #   green_rate    = (1000 - 100) / 800       = 900/800 = 1.125
    #   grid_feed_rate = 100 / 1000              = 0.10
    assert abs(r['self_use_rate'] - 0.50) < 1e-9, f"self_use_rate 期望 0.50, 得到 {r['self_use_rate']}"
    assert abs(r['green_rate'] - 1.125) < 1e-9, f"green_rate 期望 1.125, 得到 {r['green_rate']}"
    assert abs(r['grid_feed_rate'] - 0.10) < 1e-9, f"grid_feed_rate 期望 0.10, 得到 {r['grid_feed_rate']}"
    print("   [OK] 通过")

    # --- 测试合规检查 ---
    print()
    print("=" * 60)
    print("2. 合规检查测试")
    print("=" * 60)
    # self_use=0.50 (<0.60 fail), green=1.125 (>0.30 pass), feed=0.10 (<0.20 pass)
    p1, p2, p3 = check_green_requirements(r)
    print(f"   pass_self_use={p1}, pass_green={p2}, pass_feed={p3}")
    assert p1 == False and p2 == True and p3 == True, "合规检查预期不符"
    print("   [OK] 通过")

    # --- 测试场景分类 ---
    print()
    print("=" * 60)
    print("3. 场景分类测试")
    print("=" * 60)
    label = classify_scenario(r)
    print(f"   分类结果: {label}")
    assert label == "部分满足", f"期望 '部分满足', 得到 '{label}'"
    print("   [OK] 通过")

    # 全满足测试
    r_all_pass = {'self_use_rate': 0.70, 'green_rate': 0.40, 'grid_feed_rate': 0.10}
    assert classify_scenario(r_all_pass) == "全满足"
    print("   全满足场景: [OK]")

    # 全不满足测试
    r_all_fail = {'self_use_rate': 0.50, 'green_rate': 0.20, 'grid_feed_rate': 0.30}
    assert classify_scenario(r_all_fail) == "全不满足"
    print("   全不满足场景: [OK]")

    # --- 测试功率平衡 ---
    print()
    print("=" * 60)
    print("4. 功率平衡测试")
    print("=" * 60)
    # P_wind=15, P_solar=10, P_buy=5, P_sell=2, P_el=18, P_nh3=3, P_load=7
    # supply = 15+10+5=30, demand = 2+18+3+7=30, imbalance=0
    imb = power_balance(15, 10, 5, 2, 18, 3, 7)
    print(f"   不平衡量: {imb} (期望 0)")
    assert abs(imb) < 1e-9, f"功率不平衡量应为 0，得到 {imb}"
    print("   [OK] 通过")

    # --- 测试吨氨成本 ---
    print()
    print("=" * 60)
    print("5. 吨氨成本测试")
    print("=" * 60)
    # 简易场景: 日产 36 吨, 网购 1000 kWh (标量, 均价 0.60 yuan/kWh)
    cost = calc_ton_ammonia_cost(
        nh3_output_tons=36.0,
        grid_buy_kwh=1000.0,       # 标量
        grid_sell_kwh=200.0,
        grid_buy_cost=600.0,       # 1000 * 0.60
        grid_sell_revenue=200.0 * GRID_FEEDIN_PRICE,  # 200 * 0.3779 ≈ 75.58
        wind_gen_kwh=5000.0,
        solar_gen_kwh=3000.0,
        el_om_cost=200.0,
        nh3_om_cost=50.0,
        equipment_depreciation=100.0,
    )
    # 手工验算:
    #   wind_cost  = 5000 * 0.15 = 750
    #   solar_cost = 3000 * 0.12 = 360
    #   total_cost = 600 + 750 + 360 + 200 + 50 + 100 - 75.58 = 1984.42
    #   ton_cost = 1984.42 / 36 ≈ 55.12
    expected = (600 + 5000*0.15 + 3000*0.12 + 200 + 50 + 100 - 200*GRID_FEEDIN_PRICE) / 36
    print(f"   吨氨成本: {cost:.4f} yuan/ton (期望 ≈ {expected:.4f})")
    assert abs(cost - expected) < 1e-6, f"吨氨成本不匹配"
    print("   [OK] 通过")

    # --- 测试设备折旧 ---
    print()
    print("=" * 60)
    print("6. 设备折旧测试")
    print("=" * 60)
    # 储能: 1000 yuan/kWh, 15 年, 年产 10000 tons
    dep_storage = calc_equipment_depreciation(STORAGE_INVESTMENT, STORAGE_LIFETIME, 10000)
    print(f"   储能折旧 (1000 / 15 / 10000): {dep_storage:.6f} yuan/ton")
    assert abs(dep_storage - 1000/15/10000) < 1e-9
    print("   [OK] 通过")

    # --- 测试场景遍历 ---
    print()
    print("=" * 60)
    print("7. 场景遍历测试")
    print("=" * 60)
    scenarios = generate_all_scenarios()
    print(f"   场景总数: {len(scenarios)} (期望 24)")
    print(f"   前 5 个: {scenarios[:5]}")
    print(f"   后 5 个: {scenarios[-5:]}")
    assert len(scenarios) == 24, "场景数应为 24"
    assert scenarios[0] == (0, 0), "第一个应为 (0,0)"
    assert scenarios[-1] == (5, 3), "最后一个应为 (5,3)"
    print("   [OK] 通过")

    # --- 测试格式化 ---
    print()
    print("=" * 60)
    print("8. 格式化测试")
    print("=" * 60)
    print(f"   format_hour(0):  {format_hour(0)}")
    print(f"   format_hour(8):  {format_hour(8)}")
    print(f"   format_hour(23): {format_hour(23)}")
    assert format_hour(0) == "00:00"
    assert format_hour(8) == "08:00"
    assert format_hour(23) == "23:00"
    # 越界测试
    try:
        format_hour(24)
        assert False, "应抛出 ValueError"
    except ValueError:
        print("   format_hour(24) 正确抛出 ValueError")
    print("   [OK] 通过")

    # --- 测试年化换算 ---
    print()
    print("=" * 60)
    print("9. 年化换算测试")
    print("=" * 60)
    annual = calc_annual_from_daily(1000.0)
    print(f"   日值 1000 → 年化: {annual} (期望 360000)")
    assert abs(annual - 360000.0) < 1e-6
    print("   [OK] 通过")

    # --- 测试数组形式的 grid_buy_kwh (分时电价) ---
    print()
    print("=" * 60)
    print("10. 分时电价网购成本测试")
    print("=" * 60)
    # 24 小时, 每小时网购 100 kWh, 求总购电费
    hourly_buy = np.full(24, 100.0)
    cost_arr = calc_ton_ammonia_cost(
        nh3_output_tons=36.0,
        grid_buy_kwh=hourly_buy,   # array
        grid_sell_kwh=0.0,
        grid_buy_cost=0.0,         # 会被忽略 (array 模式)
        grid_sell_revenue=0.0,
        wind_gen_kwh=0.0,
        solar_gen_kwh=0.0,
        el_om_cost=0.0,
        nh3_om_cost=0.0,
        equipment_depreciation=0.0,
    )
    # 手工算 TOU 购电费:
    #   低谷(8h): 8 * 100 * 0.3424 = 273.92
    #   平段(8h): 8 * 100 * 0.6074 = 485.92
    #   高峰(8h): 8 * 100 * 0.8024 = 641.92
    #   总: 273.92 + 485.92 + 641.92 = 1401.76
    #   吨氨 = 1401.76 / 36 ≈ 38.9378
    expected_arr = (8*100*0.3424 + 8*100*0.6074 + 8*100*0.8024) / 36
    print(f"   吨氨成本 (array): {cost_arr:.4f} (期望 ≈ {expected_arr:.4f})")
    assert abs(cost_arr - expected_arr) < 1e-6, f"分时电价计算不匹配"
    print("   [OK] 通过")

    print()
    print("=" * 60)
    print("全部自检通过 [OK]")
    print("=" * 60)
