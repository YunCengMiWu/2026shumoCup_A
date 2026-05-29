"""
绿电直连三大指标 + 吨氨成本 + 年度统计
=========================================
单一模块，整合原 indicators.py 和 economics.py 的所有计算逻辑。
所有公式严格遵循 最终版公式.md。
常量从 utils.constants 统一导入。
"""

import numpy as np
from .constants import (PRICE_PEAK, PRICE_FLAT, PRICE_VALLEY, FEED_IN_PRICE,
                        WIND_COST, PV_COST,
                        ALK_OM_COST, PEM_OM_COST,
                        NH3_OM_COST, NH3_DEPR_PER_TON,
                        DELTA_T, HOURS_PER_DAY, DAYS_PER_SCENARIO, TOTAL_DAYS,
                        RATED_DAILY_NH3, get_price)


# =============================================================================
# 1. 绿电直连三大指标 (最终版公式.md:39-51)
# =============================================================================

def compute_green_indicators(E_total, E_RE, E_sell, E_buy):
    """
    计算绿电直连三大指标。

    Parameters
    ----------
    E_total : float
        总用电量 (MWh)
    E_RE : float
        新能源总发电量 (MWh)
    E_sell : float
        上网电量 (MWh)
    E_buy : float
        网购电量 (MWh)

    Returns
    -------
    tuple[float, float, float]
        (eta_self, eta_green, eta_grid)

    Formulas (最终版公式.md:39-51)
    -------------------------------
    eta_self  = (E_total - E_sell - E_buy) / E_RE      [指标1, 要求 >60%]
    eta_green = (E_RE - E_sell) / E_total               [指标2, 要求 >30%]
    eta_grid  = E_sell / E_RE                           [指标3, 要求 <20%]

    Note: eta_self = 1 - 2*eta_grid (由功率平衡推导)
    """
    eta_self = (E_total - E_sell - E_buy) / E_RE if E_RE > 0 else 0.0
    eta_green = (E_RE - E_sell) / E_total if E_total > 0 else 0.0
    eta_grid = E_sell / E_RE if E_RE > 0 else 0.0
    return eta_self, eta_green, eta_grid


# =============================================================================
# 2. 指标达标检查 (最终版公式.md:39-51 的阈值)
# =============================================================================

def compute_compliance(eta_self, eta_green, eta_grid):
    """
    检查三大指标是否达标。

    阈值:
        eta_self  >= 0.60  (正指标)
        eta_green >= 0.30  (正指标)
        eta_grid  <= 0.20  (反指标)

    Parameters
    ----------
    eta_self : float
        自用率
    eta_green : float
        绿色比例
    eta_grid : float
        上网比例

    Returns
    -------
    dict
        包含指标值、margin (>0 表示达标)、各指标及总体是否达标。
    """
    return {
        'eta_self': eta_self,
        'eta_green': eta_green,
        'eta_grid': eta_grid,
        'margin_self': eta_self - 0.60,       # >0 = 达标
        'margin_green': eta_green - 0.30,     # >0 = 达标
        'margin_grid': 0.20 - eta_grid,       # >0 = 达标 (反指标)
        'self_compliant': eta_self >= 0.60,
        'green_compliant': eta_green >= 0.30,
        'grid_compliant': eta_grid <= 0.20,
        'all_compliant': (eta_self >= 0.60) and (eta_green >= 0.30) and (eta_grid <= 0.20)
    }


# =============================================================================
# 3. 吨氨成本 (最终版公式.md:73-84)
# =============================================================================

def compute_ton_cost(p_buy, p_sell, p_wind, p_pv, p_alk, p_pem, p_nh3, daily_nh3_tons):
    """
    计算吨氨综合成本 (元/吨)。

    Formula (最终版公式.md:73):
        C_ton = [ sum_t(P_buy(t)*c_buy(t) - P_sell(t)*c_sell)*1000*Delta_t
                  + C_RE + C_OM,H2 + C_OM,NH3 + C_dep,NH3 ] / M_NH3

    Parameters
    ----------
    p_buy : array-like of 24 floats
        网购电功率 (MW)
    p_sell : array-like of 24 floats
        上网电功率 (MW)
    p_wind : array-like of 24 floats
        风电发电功率 (MW)
    p_pv : array-like of 24 floats
        光伏发电功率 (MW)
    p_alk : array-like of 24 floats
        碱性电解槽功率 (MW)
    p_pem : array-like of 24 floats
        PEM电解槽功率 (MW)
    p_nh3 : array-like of 24 floats
        合成氨功率 (MW)
    daily_nh3_tons : float
        日制氨总量 (吨)

    Returns
    -------
    float
        吨氨成本 (元/吨), 无产出时返回 inf.
    """
    # 购电成本 - 售电收入 (MW -> kW * 1000)
    grid_cost = sum(
        p_buy[t] * get_price(t) * 1000 * DELTA_T -
        p_sell[t] * FEED_IN_PRICE * 1000 * DELTA_T
        for t in range(24)
    )

    # 风光发电成本: C_RE (最终版公式.md:81)
    re_cost = sum(
        p_wind[t] * WIND_COST * 1000 * DELTA_T +
        p_pv[t] * PV_COST * 1000 * DELTA_T
        for t in range(24)
    )

    # 制氢运维: C_OM,H2 (最终版公式.md:82)
    h2_om = sum(
        p_alk[t] * ALK_OM_COST * 1000 * DELTA_T +
        p_pem[t] * PEM_OM_COST * 1000 * DELTA_T
        for t in range(24)
    )

    # 合成氨运维: C_OM,NH3 (最终版公式.md:83)
    nh3_om = sum(p_nh3[t] * NH3_OM_COST * 1000 * DELTA_T for t in range(24))

    # 合成氨日折旧: C_dep,NH3 (最终版公式.md:84)
    nh3_depr = NH3_DEPR_PER_TON * daily_nh3_tons

    total_cost = grid_cost + re_cost + h2_om + nh3_om + nh3_depr
    return total_cost / daily_nh3_tons if daily_nh3_tons > 0 else float('inf')


# =============================================================================
# 4. 全年统计 (最终版公式.md:86-95)
# =============================================================================

def compute_annual_stats(scenario_results):
    """
    基于24种场景的日结果，推算全年统计量。

    Parameters
    ----------
    scenario_results : list of dict
        每个元素包含单日计算结果，至少需要:
        - 'daily_nh3_tons' : float, 日制氨量 (吨)
        可选 (用于扩展统计):
        - 'eta_self', 'eta_green', 'eta_grid' : float
        - 'ton_cost' : float

    Returns
    -------
    dict
        M_year  : 全年制氨总量 (吨)     [最终版公式.md:90]
        U_year  : 年平均产能利用率         [最终版公式.md:94]
        avg_daily_nh3 : 日均制氨量 (吨/天)
        n_scenarios   : 场景数 (应为24)
        avg_eta_self  : 年均 eta_self  (如数据可用)
        avg_eta_green : 年均 eta_green (如数据可用)
        avg_eta_grid  : 年均 eta_grid  (如数据可用)
        avg_ton_cost  : 年均吨氨成本 (元/吨) (如数据可用)
    """
    n = len(scenario_results)

    # 全年制氨总量: M_year = 15 * sum M_NH3,s  (最终版公式.md:90)
    total_nh3 = sum(r.get('daily_nh3_tons', 0.0) for r in scenario_results) * DAYS_PER_SCENARIO

    # 产能利用率: U_year = M_year / (72 * 360)  (最终版公式.md:94)
    utilization = total_nh3 / (RATED_DAILY_NH3 * TOTAL_DAYS)

    result = {
        'M_year': total_nh3,
        'U_year': utilization,
        'avg_daily_nh3': total_nh3 / TOTAL_DAYS,
        'n_scenarios': n,
    }

    # ---- 可选：各指标年均值 (如场景数据中包含则计算) ----
    if n > 0:
        if all('eta_self' in r for r in scenario_results):
            result['avg_eta_self'] = sum(r['eta_self'] for r in scenario_results) / n
        if all('eta_green' in r for r in scenario_results):
            result['avg_eta_green'] = sum(r['eta_green'] for r in scenario_results) / n
        if all('eta_grid' in r for r in scenario_results):
            result['avg_eta_grid'] = sum(r['eta_grid'] for r in scenario_results) / n
        if all('ton_cost' in r for r in scenario_results):
            result['avg_ton_cost'] = sum(r['ton_cost'] for r in scenario_results) / n

    return result


# =============================================================================
# 5. 便捷函数: 从一日计算结果中提取三指标
# =============================================================================

def compute_daily_stats(result):
    """
    从单日仿真结果 dict 中提取三指标，兼容原 indicators.py 的 compute_q1_indicators() 接口。

    Parameters
    ----------
    result : dict
        单日仿真结果，包含 'p_gen', 'p_load', 'p_sell', 'p_buy' (array of 24)

    Returns
    -------
    dict
        E_total, E_RE, E_sell, E_buy,
        eta_self, eta_green, eta_grid,
        margin_self, margin_green, margin_grid
    """
    E_RE = np.sum(result['p_gen']) * DELTA_T
    E_total = np.sum(result['p_load']) * DELTA_T
    E_sell = np.sum(result['p_sell']) * DELTA_T
    E_buy = np.sum(result['p_buy']) * DELTA_T

    eta_self, eta_green, eta_grid = compute_green_indicators(E_total, E_RE, E_sell, E_buy)

    return {
        'E_total': E_total,
        'E_RE': E_RE,
        'E_sell': E_sell,
        'E_buy': E_buy,
        'eta_self': eta_self,
        'eta_green': eta_green,
        'eta_grid': eta_grid,
        'margin_self': eta_self - 0.60,
        'margin_green': eta_green - 0.30,
        'margin_grid': 0.20 - eta_grid,
    }
