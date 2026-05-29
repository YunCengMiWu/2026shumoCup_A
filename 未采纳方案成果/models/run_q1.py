"""Q1: 典型日绿电直连指标评估（满负荷运行场景）
====================================================
"what-if" reasoning: ALK=10MW, PEM=10MW, NH3=0.75MW all 24 hours.
Pure deterministic calculation — NO optimization (no PuLP / MILP / LP).
Computes three green energy indicators per 最终版公式.md and checks
policy compliance with margins.

Run standalone:  python models/run_q1.py
"""
import sys
import os
import numpy as np
import pandas as pd
import traceback

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    WIND_CAPACITY, PV_CAPACITY, BASE_LOAD_PEAK,
    ALK_RATED_POWER, PEM_RATED_POWER, NH3_RATED_POWER,
    NH3_POWER_RATE, DELTA_T, HOURS_PER_DAY, DAYS_PER_SCENARIO,
)
from utils.data_loader import get_typical_day, get_base_load, get_all_scenarios
from utils.indicators import compute_green_indicators, compute_compliance, compute_ton_cost


# ── Reusable helper ──────────────────────────────────────────────────────────
def _compute_daily_scenario(wind_pu, pv_pu, base_pu, verbose=True):
    """Core computation for one daily scenario.

    Parameters
    ----------
    wind_pu : np.ndarray[24]
        Wind power per-unit curve.
    pv_pu : np.ndarray[24]
        PV power per-unit curve.
    base_pu : np.ndarray[24]
        Base load per-unit curve (peak = BASE_LOAD_PEAK MW).
    verbose : bool
        If True, print Chinese-language summary to console (default).

    Returns
    -------
    result : dict[str, np.ndarray]
        Hourly time series (24 elements each):
        p_wind, p_pv, p_base, p_alk, p_pem, p_nh3,
        p_sell, p_buy, p_gen, p_load
    indicators : dict
        Energy totals, green indicators, ton-cost, compliance.
    """
    # 1 ── Convert pu → MW ────────────────────────────────────────────────
    p_wind = wind_pu * WIND_CAPACITY        # 40 MW rated
    p_pv   = pv_pu   * PV_CAPACITY          # 64 MW rated
    p_base = base_pu  * BASE_LOAD_PEAK      # 6 MW peak

    # 2 ── Full-load industrial loads (constant 24 h) ─────────────────────
    p_alk = np.full(HOURS_PER_DAY, ALK_RATED_POWER)   # 10 MW × 24 h
    p_pem = np.full(HOURS_PER_DAY, PEM_RATED_POWER)   # 10 MW × 24 h
    p_nh3 = np.full(HOURS_PER_DAY, NH3_RATED_POWER)   # 0.75 MW × 24 h

    # 3 ── Aggregates ─────────────────────────────────────────────────────
    p_gen  = p_wind + p_pv                          # RE generation
    p_load = p_base + p_alk + p_pem + p_nh3         # Total consumption

    # 4 ── Grid exchange (buy/sell mutually exclusive per hour) ───────────
    delta  = p_gen - p_load
    p_buy  = np.maximum(0, p_load - p_gen)   # Buy when load > gen
    p_sell = np.maximum(0, p_gen - p_load)   # Sell when gen > load

    # 5 ── Energy totals (MWh) ────────────────────────────────────────────
    E_RE    = float(np.sum(p_gen))   * DELTA_T
    E_total = float(np.sum(p_load))  * DELTA_T
    E_sell  = float(np.sum(p_sell))  * DELTA_T
    E_buy   = float(np.sum(p_buy))   * DELTA_T

    # 6 ── Green indicators ───────────────────────────────────────────────
    eta_self, eta_green, eta_grid = compute_green_indicators(
        E_total, E_RE, E_sell, E_buy
    )

    # 7 ── Daily NH3 production (tons) ────────────────────────────────────
    #     NH3_POWER_RATE = 0.002 ton/(kW·h); p_nh3 in MW → ×1000 → kW
    daily_nh3 = float(np.sum(p_nh3)) * NH3_POWER_RATE * 1000 * DELTA_T

    # 8 ── Ton-ammonia cost ───────────────────────────────────────────────
    ton_cost = compute_ton_cost(p_buy, p_sell, p_wind, p_pv,
                                p_alk, p_pem, p_nh3, daily_nh3)

    # 9 ── Compliance ────────────────────────────────────────────────────
    compliance = compute_compliance(eta_self, eta_green, eta_grid)

    # 10 ── Package result & indicators ───────────────────────────────────
    result = {
        'p_wind': p_wind, 'p_pv': p_pv, 'p_base': p_base,
        'p_alk':  p_alk,  'p_pem': p_pem, 'p_nh3': p_nh3,
        'p_sell': p_sell, 'p_buy': p_buy,
        'p_gen':  p_gen,  'p_load': p_load,
    }
    indicators = {
        'E_RE':     E_RE,
        'E_total':  E_total,
        'E_sell':   E_sell,
        'E_buy':    E_buy,
        'eta_self':  eta_self,
        'eta_green': eta_green,
        'eta_grid':  eta_grid,
        'daily_nh3_tons': daily_nh3,
        'ton_cost':  ton_cost,
        'compliance': compliance,
    }

    # 11 ── Chinese-language summary (only for typical-day caller) ─────────
    if verbose:
        status_self  = "达标" if compliance['self_compliant']  else "不达标"
        status_green = "达标" if compliance['green_compliant'] else "不达标"
        status_grid  = "达标" if compliance['grid_compliant']  else "不达标"
        margin_sign  = lambda m: f"{m*100:+.1f}"

        print("=" * 60)
        print("Q1: 典型日绿电直连指标评估（满负荷运行）")
        print("=" * 60)
        print(f"新能源总发电量 E_RE:    {E_RE:.2f} MWh")
        print(f"总用电量 E_total:        {E_total:.2f} MWh")
        print(f"网购电量 E_buy:          {E_buy:.2f} MWh")
        print(f"上网电量 E_sell:         {E_sell:.2f} MWh")
        print(f"日制氨产量:              {daily_nh3:.2f} 吨")
        print(f"吨氨成本:                {ton_cost:.2f} 元/吨")
        print("-" * 40)
        print(f"自发自用率 η_self:  {eta_self*100:.1f}% [要求>60%] 裕度:{margin_sign(compliance['margin_self'])}% → {status_self}")
        print(f"绿电比例 η_green:   {eta_green*100:.1f}% [要求>30%] 裕度:{margin_sign(compliance['margin_green'])}% → {status_green}")
        print(f"上网比例 η_grid:    {eta_grid*100:.1f}% [要求<20%] 裕度:{margin_sign(compliance['margin_grid'])}% → {status_grid}")
        print(f"三项全达标:            {'是' if compliance['all_compliant'] else '否'}")
        print("=" * 60)

    return result, indicators


# ── Public API ───────────────────────────────────────────────────────────────
def run_q1():
    """Full-load Q1 power flow + green indicator + ton-cost evaluation.

    Returns
    -------
    result : dict[str, np.ndarray]
        Hourly time series (24 elements each):
        p_wind, p_pv, p_base, p_alk, p_pem, p_nh3,
        p_sell, p_buy, p_gen, p_load
    indicators : dict
        Energy totals, green indicators, ton-cost, compliance.
    """
    wind_pu, pv_pu = get_typical_day()
    base_pu = get_base_load()
    return _compute_daily_scenario(wind_pu, pv_pu, base_pu, verbose=True)


# ── 24-scenario iteration ────────────────────────────────────────────────────
def run_q1_all_scenarios():
    """Iterate all 24 wind×PV scenarios, classify, build DataFrame, export CSV.

    Returns
    -------
    pd.DataFrame
        26-row DataFrame: 24 scenarios + annual summary + weighted average.
    """
    # 1 ── Load data ──────────────────────────────────────────────────────
    scenarios = get_all_scenarios()   # list[dict] × 24
    base_pu = get_base_load()

    # 2 ── Loop over 24 scenarios ─────────────────────────────────────────
    rows = []
    for sc in scenarios:
        try:
            _, indicators = _compute_daily_scenario(
                sc['wind_pu'], sc['pv_pu'], base_pu, verbose=False
            )
        except Exception:
            traceback.print_exc()
            rows.append({
                '场景编号': sc['id'],
                '风电场景': sc['wind_scene'],
                '光伏场景': sc['pv_scene'],
                '典型日用电量(MWh)': np.nan,
                '新能源发电量(MWh)': np.nan,
                '网购电量(MWh)': np.nan,
                '上网电量(MWh)': np.nan,
                '新能源自发自用电量占比(%)': np.nan,
                '总用电量绿电比例(%)': np.nan,
                '新能源上网电量比例(%)': np.nan,
                '吨氨成本(元/吨)': np.nan,
                '自消纳达标': np.nan,
                '绿电比例达标': np.nan,
                '上网比例达标': np.nan,
                '自消纳裕度(百分点)': np.nan,
                '绿电裕度(百分点)': np.nan,
                '上网裕度(百分点)': np.nan,
                '三分类': '计算失败',
            })
            continue

        comp = indicators['compliance']
        compliant_count = sum([comp['self_compliant'], comp['green_compliant'], comp['grid_compliant']])
        if compliant_count == 3:
            category = '全满足'
        elif compliant_count >= 1:
            category = '部分满足'
        else:
            category = '全不满足'

        rows.append({
            '场景编号': sc['id'],
            '风电场景': sc['wind_scene'],
            '光伏场景': sc['pv_scene'],
            '典型日用电量(MWh)': round(indicators['E_total'], 2),
            '新能源发电量(MWh)': round(indicators['E_RE'], 2),
            '网购电量(MWh)': round(indicators['E_buy'], 2),
            '上网电量(MWh)': round(indicators['E_sell'], 2),
            '新能源自发自用电量占比(%)': round(indicators['eta_self'] * 100, 2),
            '总用电量绿电比例(%)': round(indicators['eta_green'] * 100, 2),
            '新能源上网电量比例(%)': round(indicators['eta_grid'] * 100, 2),
            '吨氨成本(元/吨)': round(indicators['ton_cost'], 2),
            '自消纳达标': comp['self_compliant'],
            '绿电比例达标': comp['green_compliant'],
            '上网比例达标': comp['grid_compliant'],
            '自消纳裕度(百分点)': round(comp['margin_self'] * 100, 2),
            '绿电裕度(百分点)': round(comp['margin_green'] * 100, 2),
            '上网裕度(百分点)': round(comp['margin_grid'] * 100, 2),
            '三分类': category,
        })

    # 3 ── Build DataFrame ────────────────────────────────────────────────
    columns = [
        '场景编号', '风电场景', '光伏场景',
        '典型日用电量(MWh)', '新能源发电量(MWh)', '网购电量(MWh)', '上网电量(MWh)',
        '新能源自发自用电量占比(%)', '总用电量绿电比例(%)', '新能源上网电量比例(%)', '吨氨成本(元/吨)',
        '自消纳达标', '绿电比例达标', '上网比例达标',
        '自消纳裕度(百分点)', '绿电裕度(百分点)', '上网裕度(百分点)',
        '三分类',
    ]
    df = pd.DataFrame(rows, columns=columns)

    # 4 ── Annual summary row (index 24) ──────────────────────────────────
    full_count   = int((df['三分类'] == '全满足').sum())
    partial_count = int((df['三分类'] == '部分满足').sum())
    fail_count    = int((df['三分类'] == '全不满足').sum())
    error_count   = int((df['三分类'] == '计算失败').sum())

    full_days   = full_count * DAYS_PER_SCENARIO
    partial_days = partial_count * DAYS_PER_SCENARIO
    fail_days    = fail_count * DAYS_PER_SCENARIO

    total_days = DAYS_PER_SCENARIO * len(scenarios)  # 15 × 24 = 360
    summary_text = (
        f"全满足:{full_count}场景({full_days}天,{full_days/total_days*100:.1f}%); "
        f"部分满足:{partial_count}场景({partial_days}天,{partial_days/total_days*100:.1f}%); "
        f"全不满足:{fail_count}场景({fail_days}天,{fail_days/total_days*100:.1f}%)"
    )
    if error_count > 0:
        summary_text += f"; 计算失败:{error_count}场景"

    summary_row = {col: np.nan for col in columns}
    summary_row['场景编号'] = '年度汇总'
    summary_row['三分类'] = summary_text
    df = pd.concat([df, pd.DataFrame([summary_row], columns=columns)], ignore_index=True)

    # 5 ── Weighted average row (index 25) ────────────────────────────────
    indicator_cols = [
        '典型日用电量(MWh)', '新能源发电量(MWh)', '网购电量(MWh)', '上网电量(MWh)',
        '新能源自发自用电量占比(%)', '总用电量绿电比例(%)', '新能源上网电量比例(%)', '吨氨成本(元/吨)',
        '自消纳裕度(百分点)', '绿电裕度(百分点)', '上网裕度(百分点)',
    ]
    weighted_row = {col: np.nan for col in columns}
    weighted_row['场景编号'] = '年度加权平均'
    weighted_row['三分类'] = '年度加权平均'
    for col in indicator_cols:
        weighted_row[col] = round(df.iloc[:24][col].mean(), 2)
    df = pd.concat([df, pd.DataFrame([weighted_row], columns=columns)], ignore_index=True)

    # 6 ── Export CSV ─────────────────────────────────────────────────────
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'results', 'q1_results.csv')
    df.to_csv(out_path, encoding='utf-8-sig', index=False)

    # 7 ── Console summary ────────────────────────────────────────────────
    print(f"Q1 24场景分析完成: 全满足 {full_count} 场景, 部分满足 {partial_count} 场景, 全不满足 {fail_count} 场景")

    return df


# ── Direct execution ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_q1()
    run_q1_all_scenarios()
