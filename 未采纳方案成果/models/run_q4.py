"""
Q4: Off-grid storage optimization MILP
======================================
Independent micro-grid: 104 MW RE (40 wind+64 PV), no grid.
Single-unit ratings: ALK=10, PEM=10, NH3=0.75 MW.
Solves 24 scenarios (6 wind x 4 PV) for optimal E_cap.

Three bug fixes:
  BUG1: SOC(t=0) = SOC_INITIAL_FRAC * E_cap
  BUG2: Device p_min = MIN_LOAD_RATIO * rated
  BUG3: BIG_M = 1000 (unified)
"""
import sys, os, csv
import numpy as np
import pulp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.constants import (
    WIND_CAPACITY, PV_CAPACITY, BASE_LOAD_PEAK,
    ALK_RATED_POWER, PEM_RATED_POWER, NH3_RATED_POWER,
    ALK_H2_RATE, PEM_H2_RATE, H2_PER_TON_NH3, NH3_POWER_RATE,
    WIND_COST, PV_COST, ALK_OM_COST, PEM_OM_COST, NH3_OM_COST,
    NH3_DEPR_PER_TON, FEED_IN_PRICE,
    STORAGE_INV_COST, STORAGE_OM_COST, STORAGE_LIFE_YEARS,
    STORAGE_CHARGE_EFF, STORAGE_DISCHARGE_EFF,
    STORAGE_SELF_DISCHARGE,
    STORAGE_SOC_MIN_RATIO, STORAGE_SOC_MAX_RATIO,
    STORAGE_POWER_RATIO, SOC_INITIAL_FRAC,
    H2_STORAGE_INV_COST, H2_STORAGE_LIFE, H2_STORAGE_EFF, H2_STORAGE_P_RATIO,
    BIG_M, MIN_LOAD_RATIO,
    HOURS_PER_DAY, DELTA_T, DAYS_PER_SCENARIO, TOTAL_DAYS,
    WIND_SCENARIOS, PV_SCENARIOS, TOTAL_SCENARIOS,
)
from utils.data_loader import get_all_scenarios, get_base_load
from utils.indicators import compute_green_indicators, compute_ton_cost

# Marker to detect unwanted overwrites
_BUILD_ID = "Q4_SINGLE_UNIT_v3_20260523"

STATUS_CN = {"Optimal": "最优", "Infeasible": "不可行", "Feasible": "可行"}


def solve_one(scenario, base_pu):
    """Solve MILP for a single (wind,pv) scenario."""
    T = HOURS_PER_DAY
    p_wind = scenario['wind_pu'] * WIND_CAPACITY
    p_pv = scenario['pv_pu'] * PV_CAPACITY
    p_base = base_pu * BASE_LOAD_PEAK

    prob = pulp.LpProblem(f"Q4_S{scenario['id']:02d}", pulp.LpMinimize)

    # Storage capacity
    E_cap = pulp.LpVariable("E_cap", lowBound=0)
    P_cap = pulp.LpVariable("P_cap", lowBound=0)

    # SOC 0..T
    SOC = [pulp.LpVariable(f"soc{t:02d}", lowBound=0) for t in range(T + 1)]

    # Power variables (MW)
    p_ch = [pulp.LpVariable(f"ch{t:02d}", lowBound=0) for t in range(T)]
    p_dis = [pulp.LpVariable(f"dis{t:02d}", lowBound=0) for t in range(T)]
    p_alk = [pulp.LpVariable(f"alk{t:02d}", lowBound=0) for t in range(T)]
    p_pem = [pulp.LpVariable(f"pem{t:02d}", lowBound=0) for t in range(T)]
    p_nh3 = [pulp.LpVariable(f"nh3{t:02d}", lowBound=0) for t in range(T)]
    p_curt = [pulp.LpVariable(f"curt{t:02d}", lowBound=0) for t in range(T)]

    # Binary variables
    y_alk = [pulp.LpVariable(f"ya{t:02d}", cat="Binary") for t in range(T)]
    y_pem = [pulp.LpVariable(f"yp{t:02d}", cat="Binary") for t in range(T)]
    y_nh3 = [pulp.LpVariable(f"yn{t:02d}", cat="Binary") for t in range(T)]
    y_ch = [pulp.LpVariable(f"yc{t:02d}", cat="Binary") for t in range(T)]
    y_dis = [pulp.LpVariable(f"yd{t:02d}", cat="Binary") for t in range(T)]

    # H2 Storage variables (continuous)
    H2_cap = pulp.LpVariable("H2_cap", lowBound=0)  # kg H2
    H2_stored = [pulp.LpVariable(f"H2soc{t:02d}", lowBound=0) for t in range(T + 1)]  # kg H2
    H2_ch = [pulp.LpVariable(f"H2ch{t:02d}", lowBound=0) for t in range(T)]  # kg/h
    H2_dis = [pulp.LpVariable(f"H2dis{t:02d}", lowBound=0) for t in range(T)]  # kg/h

    # ── Constraints ────────────────────────────────────────────────────

    # P-E coupling
    prob += P_cap == STORAGE_POWER_RATIO * E_cap

    # SOC bounds
    for t in range(T + 1):
        prob += SOC[t] >= STORAGE_SOC_MIN_RATIO * E_cap
        prob += SOC[t] <= STORAGE_SOC_MAX_RATIO * E_cap

    # SOC initial (BUG1 FIX)
    prob += SOC[0] == SOC_INITIAL_FRAC * E_cap

    # SOC dynamics
    for t in range(T):
        prob += SOC[t + 1] == (
            SOC[t] * (1 - STORAGE_SELF_DISCHARGE)
            + (p_ch[t] * STORAGE_CHARGE_EFF
               - p_dis[t] / STORAGE_DISCHARGE_EFF) * DELTA_T
        )

    # H2 storage SOC initial (half-full)
    prob += H2_stored[0] == 0.5 * H2_cap

    # H2 storage SOC bounds
    for t in range(T + 1):
        prob += H2_stored[t] >= 0.1 * H2_cap
        prob += H2_stored[t] <= 0.9 * H2_cap

    # H2 storage dynamics
    for t in range(T):
        prob += H2_stored[t + 1] == (
            H2_stored[t]
            + (H2_ch[t] * H2_STORAGE_EFF - H2_dis[t]) * DELTA_T
        )

    # H2 charge/discharge rate limit
    for t in range(T):
        prob += H2_ch[t] + H2_dis[t] <= H2_cap * H2_STORAGE_P_RATIO

    # Power balance (off-grid, with curtailment)
    for t in range(T):
        prob += (
            p_wind[t] + p_pv[t] + p_dis[t]
            == p_base[t] + p_alk[t] + p_pem[t] + p_nh3[t]
            + p_ch[t] + p_curt[t]
        )

    # Charge/discharge limits + mutex (BUG3 FIX: BIG_M=1000)
    for t in range(T):
        prob += p_ch[t] <= P_cap
        prob += p_ch[t] <= BIG_M * y_ch[t]
        prob += p_dis[t] <= P_cap
        prob += p_dis[t] <= BIG_M * y_dis[t]
        prob += y_ch[t] + y_dis[t] <= 1

    # Device power range (BUG2 FIX: min = MIN_LOAD_RATIO * rated)
    for t in range(T):
        prob += p_alk[t] <= ALK_RATED_POWER * y_alk[t]
        prob += p_alk[t] >= MIN_LOAD_RATIO * ALK_RATED_POWER - BIG_M * (1 - y_alk[t])
        prob += p_pem[t] <= PEM_RATED_POWER * y_pem[t]
        prob += p_pem[t] >= MIN_LOAD_RATIO * PEM_RATED_POWER - BIG_M * (1 - y_pem[t])
        prob += p_nh3[t] <= NH3_RATED_POWER * y_nh3[t]
        prob += p_nh3[t] >= MIN_LOAD_RATIO * NH3_RATED_POWER - BIG_M * (1 - y_nh3[t])

    # H2 balance (with H2 storage buffer)
    for t in range(T):
        prob += (
            p_alk[t] * ALK_H2_RATE + p_pem[t] * PEM_H2_RATE + H2_dis[t]
            == p_nh3[t] * NH3_POWER_RATE * 1000 * H2_PER_TON_NH3 + H2_ch[t]
        )

    # ── Objective ──────────────────────────────────────────────────────
    # TAC = storage_annual_inv + 15 * daily_cost
    # Curtailment penalty = FEED_IN_PRICE * curtailed_energy (drives NH3 production)

    storage_annual_inv = STORAGE_INV_COST * E_cap * 1000 / STORAGE_LIFE_YEARS
    h2_storage_annual_inv = H2_STORAGE_INV_COST * H2_cap * 1000 / H2_STORAGE_LIFE

    re_cost = pulp.lpSum(
        (p_wind[t] * WIND_COST + p_pv[t] * PV_COST) * 1000 * DELTA_T
        for t in range(T)
    )
    h2_om = pulp.lpSum(
        (p_alk[t] * ALK_OM_COST + p_pem[t] * PEM_OM_COST) * 1000 * DELTA_T
        for t in range(T)
    )
    nh3_om_daily = pulp.lpSum(p_nh3[t] * NH3_OM_COST * 1000 * DELTA_T for t in range(T))
    daily_nh3_expr = pulp.lpSum(
        p_nh3[t] * NH3_POWER_RATE * 1000 * DELTA_T for t in range(T)
    )
    nh3_depr_daily = NH3_DEPR_PER_TON * daily_nh3_expr
    storage_om_daily = pulp.lpSum(
        STORAGE_OM_COST * (p_ch[t] + p_dis[t]) * 1000 * DELTA_T for t in range(T)
    )
    curt_penalty = pulp.lpSum(
        p_curt[t] * FEED_IN_PRICE * 1000 * DELTA_T for t in range(T)
    )

    daily_cost = re_cost + h2_om + nh3_om_daily + nh3_depr_daily + storage_om_daily + curt_penalty
    prob += storage_annual_inv + h2_storage_annual_inv + daily_cost * DAYS_PER_SCENARIO

    # ── Solve ──────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if prob.status != pulp.LpStatusOptimal:
        return None

    # ── Extract ────────────────────────────────────────────────────────
    E_cap_v = pulp.value(E_cap)
    P_cap_v = pulp.value(P_cap)
    soc_v = [pulp.value(SOC[t]) for t in range(T + 1)]
    p_ch_v = [pulp.value(p_ch[t]) for t in range(T)]
    p_dis_v = [pulp.value(p_dis[t]) for t in range(T)]
    p_alk_v = [pulp.value(p_alk[t]) for t in range(T)]
    p_pem_v = [pulp.value(p_pem[t]) for t in range(T)]
    p_nh3_v = [pulp.value(p_nh3[t]) for t in range(T)]
    p_curt_v = [pulp.value(p_curt[t]) for t in range(T)]
    H2_cap_v = pulp.value(H2_cap)
    H2_stored_v = [pulp.value(H2_stored[t]) for t in range(T + 1)]
    H2_ch_v = [pulp.value(H2_ch[t]) for t in range(T)]
    H2_dis_v = [pulp.value(H2_dis[t]) for t in range(T)]
    daily_nh3 = pulp.value(daily_nh3_expr)

    # Indicators
    E_RE_tot = float(np.sum(p_wind + p_pv)) * DELTA_T
    p_load = p_base + np.array(p_alk_v) + np.array(p_pem_v) + np.array(p_nh3_v) + np.array(p_ch_v) + np.array(p_curt_v)
    E_tot = float(np.sum(p_load)) * DELTA_T
    eta_s, eta_g, eta_gd = compute_green_indicators(E_tot, E_RE_tot, 0, 0)
    ton_c = compute_ton_cost(
        np.zeros(T), np.zeros(T), p_wind, p_pv, np.array(p_alk_v),
        np.array(p_pem_v), np.array(p_nh3_v), daily_nh3,
    )

    return {
        'id': scenario['id'], 'wind_scene': scenario['wind_scene'],
        'pv_scene': scenario['pv_scene'],
        'E_cap': E_cap_v, 'P_cap': P_cap_v,
        'annual_inv': pulp.value(storage_annual_inv),
        'annual_op': pulp.value(daily_cost) * DAYS_PER_SCENARIO,
        'TAC': pulp.value(prob.objective),
        'eta_self': eta_s, 'eta_green': eta_g, 'eta_grid': eta_gd,
        'ton_cost': ton_c, 'daily_nh3': daily_nh3,
        'SOC': soc_v, 'p_ch': p_ch_v, 'p_dis': p_dis_v,
        'p_alk': p_alk_v, 'p_pem': p_pem_v, 'p_nh3': p_nh3_v,
        'p_curt': p_curt_v,
        'H2_cap': H2_cap_v, 'H2_stored': H2_stored_v,
        'H2_ch': H2_ch_v, 'H2_dis': H2_dis_v,
        'status': 'Optimal',
    }


# =============================================================================
# Main
# =============================================================================

def main():
    scenarios = get_all_scenarios()
    base_pu = get_base_load()

    print(f"Q4: Off-grid Storage Optimization MILP  [{_BUILD_ID}]")
    print(f"BIG_M={BIG_M}, MIN_LOAD={MIN_LOAD_RATIO:.0%}, SOC_init={SOC_INITIAL_FRAC:.0%}E")
    print(f"Rated: ALK={ALK_RATED_POWER} PEM={PEM_RATED_POWER} NH3={NH3_RATED_POWER}")
    print(f"Curtailment penalty: {FEED_IN_PRICE} yuan/kWh")
    print("=" * 60)

    results, n_opt, n_inf = [], 0, 0
    for i, sc in enumerate(scenarios):
        s_id = sc['id']
        print(f"[{i+1:2d}/24] S{s_id:02d} (w{sc['wind_scene']},pv{sc['pv_scene']}) ", end="", flush=True)
        r = solve_one(sc, base_pu)
        if r is None:
            print("INFEASIBLE")
            n_inf += 1
        else:
            print(f"OK E={r['E_cap']:.1f}MWh TAC={r['TAC']:.0f} NH3={r['daily_nh3']:.2f}t/d")
            results.append(r)
            n_opt += 1

    print(f"---\nDone: {n_opt} optimal, {n_inf} infeasible ({len(scenarios)} total)")
    if results:
        emin, emax = min(r['E_cap'] for r in results), max(r['E_cap'] for r in results)
        nmin, nmax = min(r['daily_nh3'] for r in results), max(r['daily_nh3'] for r in results)
        print(f"E_cap: {emin:.1f} ~ {emax:.1f} MWh, Daily NH3: {nmin:.2f} ~ {nmax:.2f} t/d")
        # Verify BUG1
        ratios = [r['SOC'][0] / r['E_cap'] for r in results if r['E_cap'] > 1e-6]
        if ratios:
            dev = max(abs(rr - SOC_INITIAL_FRAC) for rr in ratios)
            print(f"SOC0/E avg={np.mean(ratios):.4f} max_dev={dev:.6f}  {'[OK] BUG1' if dev<0.01 else '[FAIL] BUG1'}")
    print("=" * 60)

    # CSV export
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, 'q4_results.csv')

    soc_hdrs = [f"SOC_{t}" for t in range(HOURS_PER_DAY)]
    fields = [
        '场景编号', '风电场景', '光伏场景', '储能容量(MWh)', '储能功率(MW)',
        '年化投资成本(元)', '年运行成本(元)', 'TAC(元)',
        '新能源自发自用电量占比(%)', '总用电量绿电比例(%)', '新能源上网电量比例(%)', '吨氨成本(元/吨)', '日制氨量(吨)',
    ] + soc_hdrs + ['状态']

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = {fields[0]: r['id'], fields[1]: r['wind_scene'], fields[2]: r['pv_scene'],
                   fields[3]: round(r['E_cap'], 4), fields[4]: round(r['P_cap'], 4),
                   fields[5]: round(r['annual_inv'], 2), fields[6]: round(r['annual_op'], 2),
                   fields[7]: round(r['TAC'], 2),
                   fields[8]: round(r['eta_self'], 6), fields[9]: round(r['eta_green'], 6),
                   fields[10]: round(r['eta_grid'], 6),
                   fields[11]: round(r['ton_cost'], 2), fields[12]: round(r['daily_nh3'], 4),
                   fields[-1]: STATUS_CN.get(r['status'], r['status'])}
            for t in range(HOURS_PER_DAY):
                row[f"SOC_{t}"] = round(r['SOC'][t], 4)
            w.writerow(row)

    print(f"CSV -> {csv_path} ({len(results)} rows)")
    return results


if __name__ == '__main__':
    main()
