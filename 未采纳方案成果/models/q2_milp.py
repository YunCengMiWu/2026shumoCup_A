"""
Q2 MILP: 24 scenarios × 5 daily NH3 targets, pure discrete (on=rated / off=0)
============================================================================
纯离散混合整数线性规划：所有可调度设备仅有开（额定功率）/关（0）两种状态。
目标函数最小化日运行成本（购电成本 - 售电收入）。

Run standalone:  python models/q2_milp.py
Output: results/q2_results.csv (120 rows = 5 targets × 24 scenarios)

Reference: 最终版公式.md
"""

import sys
import os
import numpy as np
import pandas as pd

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value, PULP_CBC_CMD, LpStatus

from utils.constants import (
    ALK_H2_RATE, ALK_RATED_POWER, ALK_STARTUP_COST,
    BASE_LOAD_PEAK, BIG_M,
    DELTA_T,
    FEED_IN_PRICE,
    get_price,
    H2_PER_TON_NH3, HOURS_PER_DAY,
    LAMBDA_FEED, LAMBDA_GREEN, LAMBDA_SELF,
    MIN_DOWN_TIME, MIN_UP_TIME,
    NH3_POWER_RATE, NH3_RATED_POWER,
    PEM_H2_RATE, PEM_RATED_POWER, PEM_STARTUP_COST,
    PV_CAPACITY,
    WIND_CAPACITY,
)
from utils.data_loader import get_all_scenarios, get_base_load
from utils.indicators import compute_ton_cost, compute_green_indicators, compute_compliance

# ── Status translation ─────────────────────────────────────────────────────
STATUS_CN = {"Optimal": "最优", "Infeasible": "不可行", "Feasible": "可行"}

# ── Output path ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)
CSV_PATH = os.path.join(RESULTS_DIR, 'q2_results.csv')

# ── Daily NH3 targets (tons/day) ────────────────────────────────────────────
D_TARGETS = [72, 63, 54, 45, 36]


# ============================================================================
# Single scenario + target MILP solver
# ============================================================================

def solve_q2_one(scenario, D_target):
    """Solve pure discrete MILP for one (scenario, D_target) pair.

    Parameters
    ----------
    scenario : dict
        From get_all_scenarios(): {id, wind_scene, pv_scene, wind_pu[24], pv_pu[24]}
    D_target : float
        Daily NH3 production target (tons).

    Returns
    -------
    dict
        Keys: status, operating_cost, ton_cost, p_buy_total, p_sell_total,
              p_alk_total, p_pem_total, p_nh3_total
        NaN-filled if infeasible.
    """
    # ── 1. Load and convert data ──────────────────────────────────────────
    T = HOURS_PER_DAY  # 24

    wind_pu = scenario['wind_pu']
    pv_pu = scenario['pv_pu']
    base_pu = get_base_load()

    p_wind = wind_pu * WIND_CAPACITY   # MW, shape (24,)
    p_pv = pv_pu * PV_CAPACITY         # MW, shape (24,)
    p_base = base_pu * BASE_LOAD_PEAK  # MW, shape (24,)

    # ── 2. Create problem ─────────────────────────────────────────────────
    prob = LpProblem(
        f"Q2_S{scenario['id']:02d}_D{int(D_target)}",
        LpMinimize,
    )

    # --- Binary variables: equipment on/off ---
    u_alk = [LpVariable(f"u_alk_{t}", cat='Binary') for t in range(T)]
    u_pem = [LpVariable(f"u_pem_{t}", cat='Binary') for t in range(T)]
    u_nh3 = [LpVariable(f"u_nh3_{t}", cat='Binary') for t in range(T)]

    # --- Binary variables: start/stop transitions (ALK and PEM only) ---
    y_start_alk = [LpVariable(f"y_start_alk_{t}", cat='Binary') for t in range(T)]
    y_stop_alk  = [LpVariable(f"y_stop_alk_{t}",  cat='Binary') for t in range(T)]
    y_start_pem = [LpVariable(f"y_start_pem_{t}", cat='Binary') for t in range(T)]
    y_stop_pem  = [LpVariable(f"y_stop_pem_{t}",  cat='Binary') for t in range(T)]

    # --- Binary variables: grid buy/sell mode ---
    u_buy = [LpVariable(f"u_buy_{t}", cat='Binary') for t in range(T)]
    u_sell = [LpVariable(f"u_sell_{t}", cat='Binary') for t in range(T)]

    # --- Continuous variables: grid exchange power (MW) ---
    p_buy = [LpVariable(f"p_buy_{t}", lowBound=0.0) for t in range(T)]
    p_sell = [LpVariable(f"p_sell_{t}", lowBound=0.0) for t in range(T)]

    # ── Green electricity soft-constraint slack variables ──
    s_self  = LpVariable("s_self",  lowBound=0.0)  # η_self  violation (MWh)
    s_green = LpVariable("s_green", lowBound=0.0)  # η_green violation (MWh)
    s_feed  = LpVariable("s_feed",  lowBound=0.0)  # η_grid  violation (MWh)

    # ── 3. Constraints ────────────────────────────────────────────────────

    # 3a. Power balance (per hour)
    for t in range(T):
        # Equipment power: binary × rated (pure discrete)
        p_alk_t = u_alk[t] * ALK_RATED_POWER        # 0 or 10 MW
        p_pem_t = u_pem[t] * PEM_RATED_POWER        # 0 or 10 MW
        p_nh3_t = u_nh3[t] * NH3_RATED_POWER        # 0 or 0.75 MW

        prob += (
            p_wind[t] + p_pv[t] + p_buy[t]
            == p_base[t] + p_alk_t + p_pem_t + p_nh3_t + p_sell[t]
        )

    # 3b. Buy/sell mutual exclusion + BIG_M linking
    for t in range(T):
        prob += u_buy[t] + u_sell[t] <= 1
        prob += p_buy[t] <= BIG_M * u_buy[t]
        prob += p_sell[t] <= BIG_M * u_sell[t]

    # 3c. H2 balance: sum(ALK_H2_RATE * p_alk + PEM_H2_RATE * p_pem) * Δt = 200 * D_target
    #     p_alk[t] = u_alk[t] * ALK_RATED_POWER, same for p_pem
    h2_total = lpSum(
        ALK_H2_RATE * ALK_RATED_POWER * u_alk[t]
        + PEM_H2_RATE * PEM_RATED_POWER * u_pem[t]
        for t in range(T)
    ) * DELTA_T
    prob += h2_total == H2_PER_TON_NH3 * D_target

    # 3d. NH3 production target
    #     p_nh3[t] = u_nh3[t] * NH3_RATED_POWER (MW)
    #     NH3_POWER_RATE = 0.002 ton/(kW·h) → ×1000 for ton/(MW·h)
    nh3_total = lpSum(u_nh3[t] for t in range(T)) * NH3_RATED_POWER * NH3_POWER_RATE * 1000.0 * DELTA_T
    prob += nh3_total == D_target

    # 3e. Start-stop transition constraints (ALK and PEM only)
    #     y_start[t] - y_stop[t] == u[t] - u[t-1]  (transition detection)
    #     y_start[t] + y_stop[t] <= 1               (mutual exclusion)
    #     Assume devices were OFF before t=0: u[-1] = 0
    for dev_name, u_dev, y_start, y_stop, min_up, min_down in [
        ('ALK', u_alk, y_start_alk, y_stop_alk, MIN_UP_TIME, MIN_DOWN_TIME),
        ('PEM', u_pem, y_start_pem, y_stop_pem, MIN_UP_TIME, MIN_DOWN_TIME),
    ]:
        # t=0 boundary: u[-1] = 0 assumed
        prob += y_start[0] - y_stop[0] == u_dev[0]
        prob += y_start[0] + y_stop[0] <= 1

        # t >= 1 transition detection
        for t in range(1, T):
            prob += y_start[t] - y_stop[t] == u_dev[t] - u_dev[t - 1]
            prob += y_start[t] + y_stop[t] <= 1

        # Min-up time: u[t] >= sum of startups in past (MIN_UP_TIME-1) hours
        for t in range(T):
            past_starts = [y_start[t - k] for k in range(1, min(min_up, t + 1))]
            if past_starts:
                prob += u_dev[t] >= lpSum(past_starts)

        # Min-down time: (1-u[t]) >= sum of shutdowns in past (MIN_DOWN_TIME-1) hours
        for t in range(T):
            past_stops = [y_stop[t - k] for k in range(1, min(min_down, t + 1))]
            if past_stops:
                prob += 1 - u_dev[t] >= lpSum(past_stops)

    # 3f. Soft green indicator constraints
    E_RE_const = float(np.sum(p_wind + p_pv))   # total RE, MWh (constant)
    E_base_const = float(np.sum(p_base))         # total base load, MWh

    # η_self ≥ 0.60  →  E_total - E_sell - E_buy + s_self ≥ 0.60·E_RE
    prob += (
        E_base_const
        + lpSum(u_alk[t]*ALK_RATED_POWER + u_pem[t]*PEM_RATED_POWER + u_nh3[t]*NH3_RATED_POWER for t in range(T))
        - lpSum(p_sell[t] for t in range(T))
        - lpSum(p_buy[t] for t in range(T))
        + s_self
        >= 0.60 * E_RE_const
    )

    # η_green ≥ 0.30  →  E_RE - E_sell + s_green ≥ 0.30·E_total
    prob += (
        E_RE_const
        - lpSum(p_sell[t] for t in range(T))
        + s_green
        >= 0.30 * (
            E_base_const
            + lpSum(u_alk[t]*ALK_RATED_POWER + u_pem[t]*PEM_RATED_POWER + u_nh3[t]*NH3_RATED_POWER for t in range(T))
        )
    )

    # η_grid ≤ 0.20  →  E_sell ≤ 0.20·E_RE + s_feed
    prob += (
        lpSum(p_sell[t] for t in range(T))
        <= 0.20 * E_RE_const + s_feed
    )

    # ── 4. Objective: min grid operating cost + startup costs ────────────
    #     MW → kW (×1000), then × DELTA_T for energy
    obj = lpSum(
        p_buy[t] * 1000.0 * DELTA_T * get_price(t)
        - p_sell[t] * 1000.0 * DELTA_T * FEED_IN_PRICE
        for t in range(T)
    ) + ALK_STARTUP_COST * lpSum(y_start_alk[t] for t in range(T)) \
      + PEM_STARTUP_COST * lpSum(y_start_pem[t] for t in range(T)) \
      + LAMBDA_SELF * s_self * 1000.0 \
      + LAMBDA_GREEN * s_green * 1000.0 \
      + LAMBDA_FEED * s_feed * 1000.0
    prob += obj

    # ── 5. Solve ──────────────────────────────────────────────────────────
    prob.solve(PULP_CBC_CMD(msg=False))
    solved_status = LpStatus[prob.status]

    # ── 6. Extract results ────────────────────────────────────────────────
    if solved_status in ('Optimal', 'Feasible'):
        p_alk_vals = np.array([value(u_alk[t]) * ALK_RATED_POWER for t in range(T)])
        p_pem_vals = np.array([value(u_pem[t]) * PEM_RATED_POWER for t in range(T)])
        p_nh3_vals = np.array([value(u_nh3[t]) * NH3_RATED_POWER for t in range(T)])
        p_buy_vals = np.array([value(p_buy[t]) for t in range(T)])
        p_sell_vals = np.array([value(p_sell[t]) for t in range(T)])

        # Energy totals (MWh)
        e_buy = float(np.sum(p_buy_vals)) * DELTA_T
        e_sell = float(np.sum(p_sell_vals)) * DELTA_T
        e_alk = float(np.sum(p_alk_vals)) * DELTA_T
        e_pem = float(np.sum(p_pem_vals)) * DELTA_T
        e_nh3 = float(np.sum(p_nh3_vals)) * DELTA_T

        # Operating cost = objective value (grid net cost)
        operating_cost = value(prob.objective)

        # Full ton-cost (includes RE cost + OM + depreciation)
        ton_cost = compute_ton_cost(
            p_buy_vals, p_sell_vals, p_wind, p_pv,
            p_alk_vals, p_pem_vals, p_nh3_vals,
            D_target,
        )

        # Green indicators (post-evaluation)
        # E_total = base + ALK + PEM + NH3 (MWh)
        e_base = float(np.sum(p_base)) * DELTA_T
        e_total = e_base + e_alk + e_pem + e_nh3
        e_re = float(np.sum(p_wind + p_pv)) * DELTA_T
        eta_self, eta_green, eta_grid = compute_green_indicators(e_total, e_re, e_sell, e_buy)

        return {
            'status': solved_status,
            'operating_cost': operating_cost,
            'ton_cost': ton_cost,
            'p_buy_total': e_buy,
            'p_sell_total': e_sell,
            'p_alk_total': e_alk,
            'p_pem_total': e_pem,
            'p_nh3_total': e_nh3,
            'eta_self': eta_self,
            'eta_green': eta_green,
            'eta_grid': eta_grid,
            's_self':  float(value(s_self))  if value(s_self)  is not None else 0.0,
            's_green': float(value(s_green)) if value(s_green) is not None else 0.0,
            's_feed':  float(value(s_feed))  if value(s_feed)  is not None else 0.0,
        }
    else:
        return {
            'status': solved_status,
            'operating_cost': float('nan'),
            'ton_cost': float('nan'),
            'p_buy_total': float('nan'),
            'p_sell_total': float('nan'),
            'p_alk_total': float('nan'),
            'p_pem_total': float('nan'),
            'p_nh3_total': float('nan'),
            'eta_self': float('nan'),
            'eta_green': float('nan'),
            'eta_grid': float('nan'),
            's_self':  float('nan'),
            's_green': float('nan'),
            's_feed':  float('nan'),
        }


# ============================================================================
# Main: iterate all 24 scenarios × 5 targets
# ============================================================================

def run_q2():
    """Solve Q2 MILP for all scenario × target combinations and export CSV."""
    scenarios = get_all_scenarios()
    total_combos = len(scenarios) * len(D_TARGETS)

    print("=" * 72)
    print("Q2 MILP: 纯离散优化 (on=rated / off=0)")
    print(f"场景数: {len(scenarios)} | 目标数: {len(D_TARGETS)} | 总组合: {total_combos}")
    print("=" * 72)

    rows = []
    n_feasible = 0
    n_infeasible = 0

    for D_target in D_TARGETS:
        for scenario in scenarios:
            sid = scenario['id']
            w = scenario['wind_scene']
            pv = scenario['pv_scene']

            result = solve_q2_one(scenario, D_target)

            # 三分类判定
            if result['status'] not in ('Optimal', 'Feasible'):
                category = '不可行'
            else:
                compliance = compute_compliance(result['eta_self'], result['eta_green'], result['eta_grid'])
                compliant_count = sum([compliance['self_compliant'], compliance['green_compliant'], compliance['grid_compliant']])
                if compliant_count == 3:
                    category = '全满足'
                elif compliant_count >= 1:
                    category = '部分满足'
                else:
                    category = '全不满足'

            row = {
                '场景编号': sid,
                '风电场景': w,
                '光伏场景': pv,
                '日产量目标(吨)': D_target,
                '运行成本(元)': result['operating_cost'],
                '吨氨成本(元/吨)': result['ton_cost'],
                '状态': STATUS_CN.get(result['status'], result['status']),
                '三分类': category,
                '日购电量(MWh)': result['p_buy_total'],
                '日售电量(MWh)': result['p_sell_total'],
                'ALK总功率(MW)': result['p_alk_total'],
                'PEM总功率(MW)': result['p_pem_total'],
                'NH3总功率(MW)': result['p_nh3_total'],
                '新能源自发自用电量占比(%)': round(result['eta_self'] * 100, 2),
                '总用电量绿电比例(%)': round(result['eta_green'] * 100, 2),
                '新能源上网电量比例(%)': round(result['eta_grid'] * 100, 2),
                's_self(MWh)': round(result.get('s_self', float('nan')), 4),
                's_green(MWh)': round(result.get('s_green', float('nan')), 4),
                's_feed(MWh)': round(result.get('s_feed', float('nan')), 4),
            }
            rows.append(row)

            if result['status'] in ('Optimal', 'Feasible'):
                n_feasible += 1
            else:
                n_infeasible += 1

            status_cn = '可行' if result['status'] in ('Optimal', 'Feasible') else '不可行'
            print(f"[Q2] 场景 {sid:2d}/24 日产量{D_target:.0f}t 风电{w} 光伏{pv} ... {result['status']} ({status_cn})")

    # ── Write CSV ─────────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=[
        '场景编号', '风电场景', '光伏场景', '日产量目标(吨)',
        '运行成本(元)', '吨氨成本(元/吨)', '状态', '三分类',
        '日购电量(MWh)', '日售电量(MWh)',
        'ALK总功率(MW)', 'PEM总功率(MW)', 'NH3总功率(MW)',
        '新能源自发自用电量占比(%)', '总用电量绿电比例(%)', '新能源上网电量比例(%)',
        's_self(MWh)', 's_green(MWh)', 's_feed(MWh)',
    ])

    # ── 按产量档位汇总 ───────────────────────────────────────────────────────
    summary_rows = []
    for target in D_TARGETS:
        target_df = df[df['日产量目标(吨)'] == target]
        n_total = len(target_df)
        n_quan = len(target_df[target_df['三分类'] == '全满足'])
        n_bufen = len(target_df[target_df['三分类'] == '部分满足'])
        n_bu = len(target_df[target_df['三分类'] == '全不满足'])
        n_buke = len(target_df[target_df['三分类'] == '不可行'])

        summary_rows.append({
            '场景编号': f'档位{int(target)}t汇总',
            '风电场景': '',
            '光伏场景': '',
            '日产量目标(吨)': target,
            '运行成本(元)': '',
            '吨氨成本(元/吨)': '',
            '状态': f'共{n_total}场景',
            '三分类': f'全满足{n_quan}场景({n_quan*15}天) | 部分满足{n_bufen}场景({n_bufen*15}天) | 全不满足{n_bu}场景({n_bu*15}天) | 不可行{n_buke}场景({n_buke*15}天)',
            '日购电量(MWh)': '',
            '日售电量(MWh)': '',
            'ALK总功率(MW)': '',
            'PEM总功率(MW)': '',
            'NH3总功率(MW)': '',
            '新能源自发自用电量占比(%)': '',
            '总用电量绿电比例(%)': '',
            '新能源上网电量比例(%)': '',
            's_self(MWh)': '',
            's_green(MWh)': '',
            's_feed(MWh)': '',
        })

    df = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)

    df.to_csv(CSV_PATH, index=False, encoding='utf-8-sig')
    print(f"\n{'=' * 72}")
    print(f"结果已保存: {CSV_PATH}")
    print(f"总行数: {len(df)} (5 目标 × 24 场景 = 120)")
    print(f"Q2 完成: {n_feasible}可行 / {n_infeasible}不可行")
    print(f"{'=' * 72}")


# ── Direct execution ─────────────────────────────────────────────────────────
if __name__ == '__main__':
    run_q2()
