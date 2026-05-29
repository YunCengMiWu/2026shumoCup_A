"""Q5: α参数敏感性分析 — 典型日场景 MILP 扫描
====================================================
在Q3 LP基础上引入参数α∈[1,4], 调整目标函数中售电权重。
α物理含义: α↑ → 削弱售电激励(模拟绿电优先政策),
          预期售电量↓, η_self↑, η_green↑, 运营成本↑.

Run standalone: python models/run_q5.py → results/q5_sensitivity.csv
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pulp

from utils.constants import (
    WIND_CAPACITY, PV_CAPACITY, BASE_LOAD_PEAK,
    ALK_RATED_POWER, PEM_RATED_POWER, NH3_RATED_POWER,
    ALK_H2_RATE, PEM_H2_RATE, NH3_POWER_RATE, H2_PER_TON_NH3,
    BIG_M, MIN_LOAD_RATIO,
    DELTA_T, HOURS_PER_DAY,
    RATED_DAILY_NH3, FEED_IN_PRICE,
    get_price,
)
from utils.data_loader import get_scenario_by_id, get_base_load
from utils.indicators import compute_green_indicators, compute_ton_cost

# 状态中文映射
STATUS_CN = {"Optimal": "最优", "Infeasible": "不可行", "Feasible": "可行"}

# ── 设备台数 (同Q3) ────────────────────────────────────────────────────────────
N_ALK = 4
N_PEM = 2
N_NH3 = int(np.ceil(RATED_DAILY_NH3 / (
    NH3_RATED_POWER * NH3_POWER_RATE * 1000 * HOURS_PER_DAY
)))
ALK_SYS_RATED = N_ALK * ALK_RATED_POWER
PEM_SYS_RATED = N_PEM * PEM_RATED_POWER
NH3_SYS_RATED = N_NH3 * NH3_RATED_POWER


# ═══════════════════════════════════════════════════════════════════════════════
# 单场景 MILP 模型 (α 参数化)
# ═══════════════════════════════════════════════════════════════════════════════
def optimize_q5_scenario(wind_pu, pv_pu, base_pu, alpha):
    """Q5: α-参数化目标, 日产氨 = RATED_DAILY_NH3 吨.

    与Q3结构相同, 仅在目标函数中引入α:
        min Σ_h (p_buy[h]·price[h] − α·p_sell[h]·FEED_IN_PRICE) × 1000 × Δt

    α=1 → 正常售电价; α>1 → 削弱售电激励 (绿电优先).
    """
    H = HOURS_PER_DAY

    # ── 标幺 → 实际功率 (MW) ──────────────────────────────────────────────
    P_wind = wind_pu * WIND_CAPACITY
    P_pv   = pv_pu   * PV_CAPACITY
    P_base = base_pu  * BASE_LOAD_PEAK

    # ── 构建 MILP ────────────────────────────────────────────────────────
    prob = pulp.LpProblem(f"Q5_alpha_{alpha:.1f}", pulp.LpMinimize)

    p_alk  = [pulp.LpVariable(f"pa_{t}", lowBound=0) for t in range(H)]
    p_pem  = [pulp.LpVariable(f"pp_{t}", lowBound=0) for t in range(H)]
    p_nh3  = [pulp.LpVariable(f"pn_{t}", lowBound=0) for t in range(H)]
    p_buy  = [pulp.LpVariable(f"pb_{t}", lowBound=0) for t in range(H)]
    p_sell = [pulp.LpVariable(f"ps_{t}", lowBound=0) for t in range(H)]
    u_grid = [pulp.LpVariable(f"ug_{t}", cat="Binary") for t in range(H)]

    # ── 逐时约束 ──────────────────────────────────────────────────────────
    for t in range(H):
        prob += p_alk[t] >= MIN_LOAD_RATIO * ALK_SYS_RATED
        prob += p_alk[t] <= ALK_SYS_RATED
        prob += p_pem[t] >= MIN_LOAD_RATIO * PEM_SYS_RATED
        prob += p_pem[t] <= PEM_SYS_RATED
        prob += p_nh3[t] >= MIN_LOAD_RATIO * NH3_SYS_RATED
        prob += p_nh3[t] <= NH3_SYS_RATED

        prob += p_buy[t]  <= u_grid[t]       * BIG_M
        prob += p_sell[t] <= (1 - u_grid[t]) * BIG_M

        prob += (P_wind[t] + P_pv[t] + p_buy[t]
                 == P_base[t] + p_alk[t] + p_pem[t] + p_nh3[t] + p_sell[t])

    # ── 日总氢平衡 ────────────────────────────────────────────────────────
    prob += pulp.lpSum(
        ALK_H2_RATE * p_alk[t] + PEM_H2_RATE * p_pem[t]
        for t in range(H)
    ) * DELTA_T == H2_PER_TON_NH3 * RATED_DAILY_NH3

    # ── 日制氨目标 ────────────────────────────────────────────────────────
    prob += pulp.lpSum(
        p_nh3[t] for t in range(H)
    ) * NH3_POWER_RATE * 1000 * DELTA_T == RATED_DAILY_NH3

    # ── 目标函数: α-参数化电网净成本 (优化用) ────────────────────────────
    prob += pulp.lpSum(
        (p_buy[t] * get_price(t) * 1000
         - alpha * p_sell[t] * FEED_IN_PRICE * 1000)
        * DELTA_T
        for t in range(H)
    )

    # ── 求解 ──────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if pulp.LpStatus[prob.status] != "Optimal":
        return {"status": pulp.LpStatus[prob.status]}

    # ── 提取结果 ──────────────────────────────────────────────────────────
    p_alk_val  = np.array([pulp.value(p_alk[t])  for t in range(H)])
    p_pem_val  = np.array([pulp.value(p_pem[t])  for t in range(H)])
    p_nh3_val  = np.array([pulp.value(p_nh3[t])  for t in range(H)])
    p_buy_val  = np.array([pulp.value(p_buy[t])  for t in range(H)])
    p_sell_val = np.array([pulp.value(p_sell[t]) for t in range(H)])

    # ── 日制氨量 ──────────────────────────────────────────────────────────
    daily_nh3_tons = float(np.sum(p_nh3_val)) * NH3_POWER_RATE * 1000 * DELTA_T

    # ── 真实运营成本 (α=1, 后评估用) ──────────────────────────────────────
    operating_cost = float(sum(
        p_buy_val[t] * get_price(t) * 1000 * DELTA_T
        - p_sell_val[t] * FEED_IN_PRICE * 1000 * DELTA_T
        for t in range(H)
    ))

    # ── 电量汇总 (MWh) ────────────────────────────────────────────────────
    E_alk  = float(np.sum(p_alk_val))  * DELTA_T
    E_pem  = float(np.sum(p_pem_val))  * DELTA_T
    E_nh3  = float(np.sum(p_nh3_val))  * DELTA_T
    E_buy  = float(np.sum(p_buy_val))  * DELTA_T
    E_sell = float(np.sum(p_sell_val)) * DELTA_T
    E_RE   = float(np.sum(P_wind + P_pv)) * DELTA_T
    E_total = float(np.sum(P_base) + E_alk + E_pem + E_nh3)

    # ── 绿电直连三大指标 ─────────────────────────────────────────────────
    eta_self, eta_green, eta_grid = compute_green_indicators(
        E_total, E_RE, E_sell, E_buy
    )

    # ── 吨氨成本 (真实成本, 后评估) ──────────────────────────────────────
    ton_cost = compute_ton_cost(
        p_buy_val, p_sell_val, P_wind, P_pv,
        p_alk_val, p_pem_val, p_nh3_val, daily_nh3_tons
    )

    return {
        "status":         STATUS_CN.get("Optimal", "最优"),
        "operating_cost": operating_cost,
        "ton_cost":       ton_cost,
        "E_buy":          E_buy,
        "E_sell":         E_sell,
        "eta_self":       eta_self,
        "eta_green":      eta_green,
        "eta_grid":       eta_grid,
        "daily_nh3_tons": daily_nh3_tons,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# α 参数扫描 → CSV
# ═══════════════════════════════════════════════════════════════════════════════
def run_q5():
    """遍历 α∈[1,4] 步长 0.1, 对典型日场景逐一求解, 输出 q5_sensitivity.csv."""
    scenario = get_scenario_by_id(1)   # wind=1, pv=1
    base_pu  = get_base_load()

    alphas = np.arange(1.0, 4.05, 0.1)  # 31 points
    n_total = len(alphas)

    print("=" * 70)
    print("Q5 α 参数敏感性分析 — 典型日场景 MILP 扫描")
    print(f"场景: wind=1, pv=1  |  α范围: [1.0, 4.0]  |  步长: 0.1  |  共 {n_total} 点")
    print("=" * 70)

    rows = []
    n_opt = 0

    for alpha in alphas:
        print(f"α={alpha:.1f} ... ", end="", flush=True)

        sol = optimize_q5_scenario(scenario["wind_pu"], scenario["pv_pu"], base_pu, alpha)

        if sol["status"] != "最优":
            print(f"状态: {STATUS_CN.get(sol['status'], sol['status'])}")
            rows.append({
                "alpha":                     round(alpha, 1),
                "吨氨成本(元/吨)":            float("nan"),
                "新能源自发自用电量占比(%)":   float("nan"),
                "总用电量绿电比例(%)":         float("nan"),
                "新能源上网电量比例(%)":       float("nan"),
                "运营成本(元)":               float("nan"),
                "状态":                       STATUS_CN.get(sol["status"], sol["status"]),
            })
        else:
            n_opt += 1
            print(f"吨氨成本={sol['ton_cost']:,.0f}元  "
                  f"η_self={sol['eta_self']:.4f}  "
                  f"η_green={sol['eta_green']:.4f}  Optimal")

            rows.append({
                "alpha":                     round(alpha, 1),
                "吨氨成本(元/吨)":            round(sol["ton_cost"],       2),
                "新能源自发自用电量占比(%)":   round(sol["eta_self"],       6),
                "总用电量绿电比例(%)":         round(sol["eta_green"],      6),
                "新能源上网电量比例(%)":       round(sol["eta_grid"],       6),
                "运营成本(元)":               round(sol["operating_cost"], 2),
                "状态":                       STATUS_CN.get(sol["status"], "最优"),
            })

    # ── 输出 CSV ────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "q5_sensitivity.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ── 总结 ────────────────────────────────────────────────────────────────
    print(f"\n[Q5] 全部 {n_total} 个α点求解完成 — Optimal: {n_opt}/{n_total}")
    print(f"[Q5] 结果已保存至: {csv_path}")

    if n_opt == n_total:
        print("[Q5] ALL 31 POINTS OPTIMAL")
    else:
        print(f"[Q5] WARNING: {n_total - n_opt} points non-optimal")
        sys.exit(1)


# ── 直接执行 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_q5()
