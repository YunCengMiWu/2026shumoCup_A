"""Q3: 连续功率调度 LP — 24 场景，固定日产氨 72 吨
===========================================================
设备连续功率范围 [MIN_LOAD_RATIO × rated, rated] = [10%, 100%]。
模型: MILP (含购/售电互斥 binary 变量) — CBC 求解。
目标: 最小化电网购电净成本, 日制氨约束 72 吨。

Run standalone:  python models/q3_lp.py  →  results/q3_results.csv
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
    NH3_RAMP_UP, NH3_RAMP_DOWN,
    LAMBDA_SELF, LAMBDA_GREEN, LAMBDA_FEED,
    get_price,
)
from utils.data_loader import get_all_scenarios, get_base_load
from utils.indicators import compute_green_indicators, compute_ton_cost, compute_compliance

# ── 状态中文映射 ────────────────────────────────────────────────────────────
STATUS_CN = {"Optimal": "最优", "Infeasible": "不可行", "Feasible": "可行"}

# ── 设备台数 ──────────────────────────────────────────────────────────────────
# ALK: 4 台 × 10 MW/台, PEM: 2 台 × 10 MW/台
N_ALK = 4
N_PEM = 2
# NH3: 每台 0.75 MW 满负荷日产 36 吨, 达 72 吨需 2 台
N_NH3 = int(np.ceil(RATED_DAILY_NH3 / (
    NH3_RATED_POWER * NH3_POWER_RATE * 1000 * HOURS_PER_DAY
)))

# ── 设备类型总装机 (MW) ──────────────────────────────────────────────────────
ALK_SYS_RATED = N_ALK * ALK_RATED_POWER   # 40 MW
PEM_SYS_RATED = N_PEM * PEM_RATED_POWER   # 20 MW
NH3_SYS_RATED = N_NH3 * NH3_RATED_POWER   # 1.5 MW (2 台)


# ═══════════════════════════════════════════════════════════════════════════════
# 单场景 MILP 模型
# ═══════════════════════════════════════════════════════════════════════════════
def optimize_q3_scenario(wind_pu, pv_pu, base_pu):
    """Q3: 最小化电网净成本, 日产氨 = RATED_DAILY_NH3 吨.

    Variables (24h, continuous):
        p_alk[h]  ∈ [MIN_LOAD_RATIO·ALK_SYS_RATED,  ALK_SYS_RATED]  — ALK 总功率
        p_pem[h]  ∈ [MIN_LOAD_RATIO·PEM_SYS_RATED,  PEM_SYS_RATED]  — PEM 总功率
        p_nh3[h]  ∈ [MIN_LOAD_RATIO·NH3_SYS_RATED,  NH3_SYS_RATED]  — 合成氨总功率
        p_buy[h]  ≥ 0   — 网购电功率 (MW)
        p_sell[h] ≥ 0   — 上网电功率 (MW)
        u_grid[h] ∈ {0, 1}  — 购/售电方向 (0=售电, 1=购电)

    Constraints:
        1. 设备上下限
        2. 购售电互斥 (BIG-M)
        3. 功率平衡 (每小时)
        4. 日总氢平衡: Σ H2_prod = H2_PER_TON_NH3 × RATED_DAILY_NH3
        5. 日制氨目标: Σ p_nh3 × NH3_POWER_RATE × 1000 × Δt = RATED_DAILY_NH3

    Objective:
        min Σ_h (p_buy[h]·price[h] − p_sell[h]·FEED_IN_PRICE) × 1000 × Δt

    Post-evaluation (不在目标中):
        绿电指标 + 吨氨成本 (含风光发电/运维/折旧)
    """
    H = HOURS_PER_DAY

    # ── 标幺 → 实际功率 (MW) ──────────────────────────────────────────────
    P_wind = wind_pu * WIND_CAPACITY
    P_pv   = pv_pu   * PV_CAPACITY
    P_base = base_pu  * BASE_LOAD_PEAK

    # ── 构建 MILP ────────────────────────────────────────────────────────
    prob = pulp.LpProblem("Q3_LP_MinCost", pulp.LpMinimize)

    p_alk  = [pulp.LpVariable(f"pa_{t}", lowBound=0) for t in range(H)]
    p_pem  = [pulp.LpVariable(f"pp_{t}", lowBound=0) for t in range(H)]
    p_nh3  = [pulp.LpVariable(f"pn_{t}", lowBound=0) for t in range(H)]
    p_buy  = [pulp.LpVariable(f"pb_{t}", lowBound=0) for t in range(H)]
    p_sell = [pulp.LpVariable(f"ps_{t}", lowBound=0) for t in range(H)]
    u_grid = [pulp.LpVariable(f"ug_{t}", cat="Binary") for t in range(H)]

    # ── Green electricity soft-constraint slack variables ──
    s_self  = pulp.LpVariable("s_self",  lowBound=0.0)  # η_self  violation (MWh)
    s_green = pulp.LpVariable("s_green", lowBound=0.0)  # η_green violation (MWh)
    s_feed  = pulp.LpVariable("s_feed",  lowBound=0.0)  # η_grid  violation (MWh)

    # ── 逐时约束 ──────────────────────────────────────────────────────────
    for t in range(H):
        # 设备出力范围
        prob += p_alk[t] >= MIN_LOAD_RATIO * ALK_SYS_RATED
        prob += p_alk[t] <= ALK_SYS_RATED

        prob += p_pem[t] >= MIN_LOAD_RATIO * PEM_SYS_RATED
        prob += p_pem[t] <= PEM_SYS_RATED

        prob += p_nh3[t] >= MIN_LOAD_RATIO * NH3_SYS_RATED
        prob += p_nh3[t] <= NH3_SYS_RATED

        # 购/售电互斥
        prob += p_buy[t]  <= u_grid[t]       * BIG_M
        prob += p_sell[t] <= (1 - u_grid[t]) * BIG_M

        # 功率平衡
        prob += (P_wind[t] + P_pv[t] + p_buy[t]
                 == P_base[t] + p_alk[t] + p_pem[t] + p_nh3[t] + p_sell[t])

    # ── NH3 非对称爬坡约束 ────────────────────────────────────────────────
    for t in range(1, H):
        prob += p_nh3[t] - p_nh3[t-1] <= NH3_RAMP_UP * NH3_SYS_RATED
        prob += p_nh3[t-1] - p_nh3[t] <= NH3_RAMP_DOWN * NH3_SYS_RATED

    # ── 日总氢平衡 ────────────────────────────────────────────────────────
    # Σ_h (ALK_H2_RATE·p_alk + PEM_H2_RATE·p_pem) × Δt = H2_PER_TON_NH3 × 72
    prob += pulp.lpSum(
        ALK_H2_RATE * p_alk[t] + PEM_H2_RATE * p_pem[t]
        for t in range(H)
    ) * DELTA_T == H2_PER_TON_NH3 * RATED_DAILY_NH3

    # ── 日制氨目标 ────────────────────────────────────────────────────────
    # Σ_h p_nh3 × NH3_POWER_RATE × 1000 × Δt = RATED_DAILY_NH3
    prob += pulp.lpSum(
        p_nh3[t] for t in range(H)
    ) * NH3_POWER_RATE * 1000 * DELTA_T == RATED_DAILY_NH3

    # ── 绿电消纳软约束 ──────────────────────────────────────────
    E_RE_const = float(np.sum(P_wind + P_pv))   # MWh (constant)
    E_base_const = float(np.sum(P_base))        # MWh

    # η_self ≥ 0.60  →  E_total−E_sell−E_buy + s_self ≥ 0.60·E_RE
    prob += (
        E_base_const
        + pulp.lpSum(p_alk[t] + p_pem[t] + p_nh3[t] for t in range(H))
        - pulp.lpSum(p_sell[t] for t in range(H))
        - pulp.lpSum(p_buy[t] for t in range(H))
        + s_self
        >= 0.60 * E_RE_const
    )

    # η_green ≥ 0.30  →  E_RE − E_sell + s_green ≥ 0.30·E_total
    prob += (
        E_RE_const
        - pulp.lpSum(p_sell[t] for t in range(H))
        + s_green
        >= 0.30 * (
            E_base_const
            + pulp.lpSum(p_alk[t] + p_pem[t] + p_nh3[t] for t in range(H))
        )
    )

    # η_grid ≤ 0.20  →  E_sell ≤ 0.20·E_RE + s_feed
    prob += (
        pulp.lpSum(p_sell[t] for t in range(H))
        <= 0.20 * E_RE_const + s_feed
    )

    # ── 目标函数: 电网净成本 (MW → kW × 1000) ──
    prob += pulp.lpSum(
        (p_buy[t] * get_price(t) * 1000 - p_sell[t] * FEED_IN_PRICE * 1000)
        * DELTA_T
        for t in range(H)
    ) + LAMBDA_SELF * s_self * 1000.0 \
      + LAMBDA_GREEN * s_green * 1000.0 \
      + LAMBDA_FEED * s_feed * 1000.0

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

    # ── 电网运行成本 ──────────────────────────────────────────────────────
    grid_cost = float(sum(
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
    E_RE   = float(np.sum(P_wind + P_pv))          * DELTA_T
    E_total = float(np.sum(P_base) + E_alk + E_pem + E_nh3)  # all consumption

    # ── 绿电直连三大指标 (后评估) ─────────────────────────────────────────
    eta_self, eta_green, eta_grid = compute_green_indicators(
        E_total, E_RE, E_sell, E_buy
    )

    # ── 吨氨成本 (后评估: grid + RE + OM + 折旧) ─────────────────────────
    ton_cost = compute_ton_cost(
        p_buy_val, p_sell_val, P_wind, P_pv,
        p_alk_val, p_pem_val, p_nh3_val, daily_nh3_tons
    )

    return {
        "status":  "Optimal",
        "grid_cost":     grid_cost,
        "ton_cost":      ton_cost,
        "E_alk":         E_alk,
        "E_pem":         E_pem,
        "E_nh3":         E_nh3,
        "E_buy":         E_buy,
        "E_sell":        E_sell,
        "E_RE":          E_RE,
        "E_total":       E_total,
        "eta_self":      eta_self,
        "eta_green":     eta_green,
        "eta_grid":      eta_grid,
        "s_self":  float(pulp.value(s_self))  if pulp.value(s_self)  is not None else 0.0,
        "s_green": float(pulp.value(s_green)) if pulp.value(s_green) is not None else 0.0,
        "s_feed":  float(pulp.value(s_feed))  if pulp.value(s_feed)  is not None else 0.0,
        "daily_nh3_tons": daily_nh3_tons,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 批量求解 24 场景 → CSV
# ═══════════════════════════════════════════════════════════════════════════════
def run_all():
    """遍历 24 个风光场景, 逐一求解, 输出 q3_results.csv."""
    scenarios = get_all_scenarios()
    base_pu = get_base_load()
    n_total = len(scenarios)

    print("=" * 70)
    print("Q3 连续功率调度 LP — 24 场景求解")
    print(f"制氨目标: {RATED_DAILY_NH3} 吨/日  设备范围: [{MIN_LOAD_RATIO*100:.0f}%, 100%]")
    print("=" * 70)

    rows = []
    n_opt = 0

    for s in scenarios:
        sid   = s["id"]
        w_idx = s["wind_scene"]
        p_idx = s["pv_scene"]

        print(f"[Q3] 场景 {sid:2d}/24 (风电={w_idx}, 光伏={p_idx}) ... ", end="", flush=True)

        sol = optimize_q3_scenario(s["wind_pu"], s["pv_pu"], base_pu)

        if sol["status"] != "Optimal":
            print(f"状态: {sol['status']}")
            rows.append({
                "场景编号": sid,
                "风电场景": w_idx,
                "光伏场景": p_idx,
                "运行成本(元)":    float("nan"),
                "吨氨成本(元/吨)":  float("nan"),
                "日购电量(MWh)":    float("nan"),
                "日售电量(MWh)":    float("nan"),
                "日ALK耗电量(MWh)": float("nan"),
                "日PEM耗电量(MWh)": float("nan"),
                "日NH3耗电量(MWh)": float("nan"),
                "新能源自发自用电量占比(%)": float("nan"),
                "总用电量绿电比例(%)":     float("nan"),
                "新能源上网电量比例(%)":   float("nan"),
                "三分类":                   "不可行",
                "状态":                   STATUS_CN.get(sol["status"], sol["status"]),
            })
        else:
            n_opt += 1
            print(f"成本={sol['grid_cost']:,.0f}元  "
                  f"吨氨={sol['ton_cost']:,.0f}元/吨  Optimal")

            # 三分类判定
            compliance = compute_compliance(sol['eta_self'], sol['eta_green'], sol['eta_grid'])
            compliant_count = sum([compliance['self_compliant'], compliance['green_compliant'], compliance['grid_compliant']])
            if compliant_count == 3:
                category = '全满足'
            elif compliant_count >= 1:
                category = '部分满足'
            else:
                category = '全不满足'

            rows.append({
                "场景编号": sid,
                "风电场景": w_idx,
                "光伏场景": p_idx,
                "运行成本(元)":    round(sol["grid_cost"], 2),
                "吨氨成本(元/吨)":  round(sol["ton_cost"], 2),
                "日购电量(MWh)":    round(sol["E_buy"],  4),
                "日售电量(MWh)":    round(sol["E_sell"], 4),
                "日ALK耗电量(MWh)": round(sol["E_alk"],  4),
                "日PEM耗电量(MWh)": round(sol["E_pem"],  4),
                "日NH3耗电量(MWh)": round(sol["E_nh3"],  4),
                "新能源自发自用电量占比(%)": round(sol["eta_self"] * 100, 2),
                "总用电量绿电比例(%)":     round(sol["eta_green"] * 100, 2),
                "新能源上网电量比例(%)":   round(sol["eta_grid"] * 100, 2),
                "s_self(MWh)":  round(sol.get("s_self", float('nan')), 4),
                "s_green(MWh)": round(sol.get("s_green", float('nan')), 4),
                "s_feed(MWh)":  round(sol.get("s_feed", float('nan')), 4),
                "三分类":                   category,
                "状态":                   STATUS_CN.get("Optimal", "最优"),
            })

    # ── 年度汇总 ──────────────────────────────────────────────────────────
    n_total = len(rows)
    n_quan = sum(1 for r in rows if r.get('三分类') == '全满足')
    n_bufen = sum(1 for r in rows if r.get('三分类') == '部分满足')
    n_bu = sum(1 for r in rows if r.get('三分类') == '全不满足')
    n_buke = sum(1 for r in rows if r.get('三分类') == '不可行')

    rows.append({
        "场景编号": "年度汇总",
        "风电场景": "",
        "光伏场景": "",
        "运行成本(元)":    "",
        "吨氨成本(元/吨)":  "",
        "日购电量(MWh)":    "",
        "日售电量(MWh)":    "",
        "日ALK耗电量(MWh)": "",
        "日PEM耗电量(MWh)": "",
        "日NH3耗电量(MWh)": "",
        "新能源自发自用电量占比(%)": "",
        "总用电量绿电比例(%)":     "",
        "新能源上网电量比例(%)":   "",
        "s_self(MWh)": "", "s_green(MWh)": "", "s_feed(MWh)": "",
        "三分类": f"全满足{n_quan}场景({n_quan*15}天) | 部分满足{n_bufen}场景({n_bufen*15}天) | 全不满足{n_bu}场景({n_bu*15}天) | 不可行{n_buke}场景({n_buke*15}天)",
        "状态": f"共{n_total}场景，{n_opt}个最优",
    })

    # ── 输出 CSV ────────────────────────────────────────────────────────────
    df = pd.DataFrame(rows, columns=[
        "场景编号", "风电场景", "光伏场景",
        "运行成本(元)", "吨氨成本(元/吨)",
        "日购电量(MWh)", "日售电量(MWh)",
        "日ALK耗电量(MWh)", "日PEM耗电量(MWh)", "日NH3耗电量(MWh)",
        "新能源自发自用电量占比(%)", "总用电量绿电比例(%)", "新能源上网电量比例(%)",
        "s_self(MWh)", "s_green(MWh)", "s_feed(MWh)",
        "三分类", "状态",
    ])
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
    )
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "q3_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # ── 总结 ────────────────────────────────────────────────────────────────
    print(f"\n[Q3] 全部 {n_total} 场景求解完成 — Optimal: {n_opt}/{n_total}")
    print(f"[Q3] 结果已保存至: {csv_path}")

    if n_opt == n_total:
        print("[Q3] ALL SCENARIOS OPTIMAL")
    else:
        print(f"[Q3] WARNING: {n_total - n_opt} scenarios non-optimal")
        sys.exit(1)


# ── 直接执行 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_all()
