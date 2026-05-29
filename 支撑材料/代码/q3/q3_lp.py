# -*- coding: utf-8 -*-
"""
问题三：连续制氨调节 MILP 调度优化
电氢氨园区 - 24小时连续调度，设备功率半连续可调 (0或≥10%额定)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

from pulp import LpProblem, LpVariable, LpMinimize, lpSum, value, PULP_CBC_CMD
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import os, csv
# 设置路径以导入上级目录的 config 和 utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(SCRIPT_DIR))
import config
from config import (
    NH3_BASELINE_CAPACITY,
    ALKEL_H2_RATE, PEMEL_H2_RATE,
    NH3_H2_CONSUMPTION, NH3_RATE,
    ALKEL_RATED, PEMEL_RATED, NH3_RATED,
    ALKEL_OM, PEMEL_OM, NH3_OM,
    N_ALK, N_PEM, N_NH3,
    ALK_SYS_POWER, PEM_SYS_POWER, NH3_SYS_POWER,
    ALK_H2_PER_MW, PEM_H2_PER_MW,
    NH3_LOAD_MIN, NH3_LOAD_MAX,
    NH3_RAMP_RATE,
    NH3_POWER_RATE, H2_PER_TON_NH3,
    BIG_M,
)
from utils import (
    calc_green_indicators,
    classify_scenario,
    calc_ton_ammonia_cost,
    calc_total_depreciation,
    generate_all_scenarios,
    add_green_penalty_to_problem,
    build_h2_balance_constraint,
    build_nh3_ramp_constraint,
)


# ============================================================
# 常量（基于72吨/日产能从config.py推导）
# ============================================================
SCALE_72 = 72.0 / NH3_BASELINE_CAPACITY  # 2.0

# 氢气产率（kgH₂/h）
ALKEL_H2_72 = ALKEL_H2_RATE * SCALE_72   # 280 kgH₂/h
PEMEL_H2_72 = PEMEL_H2_RATE * SCALE_72   # 320 kgH₂/h
TOTAL_H2_RATE = ALKEL_H2_72 + PEMEL_H2_72  # 600 kgH₂/h

# 合成氨产率（吨/时）
NH3_RATE_72 = NH3_RATE * SCALE_72         # 3.0 ton/h

# 设备额定功率（MW）
P_ALKEL = ALKEL_RATED * SCALE_72          # 20 MW
P_PEMEL = PEMEL_RATED * SCALE_72          # 20 MW
P_NH3   = NH3_RATED * SCALE_72            # 1.5 MW
P_TOTAL = P_ALKEL + P_PEMEL + P_NH3       # 41.5 MW

# 氢气质量平衡校验
assert abs(TOTAL_H2_RATE - NH3_RATE_72 * 1000 * NH3_H2_CONSUMPTION) < 1e-6, \
    "氢气质量不平衡: 产量={}, 消耗量={}".format(
        TOTAL_H2_RATE, NH3_RATE_72 * 1000 * NH3_H2_CONSUMPTION
    )

PRODUCTION_LEVELS = [72, 63, 54, 45, 36]  # 吨/天

# 运维系数（碱性+PEM混合）
EL_OM_BLENDED = (ALKEL_OM + PEMEL_OM) / 2  # 0.125 元/kWh
NH3_OM_RATE = NH3_OM                        # 0.002 元/kWh


# ============================================================
# 求解后校验函数
# ============================================================
def _verify_solution(result, wind, solar, load, daily_output):
    """v2校验: 分拆ALK/PEM/NH3独立检查."""
    p_alk  = result["p_alk_schedule"]
    p_pem  = result["p_pem_schedule"]
    p_nh3  = result["p_nh3_schedule"]
    buy_vals = result["buy_schedule"]
    sell_vals = result["sell_schedule"]

    P_MIN_ALK = 0.10 * ALK_SYS_POWER
    P_MIN_PEM = 0.10 * PEM_SYS_POWER
    P_MIN_NH3 = NH3_SYS_POWER * NH3_LOAD_MIN

    # 0. 半连续约束 (per-device)
    for t in range(24):
        p_tot = p_alk[t] + p_pem[t] + p_nh3[t]
        for name, pv, pmin, pmax in [("ALK",p_alk[t],P_MIN_ALK,ALK_SYS_POWER),
                                      ("PEM",p_pem[t],P_MIN_PEM,PEM_SYS_POWER),
                                      ("NH3",p_nh3[t],P_MIN_NH3,NH3_SYS_POWER*NH3_LOAD_MAX)]:
            if pv > 1e-9:
                if pv < pmin - 1e-9:
                    raise ValueError(f"半连续{name} t={t}: p={pv:.4f}<Pmin={pmin}")
                if pv > pmax + 1e-9:
                    raise ValueError(f"上限{name} t={t}: p={pv:.4f}>Pmax={pmax}")

    # 1. 功率平衡
    for t in range(24):
        imb = abs(wind[t]+solar[t]+buy_vals[t] - sell_vals[t] - p_alk[t] - p_pem[t] - p_nh3[t] - load[t])
        if imb >= 1e-4:
            raise ValueError(f"功率平衡 t={t}: 不平衡={imb:.6f}")

    # 2. 产量约束: Σp_nh3 * NH3_POWER_RATE * 1000 == daily_output
    actual_prod = np.sum(p_nh3) * NH3_POWER_RATE * 1000
    if abs(actual_prod - daily_output) >= 1e-4:
        raise ValueError(f"产量约束: {actual_prod:.4f} != {daily_output}")

    # 3. H2平衡
    if abs(result["h2_produced_kg"] - result["h2_consumed_kg"]) >= 1e-3:
        raise ValueError(f"H2平衡: {result['h2_produced_kg']:.1f} != {result['h2_consumed_kg']:.1f}")

    # 4. 绿电指标范围 (警告)
    indicators = calc_green_indicators(
        result["total_wind_kwh"] + result["total_solar_kwh"],
        result["total_load_kwh"], result["total_buy_kwh"], result["total_sell_kwh"])
    for name in ("self_use_rate","green_rate","grid_feed_rate"):
        if not (0.0 <= indicators[name] <= 1.0+1e-4):
            print(f"  [警告] {name}={indicators[name]:.4f} 超出[0,1]")

    # 5. 打印PEM/ALK比例
    total_alk = np.sum(p_alk)
    total_pem = np.sum(p_pem)
    if total_alk > 0:
        print(f"  PEM/ALK比: {total_pem/total_alk:.2f} (PEM={total_pem:.1f} ALK={total_alk:.1f} NW)")


# ============================================================
# 核心 MILP 求解器（半连续功率）
# ============================================================
def build_and_solve_lp(wind, solar, load, daily_output):
    """
    为单个场景 × 产量级别构建并求解半连续功率 MILP。

    参数
    ----------
    wind      : 24小时风电出力 (MW)，list/array
    solar     : 24小时光伏出力 (MW)，list/array
    load      : 24小时常规负荷 (MW)，list/array
    daily_output : 目标日合成氨产量 (吨/天)

    返回
    -------
    dict，包含:
        status, objective, p_schedule, z_schedule, buy_schedule, sell_schedule,
        runtime_eq_hours, 以及全部成本构成
    """
    # 所需总设备功率 (MWh)
    required_power = daily_output * P_TOTAL / NH3_RATE_72

    # ---- 建立模型 ----
    prob = LpProblem("Q3_MILP", LpMinimize)

    # === v2 变量: 拆分ALK/PEM/NH3独立连续功率 ===
    P_MIN_ALK = 0.10 * ALK_SYS_POWER
    P_MIN_PEM = 0.10 * PEM_SYS_POWER
    P_MIN_NH3 = NH3_RATED * 2 * NH3_LOAD_MIN  # NH3 min load from config

    p_alk = [LpVariable(f"p_alk_{t}", lowBound=0, upBound=ALK_SYS_POWER) for t in range(24)]
    p_pem = [LpVariable(f"p_pem_{t}", lowBound=0, upBound=PEM_SYS_POWER) for t in range(24)]
    p_nh3 = [LpVariable(f"p_nh3_{t}", lowBound=0, upBound=NH3_SYS_POWER * NH3_LOAD_MAX) for t in range(24)]

    z_alk = [LpVariable(f"z_alk_{t}", cat="Binary") for t in range(24)]
    z_pem = [LpVariable(f"z_pem_{t}", cat="Binary") for t in range(24)]
    z_nh3 = [LpVariable(f"z_nh3_{t}", cat="Binary") for t in range(24)]

    buy = [LpVariable(f"buy_{t}", lowBound=0) for t in range(24)]
    sell = [LpVariable(f"sell_{t}", lowBound=0) for t in range(24)]
    grid_dir = [LpVariable(f"grid_dir_{t}", cat="Binary") for t in range(24)]

    # ---- 约束 ----
    # 半连续约束 (per device)
    for t in range(24):
        prob += p_alk[t] <= ALK_SYS_POWER * z_alk[t], f"Semi_alk_up_{t}"
        prob += p_alk[t] >= P_MIN_ALK * z_alk[t],   f"Semi_alk_lo_{t}"
        prob += p_pem[t] <= PEM_SYS_POWER * z_pem[t], f"Semi_pem_up_{t}"
        prob += p_pem[t] >= P_MIN_PEM * z_pem[t],   f"Semi_pem_lo_{t}"
        prob += p_nh3[t] <= NH3_SYS_POWER * NH3_LOAD_MAX * z_nh3[t], f"Semi_nh3_up_{t}"
        prob += p_nh3[t] >= P_MIN_NH3 * z_nh3[t],   f"Semi_nh3_lo_{t}"

    # 产量约束: ∑ p_nh3 * NH3_POWER_RATE * 1000 = daily_output
    prob += lpSum(p_nh3) * NH3_POWER_RATE * 1000 == daily_output, "NH3_Target"

    # H2逐时+逐日平衡
    build_h2_balance_constraint(prob, p_alk, p_pem, p_nh3, 24)

    # NH3爬坡约束 (赛题: 效率变化 ≤ 20%/h → 0.30 MW/h @72t/d)
    build_nh3_ramp_constraint(prob, p_nh3, 24, P_NH3, NH3_RAMP_RATE)

    # 购售电互斥 (Big-M)
    for t in range(24):
        prob += buy[t] <= BIG_M * (1 - grid_dir[t]), f"NoBuyWhenSell_{t}"
        prob += sell[t] <= BIG_M * grid_dir[t],      f"NoSellWhenBuy_{t}"

    # 逐时功率平衡
    for t in range(24):
        prob += (wind[t] + solar[t] + buy[t] == sell[t] + p_alk[t] + p_pem[t] + p_nh3[t] + load[t]), f"Bal_{t}"

    # ---- 绿电惩罚 ----
    total_load_expr = lpSum(load) + lpSum(p_alk) + lpSum(p_pem) + lpSum(p_nh3)
    green_penalty = add_green_penalty_to_problem(prob, total_load_expr,
        float(np.sum(wind) + np.sum(solar)), lpSum(sell), lpSum(buy), config)

    # ---- 目标函数 ----
    buy_cost = lpSum(buy[t] * config.get_price(t) * 1000 for t in range(24))
    sell_revenue = lpSum(sell[t] * config.GRID_FEEDIN_PRICE * 1000 for t in range(24))
    om_cost = lpSum((p_alk[t] * 1000 * ALKEL_OM + p_pem[t] * 1000 * PEMEL_OM + p_nh3[t] * 1000 * NH3_OM) for t in range(24))

    prob += buy_cost + om_cost - sell_revenue + green_penalty, "TotalCost"

    # ---- 求解 ----
    prob.solve(PULP_CBC_CMD(msg=False))

    if prob.status != 1:
        raise RuntimeError(
            f"MILP 求解失败，状态码 {prob.status} "
            f"(期望 1=Optimal)"
        )

    # ---- 提取结果 ----
    p_alk_vals = np.array([value(p_alk[t]) for t in range(24)])
    p_pem_vals = np.array([value(p_pem[t]) for t in range(24)])
    p_nh3_vals = np.array([value(p_nh3[t]) for t in range(24)])
    p_vals = p_alk_vals + p_pem_vals + p_nh3_vals
    z_vals = np.array([int(round(value(z_alk[t]))) for t in range(24)])
    buy_vals = np.array([value(buy[t]) for t in range(24)])
    sell_vals = np.array([value(sell[t]) for t in range(24)])
    obj_val = value(prob.objective)

    # 计算各项成本
    hourly_buy_kwh = buy_vals * 1000   # MW → kW
    total_sell_kwh = np.sum(sell_vals) * 1000
    total_wind_kwh = np.sum(wind) * 1000
    total_solar_kwh = np.sum(solar) * 1000
    total_buy_kwh = np.sum(hourly_buy_kwh)
    total_alk_mwh = np.sum(p_alk_vals)
    total_pem_mwh = np.sum(p_pem_vals)
    total_nh3_mwh = np.sum(p_nh3_vals)
    total_p_mwh = total_alk_mwh + total_pem_mwh + total_nh3_mwh
    total_load_kwh = (np.sum(load) + total_p_mwh) * 1000

    # 运维费（拆分: ALK vs PEM vs NH3）— 保留用于result输出
    el_om_cost = total_alk_mwh * 1000 * ALKEL_OM + total_pem_mwh * 1000 * PEMEL_OM
    nh3_om_cost = total_nh3_mwh * 1000 * NH3_OM

    # 氢气实际产量与消耗
    h2_produced_kg = total_alk_mwh * ALK_H2_PER_MW + total_pem_mwh * PEM_H2_PER_MW
    h2_consumed_kg = daily_output * 1000 * NH3_H2_CONSUMPTION

    # 完整吨氨成本（含风光度电成本、折旧）— v2: 传逐时数组内部计算OM
    ton_cost = calc_ton_ammonia_cost(
        nh3_output_tons=daily_output,
        grid_buy_kwh=hourly_buy_kwh,
        grid_sell_kwh=total_sell_kwh,
        grid_sell_revenue=total_sell_kwh * config.GRID_FEEDIN_PRICE,
        wind_gen_kwh=total_wind_kwh,
        solar_gen_kwh=total_solar_kwh,
        equipment_depreciation=calc_total_depreciation(daily_output),
        p_alk_mw=p_alk_vals,
        p_pem_mw=p_pem_vals,
        p_nh3_mw=p_nh3_vals,
    )

    result = {
        "status": "Optimal" if prob.status == 1 else str(prob.status),
        "objective": obj_val,
        "p_schedule": p_vals,
        "p_alk_schedule": p_alk_vals,
        "p_pem_schedule": p_pem_vals,
        "p_nh3_schedule": p_nh3_vals,
        "z_schedule": z_vals,
        "buy_schedule": buy_vals,
        "sell_schedule": sell_vals,
        "total_p_mwh": total_p_mwh,
        "total_alk_mwh": total_alk_mwh,
        "total_pem_mwh": total_pem_mwh,
        "total_nh3_mwh": total_nh3_mwh,
        "hourly_buy_kwh": hourly_buy_kwh,
        "total_sell_kwh": total_sell_kwh,
        "total_wind_kwh": total_wind_kwh,
        "total_solar_kwh": total_solar_kwh,
        "total_buy_kwh": total_buy_kwh,
        "total_load_kwh": total_load_kwh,
        "el_om_cost": el_om_cost,
        "nh3_om_cost": nh3_om_cost,
        "ton_ammonia_cost": ton_cost,
        "h2_produced_kg": h2_produced_kg,
        "h2_consumed_kg": h2_consumed_kg,
        "green_penalty": value(green_penalty),
    }
    _verify_solution(result, wind, solar, load, daily_output)
    return result


# ============================================================
# 第一部分：典型场景分析
# ============================================================
def run_typical_scenario():
    """典型场景下 5 个产量级别的 LP 求解。"""
    wind = config.get_typical_wind()
    solar = config.get_typical_solar()
    load = config.get_load_profile()

    results = []
    failed = []
    for daily_output in PRODUCTION_LEVELS:
        print(f"\n--- 典型场景: 日产量 {daily_output} 吨/天 ---")
        try:
            res = build_and_solve_lp(wind, solar, load, daily_output)
        except (RuntimeError, ValueError) as e:
            print(f"FAILED: {e}")
            failed.append((daily_output, str(e)))
            continue
        results.append(res)

        # Print schedule table
        print(f"{'小时':>4} {'p(MW)':>8} {'z':>3} {'买电MW':>8} {'卖电MW':>8} {'风电MW':>8} {'光伏MW':>8} {'负荷MW':>8}")
        for t in range(24):
            print(
                f"{t:>4} {res['p_schedule'][t]:>8.2f} {res['z_schedule'][t]:>3} "
                f"{res['buy_schedule'][t]:>8.2f} {res['sell_schedule'][t]:>8.2f} "
                f"{wind[t]:>8.2f} {solar[t]:>8.2f} {load[t]:>8.2f}"
            )
        print(f"等效满负荷小时: {res['total_p_mwh']/(ALK_SYS_POWER+PEM_SYS_POWER+NH3_SYS_POWER):.1f}h (ALK={res['total_alk_mwh']:.0f} PEM={res['total_pem_mwh']:.0f} NH3={res['total_nh3_mwh']:.0f})")
        print(f"吨氨成本: {res['ton_ammonia_cost']:.2f} ¥/ton")

    if failed:
        print(f"\n典型场景失败: {len(failed)}/{len(PRODUCTION_LEVELS)} 个产量级别")
        for dl, err in failed:
            print(f"  产量{dl}t/d: {err}")

    if not results:
        print("典型场景全部求解失败，无法继续")
        return results, None, failed

    # Find best production level
    best = min(results, key=lambda r: r["ton_ammonia_cost"])
    best_idx = PRODUCTION_LEVELS.index(
        next(
            dl for dl, r in zip(PRODUCTION_LEVELS, results)
            if r["ton_ammonia_cost"] == best["ton_ammonia_cost"]
        )
    )
    best_daily = PRODUCTION_LEVELS[best_idx]
    print(f"\n最优产量: {best_daily} 吨/天, 吨氨成本: {best['ton_ammonia_cost']:.2f} ¥/ton")

    return results, best_daily, failed


# ============================================================
# 第二部分：全部 24 场景 × 5 产量（共 120 次求解）
# ============================================================
def run_all_scenarios():
    """全部 24 种风光场景 × 5 个产量级别的 MILP 求解。"""
    scenarios = generate_all_scenarios()  # [(w,s), ...]
    all_results = []  # list of dicts per scenario × level
    failed_scenarios = []  # (scenario_name, daily_output, error_msg)

    total_solves = len(scenarios) * len(PRODUCTION_LEVELS)
    count = 0

    for w, s in scenarios:
        wind = config.get_wind_profile(w)
        solar = config.get_solar_profile(s)
        load = config.get_load_profile()
        scenario_name = f"W{w}S{s}"

        for daily_output in PRODUCTION_LEVELS:
            count += 1
            print(f"[{count}/{total_solves}] {scenario_name} 产量{daily_output}t/d ...", end=" ")
            try:
                res = build_and_solve_lp(wind, solar, load, daily_output)
            except (RuntimeError, ValueError) as e:
                print(f"FAILED: {e}")
                failed_scenarios.append((scenario_name, daily_output, str(e)))
                continue

            # Compute green indicators
            indicators = calc_green_indicators(
                total_gen=res["total_wind_kwh"] + res["total_solar_kwh"],
                total_load=res["total_load_kwh"],
                grid_buy=res["total_buy_kwh"],
                grid_sell=res["total_sell_kwh"],
            )
            classification = classify_scenario(indicators)

            all_results.append({
                "scenario": scenario_name,
                "wind_idx": w,
                "solar_idx": s,
                "daily_output": daily_output,
                "ton_ammonia_cost": res["ton_ammonia_cost"],
                "classification": classification,
                "status": res["status"],
                "eta_self": indicators["self_use_rate"],
                "eta_green": indicators["green_rate"],
                "eta_grid": indicators["grid_feed_rate"],
                "total_buy_kwh": res["total_buy_kwh"],
                "total_sell_kwh": res["total_sell_kwh"],
                "total_wind_kwh": res["total_wind_kwh"],
                "total_solar_kwh": res["total_solar_kwh"],
                "total_load_kwh": res["total_load_kwh"],
                "total_p_mwh": res["total_p_mwh"],
                "total_alk_mwh": res["total_alk_mwh"],
                "total_pem_mwh": res["total_pem_mwh"],
                "total_nh3_mwh": res["total_nh3_mwh"],
                "green_penalty": res["green_penalty"],
                "el_om_cost": res["el_om_cost"],
                "nh3_om_cost": res["nh3_om_cost"],
            })
            print(f"OK 成本={res['ton_ammonia_cost']:.2f} {classification}")

    print(f"\n全部{len(all_results)}个LP求解完成")

    if failed_scenarios:
        print(f"\n{'='*60}")
        print(f"失败场景: {len(failed_scenarios)}/{total_solves} 个求解失败:")
        for name, dl, err in failed_scenarios:
            print(f"  {name} 产量{dl}t/d: {err}")
        print(f"{'='*60}")

    # Annual statistics
    _compute_annual_stats(all_results, scenarios)
    # Cost distribution histogram
    _plot_cost_distribution(all_results)
    # Save summary CSV
    _save_annual_csv(all_results)
    # Q2 vs Q3 comparison
    _compare_q2_q3(all_results)
    # Save comparison CSV
    _save_comparison_csv(all_results)

    # Visualization and summary
    _plot_boxplot_costs(all_results)
    _plot_scatter_cost_vs_green(all_results)
    _plot_pie_compliance(all_results)
    _plot_cost_breakdown(all_results)
    _print_summary_table(all_results)
    _cluster_and_radar(all_results)

    return all_results, failed_scenarios


def _compute_annual_stats(all_results, scenarios):
    """计算年度分类统计与加权平均成本。"""
    days_per_scenario = 15

    classification_counts = {"全满足": 0, "部分满足": 0, "全不满足": 0}
    total_cost_sum = 0.0
    total_count = 0

    for r in all_results:
        cls = r["classification"]
        if cls in classification_counts:
            classification_counts[cls] += days_per_scenario
        total_cost_sum += r["ton_ammonia_cost"] * days_per_scenario
        total_count += days_per_scenario

    avg_annual_cost = total_cost_sum / total_count if total_count > 0 else 0

    print("\n=== 年度统计 ===")
    print(f"全满足天数: {classification_counts['全满足']}")
    print(f"部分满足天数: {classification_counts['部分满足']}")
    print(f"全不满足天数: {classification_counts['全不满足']}")
    print(f"加权平均吨氨成本: {avg_annual_cost:.2f} ¥/ton")

    # Find best scenario × level
    best = min(all_results, key=lambda r: r["ton_ammonia_cost"])
    print(
        f"最优方案: {best['scenario']} "
        f"产量{best['daily_output']}t/d "
        f"成本{best['ton_ammonia_cost']:.2f} ¥/ton "
        f"分类{best['classification']}"
    )

    return classification_counts, avg_annual_cost


def _plot_cost_distribution(all_results):
    """绘制年度吨氨成本分布直方图。"""
    costs = [r["ton_ammonia_cost"] for r in all_results]

    plt.figure(figsize=(10, 6))
    plt.hist(costs, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    plt.xlabel("吨氨成本 (¥/ton)")
    plt.ylabel("频次 (场景×产量)")
    plt.title("年度吨氨成本分布 (120个LP求解 — 连续制氨调节)")
    plt.grid(axis="y", alpha=0.3)

    # Add mean and median lines
    mean_cost = np.mean(costs)
    median_cost = np.median(costs)
    plt.axvline(mean_cost, color="red", linestyle="--", linewidth=1.5,
                label=f"均值: {mean_cost:.0f} ¥/ton")
    plt.axvline(median_cost, color="orange", linestyle="--", linewidth=1.5,
                label=f"中位数: {median_cost:.0f} ¥/ton")
    plt.legend()

    png_path = "q3_annual_cost_distribution.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"成本分布图已保存: {png_path}")


def _save_annual_csv(all_results):
    """保存年度汇总结果到 CSV（V2 格式，中文字段名）。"""
    csv_path = os.path.join(SCRIPT_DIR, "q3_annual_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "场景", "风电档位", "光伏档位", "日产量(吨/天)", "吨氨成本(元/吨)",
            "eta_self", "eta_green", "eta_grid",
            "日购电量(kWh)", "日售电量(kWh)", "日发电量(kWh)", "日用电量(kWh)",
            "分类", "求解状态",
            "绿电惩罚(元)",
        ])
        for r in all_results:
            # 等效满负荷小时 = total_p_mwh / 总设备功率
            eq_hours = r["total_p_mwh"] / (P_ALKEL + P_PEMEL + P_NH3)
            total_gen = r["total_wind_kwh"] + r["total_solar_kwh"]
            writer.writerow([
                r["scenario"], r["wind_idx"], r["solar_idx"],
                r["daily_output"], f"{r['ton_ammonia_cost']:.2f}",
                f"{r['eta_self']:.4f}", f"{r['eta_green']:.4f}", f"{r['eta_grid']:.4f}",
                f"{r['total_buy_kwh']:.2f}",
                f"{r['total_sell_kwh']:.2f}",
                f"{total_gen:.2f}",
                f"{r['total_load_kwh']:.2f}",
                r["classification"], r["status"],
                f"{r['green_penalty']:.2f}",
            ])
    print(f"年度汇总已保存: {csv_path}")


def _compare_q2_q3(all_q3_results):
    """Q3 LP 结果与 Q2 MILP 基准对比。"""
    q2_csv = os.path.join(os.path.dirname(__file__), "..", "q2", "q2_annual_summary.csv")
    if not os.path.exists(q2_csv):
        print("[警告] 未找到 Q2 结果文件，跳过对比")
        return None

    # 构建 Q2 查找表: (场景, 日产量) -> {q2_cost, classification}
    q2_lookup = {}
    with open(q2_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["场景"], int(float(row["日产量(吨/天)"])))
            q2_lookup[key] = {
                "cost": float(row["吨氨成本(¥/ton)"]),
                "classification": row["分类"],
            }

    # Build comparison row for each Q3 result
    comparisons = []
    total_diff = 0.0
    n_better = 0
    n_equal = 0
    n_worse = 0

    for r in all_q3_results:
        q3_cost = r["ton_ammonia_cost"]
        q3_output = r["daily_output"]
        q3_scenario = r["scenario"]

        key = (q3_scenario, int(q3_output))
        if key in q2_lookup:
            q2_cost = q2_lookup[key]["cost"]
            diff = q3_cost - q2_cost
            diff_pct = (diff / q2_cost * 100) if q2_cost != 0 else 0.0

            comparisons.append({
                "scenario": q3_scenario,
                "daily_output": q3_output,
                "q2_cost": q2_cost,
                "q3_cost": q3_cost,
                "diff": diff,
                "diff_pct": diff_pct,
            })

            total_diff += diff
            if diff < -1e-6:
                n_better += 1
            elif diff > 1e-6:
                n_worse += 1
            else:
                n_equal += 1

    if comparisons:
        avg_diff = total_diff / len(comparisons) if comparisons else 0.0
        print(f"\n=== Q2 vs Q3 成本对比 ===")
        print(f"对比场景数: {len(comparisons)}")
        print(f"Q3更优: {n_better}, Q3更差: {n_worse}, 持平: {n_equal}")
        print(f"平均成本差异: {avg_diff:+.2f} 元/吨")
        if n_better > 0:
            better_diffs = [c["diff"] for c in comparisons if c["diff"] < -1e-6]
            print(f"平均降幅: {np.mean(better_diffs):.2f} ¥/ton ({abs(np.mean(better_diffs))/np.mean([c['q2_cost'] for c in comparisons if c['diff'] < -1e-6])*100:.1f}%)" if better_diffs else "")

    return comparisons


def _save_comparison_csv(all_q3_results):
    """保存 Q2 vs Q3 对比到 CSV。"""
    q2_csv = os.path.join(os.path.dirname(__file__), "..", "q2", "q2_annual_summary.csv")
    if not os.path.exists(q2_csv):
        return

    # Build Q2 lookup
    q2_lookup = {}
    with open(q2_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["场景"], int(float(row["日产量(吨/天)"])))
            q2_lookup[key] = float(row["吨氨成本(¥/ton)"])

    cmp_path = os.path.join(SCRIPT_DIR, "q3_vs_q2_comparison.csv")
    with open(cmp_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "场景", "产量", "吨氨成本(Q2)", "吨氨成本(Q3)", "成本差异", "成本降幅%",
        ])
        for r in all_q3_results:
            key = (r["scenario"], int(r["daily_output"]))
            if key in q2_lookup:
                q2_cost = q2_lookup[key]
                q3_cost = r["ton_ammonia_cost"]
                diff = q3_cost - q2_cost
                diff_pct = (diff / q2_cost * 100) if q2_cost != 0 else 0.0
                writer.writerow([
                    r["scenario"],
                    r["daily_output"],
                    f"{q2_cost:.2f}",
                    f"{q3_cost:.2f}",
                    f"{diff:.2f}",
                    f"{diff_pct:.2f}",
                ])
    print(f"Q2-Q3对比已保存: {cmp_path}")


# ============================================================
# 新增可视化函数
# ============================================================
def _plot_boxplot_costs(all_results):
    """各产量级别吨氨成本箱线图。"""
    import matplotlib.pyplot as plt
    levels = sorted(set(r["daily_output"] for r in all_results), reverse=True)
    data_by_level = {lv: [] for lv in levels}
    for r in all_results:
        data_by_level[r["daily_output"]].append(r["ton_ammonia_cost"])
    plot_data = [data_by_level[lv] for lv in levels]

    plt.figure(figsize=(10, 6))
    bp = plt.boxplot(plot_data, labels=[str(lv) for lv in levels], patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightblue")
    plt.xlabel("日产量 (吨/天)")
    plt.ylabel("吨氨成本 (¥/ton)")
    plt.title("Q3 各产量级别吨氨成本分布 (N=120)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("q3/q3_boxplot.png", dpi=150)
    plt.close()
    print("箱线图已保存: q3/q3_boxplot.png")


def _plot_scatter_cost_vs_green(all_results):
    """吨氨成本 vs 绿电比例散点图。"""
    import matplotlib.pyplot as plt
    levels = sorted(set(r["daily_output"] for r in all_results), reverse=True)
    colors = plt.cm.viridis(np.linspace(0, 1, len(levels)))
    level_color = {lv: c for lv, c in zip(levels, colors)}

    marker_map = {"全满足": "o", "部分满足": "^", "全不满足": "x"}

    plt.figure(figsize=(12, 7))
    for r in all_results:
        c = level_color[r["daily_output"]]
        m = marker_map.get(r["classification"], "o")
        plt.scatter(r["eta_green"], r["ton_ammonia_cost"],
                    c=[c], marker=m, s=60, alpha=0.7,
                    label=f"{r['daily_output']}t/d" if r is all_results[0] else "")

    # Build legend for production levels
    from matplotlib.lines import Line2D
    legend_handles = [Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=level_color[lv], markersize=8,
                              label=f"{lv} t/d") for lv in levels]
    legend_handles += [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
               markersize=8, label="全满足 (○)"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
               markersize=8, label="部分满足 (△)"),
        Line2D([0], [0], marker="x", color="w", markerfacecolor="gray",
               markersize=8, label="全不满足 (×)"),
    ]
    plt.legend(handles=legend_handles, loc="upper left", fontsize=8)

    plt.axvline(x=0.30, color="red", linestyle="--", linewidth=1.5,
                label="η_green=0.30 阈值")
    plt.xlabel("η_green (绿电比例)")
    plt.ylabel("吨氨成本 (¥/ton)")
    plt.title("Q3 吨氨成本 vs 绿电比例 (N=120)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("q3/q3_scatter.png", dpi=150)
    plt.close()
    print("散点图已保存: q3/q3_scatter.png")


def _plot_pie_compliance(all_results):
    """绿电合规分类饼图。"""
    import matplotlib.pyplot as plt
    counts = {"全满足": 0, "部分满足": 0, "全不满足": 0}
    for r in all_results:
        cls = r["classification"]
        if cls in counts:
            counts[cls] += 1

    labels = []
    sizes = []
    colors = ["#2ca02c", "#ff7f0e", "#d62728"]  # green, orange, red
    for cls, color in zip(["全满足", "部分满足", "全不满足"], colors):
        if counts[cls] > 0:
            labels.append(f"{cls}\n({counts[cls]}, {counts[cls]/len(all_results)*100:.1f}%)")
            sizes.append(counts[cls])

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, colors=colors[:len(sizes)],
            autopct="", startangle=90,
            textprops={"fontsize": 12})
    plt.title("Q3 绿电合规分类占比 (N=120)")
    plt.tight_layout()
    plt.savefig("q3/q3_pie.png", dpi=150)
    plt.close()
    print("饼图已保存: q3/q3_pie.png")


def _plot_cost_breakdown(all_results):
    """各产量级别 W0S0 情景成本构成堆叠柱状图。"""
    import matplotlib.pyplot as plt
    levels = sorted(set(r["daily_output"] for r in all_results), reverse=True)
    w0s0_data = {}
    for r in all_results:
        if r["scenario"] == "W0S0":
            w0s0_data[r["daily_output"]] = r

    if not w0s0_data:
        print("[警告] 未找到W0S0场景数据，跳过成本构成图")
        return

    categories = []
    grid_costs, wind_costs, solar_costs = [], [], []
    el_om_costs, nh3_om_costs, sell_revenues = [], [], []

    for lv in levels:
        if lv not in w0s0_data:
            continue
        r = w0s0_data[lv]
        categories.append(f"{lv}t/d")
        grid_costs.append(r["total_buy_kwh"] * 0.6)
        wind_costs.append(r["total_wind_kwh"] * 0.15)
        solar_costs.append(r["total_solar_kwh"] * 0.12)
        el_om_costs.append(r["el_om_cost"])
        nh3_om_costs.append(r["nh3_om_cost"])
        sell_revenues.append(-r["total_sell_kwh"] * 0.3779)

    x = np.arange(len(categories))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 7))

    # Stack positive costs upward
    p1 = ax.bar(x, grid_costs, width, label="购电成本", color="gray")
    p2 = ax.bar(x, wind_costs, width, bottom=grid_costs,
                label="风电成本", color="lightblue")
    bottom2 = [g + w for g, w in zip(grid_costs, wind_costs)]
    p3 = ax.bar(x, solar_costs, width, bottom=bottom2,
                label="光伏成本", color="yellow")
    bottom3 = [b + s for b, s in zip(bottom2, solar_costs)]
    p4 = ax.bar(x, el_om_costs, width, bottom=bottom3,
                label="电解运维", color="orange")
    bottom4 = [b + e for b, e in zip(bottom3, el_om_costs)]
    p5 = ax.bar(x, nh3_om_costs, width, bottom=bottom4,
                label="合成氨运维", color="green")
    bottom5 = [b + n for b, n in zip(bottom4, nh3_om_costs)]
    # Sell revenue as negative (below zero)
    p6 = ax.bar(x, sell_revenues, width, label="售电收入 (负)", color="red")

    ax.set_xlabel("日产量级别")
    ax.set_ylabel("成本/收入 (¥)")
    ax.set_title("Q3 各产量级别 W0S0 成本构成")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.axhline(y=0, color="black", linewidth=0.8)

    for i in range(len(categories)):
        total_pos = bottom5[i] if i < len(bottom5) else 0
        ax.text(i, total_pos + max(bottom5) * 0.02,
                f"{total_pos:.0f}", ha="center", fontsize=8)

    plt.tight_layout()
    plt.savefig("q3/q3_cost_breakdown.png", dpi=150)
    plt.close()
    print("成本构成图已保存: q3/q3_cost_breakdown.png")


def _print_summary_table(all_results):
    """打印汇总表格。"""
    levels = sorted(set(r["daily_output"] for r in all_results), reverse=True)

    # (a) By production level
    print("\n" + "=" * 75)
    print("(a) 各产量级别成本统计")
    print("=" * 75)
    print(f"{'产量(t/d)':<10} {'最低成本':<12} {'最高成本':<12} {'平均成本':<12} {'合规率':<10}")
    print("-" * 56)
    total_compliant = 0
    for lv in levels:
        lv_results = [r for r in all_results if r["daily_output"] == lv]
        costs = [r["ton_ammonia_cost"] for r in lv_results]
        compliant = sum(1 for r in lv_results if r["classification"] == "全满足")
        compliance_rate = compliant / len(lv_results) * 100 if lv_results else 0
        total_compliant += compliant
        print(f"{lv:<10} {min(costs):<12.2f} {max(costs):<12.2f} "
              f"{np.mean(costs):<12.2f} {compliance_rate:<10.1f}%")
    print("-" * 56)
    print(f"合计合规率: {total_compliant}/{len(all_results)} ({total_compliant/len(all_results)*100:.1f}%)")

    # (b) Top 5 best
    sorted_results = sorted(all_results, key=lambda r: r["ton_ammonia_cost"])
    print("\n" + "=" * 75)
    print("(b) Top 5 最优方案 (按吨氨成本)")
    print("=" * 75)
    print(f"{'排名':<6} {'场景':<12} {'产量(t/d)':<10} {'吨氨成本':<12} "
          f"{'η_green':<10} {'分类':<10}")
    print("-" * 60)
    for i, r in enumerate(sorted_results[:5], 1):
        print(f"{i:<6} {r['scenario']:<12} {r['daily_output']:<10} "
              f"{r['ton_ammonia_cost']:<12.2f} {r['eta_green']:<10.3f} "
              f"{r['classification']:<10}")

    # (c) Bottom 5 worst
    print("\n" + "=" * 75)
    print("(c) Bottom 5 最差方案 (按吨氨成本)")
    print("=" * 75)
    print(f"{'排名':<6} {'场景':<12} {'产量(t/d)':<10} {'吨氨成本':<12} "
          f"{'η_green':<10} {'分类':<10}")
    print("-" * 60)
    for i, r in enumerate(sorted_results[-5:][::-1], 1):
        print(f"{i:<6} {r['scenario']:<12} {r['daily_output']:<10} "
              f"{r['ton_ammonia_cost']:<12.2f} {r['eta_green']:<10.3f} "
              f"{r['classification']:<10}")
    print("=" * 75)


# ============================================================
# 聚类分析 + 雷达图（72t/d 子集）
# ============================================================
try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _manual_standardize(X):
    """手动 z-score 标准化：(x-mean)/std"""
    X = np.array(X, dtype=np.float64)
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-12] = 1.0  # 防止除零
    return (X - mean) / std


def _manual_kmeans(X, k, max_iters=100, seed=42):
    """手动实现 k-means，返回 labels 和 centroids。"""
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    # 随机选择 k 个点作为初始质心
    idx = rng.choice(n, k, replace=False)
    centroids = X[idx].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iters):
        # 分配：计算每个点到各质心的欧氏距离
        dists = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(dists, axis=1)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # 更新质心
        for j in range(k):
            mask = (labels == j)
            if np.any(mask):
                centroids[j] = X[mask].mean(axis=0)

    return labels, centroids


def _manual_silhouette(X, labels):
    """手动计算轮廓系数（欧氏距离）。"""
    n = X.shape[0]
    if n <= 1:
        return 0.0

    # 成对距离矩阵
    dist = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))

    s_vals = np.zeros(n)
    unique_labels = np.unique(labels)
    if len(unique_labels) <= 1:
        return 0.0

    for i in range(n):
        li = labels[i]
        same_mask = (labels == li)
        same_mask[i] = False
        if np.any(same_mask):
            a = np.mean(dist[i, same_mask])
        else:
            a = 0.0

        b_vals = []
        for other_l in unique_labels:
            if other_l == li:
                continue
            other_mask = (labels == other_l)
            if np.any(other_mask):
                b_vals.append(np.mean(dist[i, other_mask]))
        b = min(b_vals) if b_vals else 0.0
        denom = max(a, b)
        s_vals[i] = (b - a) / denom if denom > 0 else 0.0

    return float(np.mean(s_vals))


def _name_cluster(feature_means, global_mean, feature_names):
    """根据簇特征均值 vs 全局均值命名簇。"""
    above = feature_means >= global_mean
    below = feature_means < global_mean

    descriptors_high = {
        "吨氨成本": "高成本",
        "η_self": "高自消纳",
        "η_green": "高绿电",
        "η_grid": "高反送",
        "日购电量": "高购电",
        "日售电量": "高售电",
    }
    descriptors_low = {
        "吨氨成本": "低成本",
        "η_self": "低自消纳",
        "η_green": "低绿电",
        "η_grid": "低反送",
        "日购电量": "低购电",
        "日售电量": "低售电",
    }

    parts = []
    for i, name in enumerate(feature_names):
        if above[i]:
            parts.append(descriptors_high[name])
        else:
            parts.append(descriptors_low[name])

    # 取前两个最显著的偏离（按 |mean-global| 排序）
    deviations = np.abs(feature_means - global_mean)
    top_indices = np.argsort(deviations)[::-1][:2]
    top_parts = [parts[i] for i in top_indices]
    return "-".join(top_parts)


def _cluster_and_radar(all_results):
    """对 72t/d 子集进行聚类分析并绘制雷达图。"""
    import matplotlib.pyplot as plt

    # 1. 筛选 72t/d
    subset = [r for r in all_results if r["daily_output"] == 72]
    if len(subset) < 4:
        print(f"[聚类] 72t/d 仅 {len(subset)} 条记录，不足以聚类，跳过")
        return

    print(f"\n[聚类] 72t/d 子集: {len(subset)} 条记录")

    # 2. 构建 6 维特征矩阵
    features = np.array([
        [r["ton_ammonia_cost"], r["eta_self"], r["eta_green"],
         r["eta_grid"], r["total_buy_kwh"], r["total_sell_kwh"]]
        for r in subset
    ])
    feature_names = ["吨氨成本", "η_self", "η_green", "η_grid", "日购电量", "日售电量"]

    # 3. 标准化
    if HAS_SKLEARN:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(features)
    else:
        X_scaled = _manual_standardize(features)

    # 4. 聚类
    best_k = 2
    best_sil = -1.0
    best_labels = None

    for k in (2, 3, 4):
        if k >= len(subset):
            break
        try:
            if HAS_SKLEARN:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = km.fit_predict(X_scaled)
            else:
                labels, _ = _manual_kmeans(X_scaled, k)
        except Exception as e:
            print(f"  K={k} 聚类失败: {e}")
            continue

        unique_cnt = len(np.unique(labels))
        if unique_cnt < k:
            print(f"  K={k} 空洞簇 (仅{unique_cnt}个非空)，跳过")
            continue

        try:
            if HAS_SKLEARN:
                sil = silhouette_score(X_scaled, labels)
            else:
                sil = _manual_silhouette(X_scaled, labels)
        except Exception as e:
            print(f"  K={k} 轮廓系数计算失败: {e}")
            continue

        print(f"  K={k}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_labels = labels

    if best_labels is None:
        print("[聚类] 所有 k 均失败，跳过")
        return

    print(f"  最佳 K={best_k} (silhouette={best_sil:.4f})")

    # 5. 保存聚类报告
    report_path = os.path.join(os.path.dirname(__file__), "q3_cluster_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"q3 聚类分析报告 (72t/d 子集, N={len(subset)})\n")
        f.write(f"最佳聚类数 K={best_k}, 轮廓系数={best_sil:.4f}\n\n")
        for ci in range(best_k):
            mask = (best_labels == ci)
            cluster_rows = features[mask]
            cluster_scenarios = [subset[i]["scenario"] for i, m in enumerate(mask) if m]
            f.write(f"--- 簇 {ci+1} (n={np.sum(mask)}) ---\n")
            f.write(f"场景: {', '.join(cluster_scenarios)}\n")
            for j, name in enumerate(feature_names):
                f.write(f"  {name}: mean={cluster_rows[:, j].mean():.4f}, std={cluster_rows[:, j].std():.4f}\n")
            f.write("\n")
    print(f"  聚类报告已保存: {report_path}")

    # 6. 生成雷达图
    # Min-Max 归一化到 [0,1]
    feat_min = features.min(axis=0)
    feat_max = features.max(axis=0)
    feat_range = feat_max - feat_min
    feat_range[feat_range < 1e-12] = 1.0
    feat_norm = (features - feat_min) / feat_range

    # 计算全局均值用于命名
    global_mean = features.mean(axis=0)

    angles = np.linspace(0, 2 * np.pi, len(feature_names), endpoint=False).tolist()
    angles += angles[:1]  # 闭合

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, best_k))

    for ci in range(best_k):
        mask = (best_labels == ci)
        cluster_mean_norm = feat_norm[mask].mean(axis=0)
        cluster_mean_raw = features[mask].mean(axis=0)
        cluster_name = _name_cluster(cluster_mean_raw, global_mean, feature_names)

        values = cluster_mean_norm.tolist()
        values += values[:1]
        ax.fill(angles, values, alpha=0.25, color=colors[ci], label=f"簇{ci+1}: {cluster_name}")
        ax.plot(angles, values, color=colors[ci], linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"72t/d 子集聚类雷达图 (K={best_k}, N={len(subset)})", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    radar_path = os.path.join(os.path.dirname(__file__), "q3_radar.png")
    plt.tight_layout()
    plt.savefig(radar_path, dpi=150)
    plt.close()
    print(f"  雷达图已保存: {radar_path}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("问题三: 连续制氨调节 MILP 调度优化")
    print("=" * 60)

    total_failed = 0

    # Part (1): Typical scenario
    print("\n>>> 第一部分: 典型场景分析")
    _, _, failed_typical = run_typical_scenario()
    total_failed += len(failed_typical)

    # Part (2): All scenarios
    print("\n>>> 第二部分: 全部24场景×5产量 (共120个LP)")
    _, failed_scenarios = run_all_scenarios()
    total_failed += len(failed_scenarios)

    print("\n问题三完成")

    if total_failed > 0:
        print(f"\n错误: {total_failed} 个场景求解或校验失败，退出码=1")
        sys.exit(1)
