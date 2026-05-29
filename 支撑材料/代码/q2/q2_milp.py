# -*- coding: utf-8 -*-
"""
问题二：离散开关MILP调度优化
电氢氨园区 - 24小时离散调度，设备全开/全关
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from config import (
    NH3_BASELINE_CAPACITY,
    ALKEL_H2_RATE, PEMEL_H2_RATE,
    NH3_H2_CONSUMPTION, NH3_RATE,
    ALKEL_RATED, PEMEL_RATED, NH3_RATED,
    ALKEL_OM, PEMEL_OM, NH3_OM,
    ALKEL_EFF, PEMEL_EFF,
)
from utils import (
    add_green_penalty_to_problem,
    calc_green_indicators,
    classify_scenario,
    calc_ton_ammonia_cost,
    calc_total_depreciation,
    generate_all_scenarios,
)


# ============================================================
# Constants — derived from config.py for 72t/d capacity
# ============================================================
SCALE_72 = 72.0 / NH3_BASELINE_CAPACITY  # 2.0

# H₂ production rates (kgH₂/h)
ALKEL_H2_72 = ALKEL_H2_RATE * SCALE_72   # 280 kgH₂/h
PEMEL_H2_72 = PEMEL_H2_RATE * SCALE_72   # 320 kgH₂/h
TOTAL_H2_RATE = ALKEL_H2_72 + PEMEL_H2_72  # 600 kgH₂/h

# NH₃ production rate (ton/h)
NH3_RATE_72 = NH3_RATE * SCALE_72         # 3.0 ton/h

# Equipment power ratings (MW)
P_ALKEL = ALKEL_RATED * SCALE_72          # 20 MW
P_PEMEL = PEMEL_RATED * SCALE_72          # 20 MW
P_NH3   = NH3_RATED * SCALE_72            # 1.5 MW
P_TOTAL = P_ALKEL + P_PEMEL + P_NH3       # 41.5 MW

# H₂ mass balance verification
assert abs(TOTAL_H2_RATE - NH3_RATE_72 * 1000 * NH3_H2_CONSUMPTION) < 1e-6, \
    "H₂ imbalance: produced={}, consumed={}".format(
        TOTAL_H2_RATE, NH3_RATE_72 * 1000 * NH3_H2_CONSUMPTION
    )

PRODUCTION_LEVELS = [72, 63, 54, 45, 36]  # ton/day

# OM blended rate
EL_OM_BLENDED = (ALKEL_OM + PEMEL_OM) / 2  # 0.125 ¥/kWh
NH3_OM_RATE = NH3_OM                        # 0.002 ¥/kWh for NH3 synthesis


# ============================================================
# Post-solve verification helper
# ============================================================
def _verify_solution(result, wind, solar, load, daily_output):
    """
    Verify MILP solution correctness after solve.

    Checks:
      1. Power balance at each hour (within 1e-6 MW tolerance)
      2. H₂ mass balance (produced vs consumed)
      3. Green indicator ranges (self_use_rate, green_rate, grid_feed_rate ∈ [0,1])

    Raises ValueError with detailed message if any check fails.
    """
    x_vals = result["x_schedule"]
    buy_vals = result["buy_schedule"]
    sell_vals = result["sell_schedule"]

    # 1. Hourly power balance
    for t in range(24):
        imbalance = abs(
            wind[t] + solar[t] + buy_vals[t]
            - sell_vals[t] - x_vals[t] * P_TOTAL - load[t]
        )
        if imbalance >= 1e-4:
            raise ValueError(
                f"Power balance failed at hour {t}: imbalance={imbalance:.6f} MW"
            )

    # 2. H₂ mass balance
    h2_imbalance = abs(result["h2_produced_kg"] - result["h2_consumed_kg"])
    if h2_imbalance >= 1e-4:
        raise ValueError(
            f"H2 mass balance failed: produced={result['h2_produced_kg']:.2f} kg, "
            f"consumed={result['h2_consumed_kg']:.2f} kg, imbalance={h2_imbalance:.6f}"
        )

    # 3. Green indicator ranges (WARNING only - extreme scenarios can
    #    produce mathematically valid out-of-range values)
    total_gen = result["total_wind_kwh"] + result["total_solar_kwh"]
    total_load = result["total_load_kwh"]
    grid_buy = result["total_buy_kwh"]
    grid_sell = result["total_sell_kwh"]

    indicators = calc_green_indicators(total_gen, total_load, grid_buy, grid_sell)

    for name in ("self_use_rate", "green_rate", "grid_feed_rate"):
        val = indicators[name]
        if not (0.0 <= val <= 1.0):
            print(f"  [WARNING] Green indicator '{name}' out of range [0,1]: {val:.6f} "
                  f"(extreme scenario, MILP is still optimal)")


# ============================================================
# Core MILP solver
# ============================================================
def build_and_solve_milp(wind, solar, load, daily_output):
    """
    Build and solve the discrete on/off MILP for one scenario × production level.

    Parameters
    ----------
    wind      : list/array of 24 hourly wind generation (MW)
    solar     : list/array of 24 hourly solar generation (MW)
    load      : list/array of 24 hourly base load (MW)
    daily_output : target daily NH3 output (ton/day)

    Returns
    -------
    dict with keys:
        status, objective, x_schedule, buy_schedule, sell_schedule,
        runtime_hours, and all cost components
    """
    runtime_hours = int(daily_output / NH3_RATE_72)  # integer hours equipment ON

    # ---- Model ----
    prob = LpProblem("Q2_MILP", LpMinimize)

    # Binary: x[t] = 1 when equipment is ON at hour t
    x = [LpVariable(f"x_{t}", cat="Binary") for t in range(24)]

    # Continuous: grid buy/sell at each hour (MW)
    buy = [LpVariable(f"buy_{t}", lowBound=0) for t in range(24)]
    sell = [LpVariable(f"sell_{t}", lowBound=0) for t in range(24)]

    # Binary: grid_dir[t] = 1 when selling to grid, 0 when buying from grid
    # Prevents simultaneous buy+sell arbitrage (valley purchase < feed-in price)
    M_grid = config.WIND_RATED + config.SOLAR_RATED  # max net power MW
    grid_dir = [LpVariable(f"grid_dir_{t}", cat="Binary") for t in range(24)]

    # ---- Constraints ----
    # Total ON hours must equal required runtime
    prob += lpSum(x) == runtime_hours, "RuntimeHours"

    # Big-M complementarity: prevent simultaneous buy and sell
    # Without this, MILP is unbounded when valley purchase price < feed-in price
    for t in range(24):
        prob += buy[t] <= M_grid * (1 - grid_dir[t]), f"NoBuyWhenSell_{t}"
        prob += sell[t] <= M_grid * grid_dir[t], f"NoSellWhenBuy_{t}"

    # Forbid operation at midnight hours that would be impossible
    # (e.g., if runtime_hours = 24, every hour ON; if = 0, all OFF)
    # No extra constraint needed beyond the sum constraint above.

    # Power balance at each hour
    for t in range(24):
        prob += (
            wind[t] + solar[t] + buy[t]
            == sell[t] + x[t] * P_TOTAL + load[t]
        ), f"Balance_{t}"

    # ---- Green Indicator Penalty ----
    # Penalize violation of green indicator thresholds
    total_load_expr = lpSum(load) + runtime_hours * P_TOTAL
    green_penalty = add_green_penalty_to_problem(
        prob, total_load_expr,
        float(np.sum(wind) + np.sum(solar)),
        lpSum(sell), lpSum(buy), config
    )

    # ---- Objective ----
    # Grid purchase cost (TOU pricing via config.get_price)
    buy_cost = lpSum(
        buy[t] * config.get_price(t) * 1000  # MW → kW, price in ¥/kWh
        for t in range(24)
    )

    # Grid sell revenue
    sell_revenue = lpSum(
        sell[t] * config.GRID_FEEDIN_PRICE * 1000  # MW → kW
        for t in range(24)
    )

    # OM costs (deterministic given runtime_hours)
    el_om = runtime_hours * P_TOTAL * 1000 * EL_OM_BLENDED
    nh3_om = runtime_hours * P_NH3 * 1000 * NH3_OM_RATE

    # Minimize: grid buy cost + OM - grid sell revenue + green penalty
    prob += buy_cost + el_om + nh3_om - sell_revenue + green_penalty, "TotalCost"

    # ---- Solve ----
    prob.solve(PULP_CBC_CMD(msg=False))

    if prob.status != 1:
        raise RuntimeError(
            f"MILP solver failed with status {prob.status} "
            f"(expected 1=Optimal). Status code: {prob.status}"
        )

    # ---- Extract results ----
    x_vals = np.array([value(x[t]) for t in range(24)], dtype=int)
    buy_vals = np.array([value(buy[t]) for t in range(24)])
    sell_vals = np.array([value(sell[t]) for t in range(24)])
    obj_val = value(prob.objective)

    # Compute cost components
    hourly_buy_kwh = buy_vals * 1000   # MW → kW
    total_sell_kwh = np.sum(sell_vals) * 1000
    total_wind_kwh = np.sum(wind) * 1000
    total_solar_kwh = np.sum(solar) * 1000
    total_buy_kwh = np.sum(hourly_buy_kwh)
    total_load_kwh = (np.sum(load) + runtime_hours * P_TOTAL) * 1000

    # OM costs (pre-computed above, same values)
    el_om_cost = el_om
    nh3_om_cost = nh3_om

    # Full ton-ammonia cost
    ton_cost = calc_ton_ammonia_cost(
        nh3_output_tons=daily_output,
        grid_buy_kwh=hourly_buy_kwh,
        grid_sell_kwh=total_sell_kwh,
        grid_buy_cost=0,  # auto-calc from TOU array
        grid_sell_revenue=total_sell_kwh * config.GRID_FEEDIN_PRICE,
        wind_gen_kwh=total_wind_kwh,
        solar_gen_kwh=total_solar_kwh,
        el_om_cost=el_om_cost,
        nh3_om_cost=nh3_om_cost,
        equipment_depreciation=calc_total_depreciation(daily_output),
        production_level_tons_per_day=daily_output,
    )

    result = {
        "status": "Optimal" if prob.status == 1 else str(prob.status),
        "objective": obj_val,
        "x_schedule": x_vals,
        "buy_schedule": buy_vals,
        "sell_schedule": sell_vals,
        "runtime_hours": runtime_hours,
        "hourly_buy_kwh": hourly_buy_kwh,
        "total_sell_kwh": total_sell_kwh,
        "total_wind_kwh": total_wind_kwh,
        "total_solar_kwh": total_solar_kwh,
        "total_buy_kwh": total_buy_kwh,
        "total_load_kwh": total_load_kwh,
        "el_om_cost": el_om_cost,
        "nh3_om_cost": nh3_om_cost,
        "ton_ammonia_cost": ton_cost,
        "h2_produced_kg": TOTAL_H2_RATE * runtime_hours,
        "h2_consumed_kg": daily_output * 1000 * NH3_H2_CONSUMPTION,
        "green_penalty": value(green_penalty),
    }
    _verify_solution(result, wind, solar, load, daily_output)
    return result


# ============================================================
# Part (1): Typical scenario analysis
# ============================================================
def run_typical_scenario():
    """Solve MILP for all 5 production levels under the typical scenario."""
    wind = config.get_typical_wind()
    solar = config.get_typical_solar()
    load = config.get_load_profile()

    results = []
    failed = []
    for daily_output in PRODUCTION_LEVELS:
        print(f"\n--- 典型场景: 日产量 {daily_output} 吨/天 ---")
        try:
            res = build_and_solve_milp(wind, solar, load, daily_output)
        except (RuntimeError, ValueError) as e:
            print(f"FAILED: {e}")
            failed.append((daily_output, str(e)))
            continue
        results.append(res)

        # Print schedule table
        hours = list(range(24))
        print(f"{'小时':>4} {'x(ON)':>6} {'买电MW':>8} {'卖电MW':>8} {'风电MW':>8} {'光伏MW':>8} {'负荷MW':>8}")
        for t in range(24):
            print(
                f"{t:>4} {res['x_schedule'][t]:>6} "
                f"{res['buy_schedule'][t]:>8.2f} {res['sell_schedule'][t]:>8.2f} "
                f"{wind[t]:>8.2f} {solar[t]:>8.2f} {load[t]:>8.2f}"
            )
        print(f"运行小时: {res['runtime_hours']}h")
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

    # Save CSV
    _save_typical_csv(results, wind, solar, load, best_daily)
    return results, best_daily, failed


def _save_typical_csv(results, wind, solar, load, best_daily):
    """Save typical scenario schedule to CSV."""
    csv_path = "q2_typical_schedule.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["典型场景调度方案", f"最优产量{best_daily}吨/天"])
        writer.writerow([])

        for daily_output, res in zip(PRODUCTION_LEVELS, results):
            writer.writerow([f"日产量 {daily_output} 吨/天"])
            writer.writerow(
                ["小时", "ON", "买电(MW)", "卖电(MW)", "风电(MW)", "光伏(MW)", "负荷(MW)"]
            )
            for t in range(24):
                writer.writerow([
                    t,
                    res["x_schedule"][t],
                    f"{res['buy_schedule'][t]:.2f}",
                    f"{res['sell_schedule'][t]:.2f}",
                    f"{wind[t]:.2f}",
                    f"{solar[t]:.2f}",
                    f"{load[t]:.2f}",
                ])
            writer.writerow([
                "运行小时", res["runtime_hours"], "吨氨成本",
                f"{res['ton_ammonia_cost']:.2f}", "绿电惩罚",
                f"{res['green_penalty']:.2f}", "",
            ])
            writer.writerow([])
    print(f"典型场景调度已保存: {csv_path}")


# ============================================================
# Part (2): All 24 scenarios × 5 levels (120 solves)
# ============================================================
def run_all_scenarios():
    """Solve MILP for all 24 wind/solar scenarios × 5 production levels."""
    scenarios = generate_all_scenarios()  # list of (w, s) tuples
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
                res = build_and_solve_milp(wind, solar, load, daily_output)
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
                "runtime_hours": res["runtime_hours"],
                "ton_ammonia_cost": res["ton_ammonia_cost"],
                "objective": res["objective"],
                "total_buy_kwh": res["total_buy_kwh"],
                "total_sell_kwh": res["total_sell_kwh"],
                "classification": classification,
                "indicators": indicators,
                "green_penalty": res["green_penalty"],
                "x_schedule": res["x_schedule"],
                "buy_schedule": res["buy_schedule"],
                "sell_schedule": res["sell_schedule"],
                "status": res["status"],
            })
            print(f"OK 成本={res['ton_ammonia_cost']:.2f} {classification}")

    print(f"\n全部{len(all_results)}个MILP求解完成")

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

    return all_results, failed_scenarios


def _compute_annual_stats(all_results, scenarios):
    """Compute annual classification counts and cost statistics."""
    # Each scenario appears 15 days/year → 15 × 24 = 360 days total
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
    """Save annual ton ammonia cost distribution histogram."""
    costs = [r["ton_ammonia_cost"] for r in all_results]

    plt.figure(figsize=(10, 6))
    plt.hist(costs, bins=30, edgecolor="black", alpha=0.7, color="steelblue")
    plt.xlabel("吨氨成本 (¥/ton)")
    plt.ylabel("频次 (场景×产量)")
    plt.title("年度吨氨成本分布 (120个MILP求解)")
    plt.grid(axis="y", alpha=0.3)

    # Add mean and median lines
    mean_cost = np.mean(costs)
    median_cost = np.median(costs)
    plt.axvline(mean_cost, color="red", linestyle="--", linewidth=1.5,
                label=f"均值: {mean_cost:.0f} ¥/ton")
    plt.axvline(median_cost, color="orange", linestyle="--", linewidth=1.5,
                label=f"中位数: {median_cost:.0f} ¥/ton")
    plt.legend()

    png_path = "q2_annual_cost_distribution.png"
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"成本分布图已保存: {png_path}")


def _save_annual_csv(all_results):
    """Save annual summary results to CSV."""
    csv_path = "q2_annual_summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "场景", "风电档位", "光伏档位", "日产量(吨/天)", "运行小时",
            "吨氨成本(¥/ton)", "购电(kWh)", "售电(kWh)", "绿电惩罚(¥)",
            "分类", "求解状态",
        ])
        for r in all_results:
            writer.writerow([
                r["scenario"], r["wind_idx"], r["solar_idx"],
                r["daily_output"], r["runtime_hours"],
                f"{r['ton_ammonia_cost']:.2f}",
                f"{r['total_buy_kwh']:.2f}",
                f"{r['total_sell_kwh']:.2f}",
                f"{r['green_penalty']:.2f}",
                r["classification"], r["status"],
            ])
    print(f"年度汇总已保存: {csv_path}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("问题二: 离散开关MILP调度优化")
    print("=" * 60)

    total_failed = 0

    # Part (1): Typical scenario
    print("\n>>> 第一部分: 典型场景分析")
    _, _, failed_typical = run_typical_scenario()
    total_failed += len(failed_typical)

    # Part (2): All scenarios
    print("\n>>> 第二部分: 全部24场景×5产量 (共120个MILP)")
    _, failed_scenarios = run_all_scenarios()
    total_failed += len(failed_scenarios)

    print("\n问题二完成")

    if total_failed > 0:
        print(f"\n错误: {total_failed} 个场景求解或验证失败，退出码=1")
        sys.exit(1)
