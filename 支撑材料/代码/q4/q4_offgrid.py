# -*- coding: utf-8 -*-
"""问题四：离网运行分析及储能配置研究"""
import sys; sys.stdout.reconfigure(encoding='utf-8')

from pulp import LpProblem, LpVariable, LpMinimize, LpMaximize, lpSum, value, PULP_CBC_CMD
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
    NH3_BASELINE_CAPACITY, ALKEL_H2_RATE, PEMEL_H2_RATE,
    NH3_H2_CONSUMPTION, NH3_RATE, ALKEL_RATED, PEMEL_RATED, NH3_RATED,
    ALKEL_OM, PEMEL_OM, NH3_OM,
    STORAGE_INVESTMENT, STORAGE_LIFETIME,
    STORAGE_CHARGE_EFF, STORAGE_DISCHARGE_EFF, STORAGE_SELF_DISCHARGE, STORAGE_OM,
)
from utils import (
    calc_green_indicators, classify_scenario,
    calc_ton_ammonia_cost, calc_total_depreciation, generate_all_scenarios,
    WIND_LCOE, SOLAR_LCOE,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# =============================================================================
# Constants — derived from config.py for 72t/d capacity
# =============================================================================
SCALE_72 = 72.0 / NH3_BASELINE_CAPACITY  # 2.0
P_ALKEL = ALKEL_RATED * SCALE_72           # 20 MW
P_PEMEL = PEMEL_RATED * SCALE_72           # 20 MW
P_NH3   = NH3_RATED * SCALE_72             # 1.5 MW
P_TOTAL = P_ALKEL + P_PEMEL + P_NH3        # 41.5 MW
NH3_RATE_72 = NH3_RATE * SCALE_72          # 3.0 ton/h
EL_OM_BLENDED = (ALKEL_OM + PEMEL_OM) / 2  # 0.125 ¥/kWh
NH3_OM_RATE = NH3_OM                        # 0.002 ¥/kWh
EL_RATIO = (P_ALKEL + P_PEMEL) / P_TOTAL    # 40/41.5 ≈ 0.964 (电解占总功率比例)
NH3_RATIO = P_NH3 / P_TOTAL                 # 1.5/41.5 ≈ 0.036 (合成氨占总功率比例)
# Q4 使用聚合变量 p[t] = P_ALK + P_PEM + P_NH3 (简化模型)
# 假设各设备按额定功率比例运行 (40:40:1.5), H2平衡自动满足 (电解产H2率15kg/MW·h匹配NH3耗H2)
# 逐台最小值: ALK=2(10%×20), PEM=2(10%×20), NH3=0.15(10%×1.5) → 合计 4.15 MW
# 注意: 聚合模型无法强制逐台最小, 此值为近似; 电解槽折旧含在OM费率中
P_MIN = 2.0 + 2.0 + 0.15  # 4.15 MW (逐台最小值之和, 统一10%下限)

# Storage efficiency factors
ETA_C = STORAGE_CHARGE_EFF                  # 0.90
ETA_D = STORAGE_DISCHARGE_EFF              # 0.90
SELF_DISCH = 1.0 - STORAGE_SELF_DISCHARGE   # 0.998

# Big-M constant for semi-continuous and grid direction
BIG_M = config.WIND_RATED + config.SOLAR_RATED + 200  # large enough


# =============================================================================
# 第一部分：离网无储能 — 全部24场景
# =============================================================================
def run_part1():
    """计算全部24个场景的离网无储能运行结果。

    对每个 (风电, 光伏) 组合:
      - p[t] = 扣除常规负荷后的可用功率，上限 P_TOTAL，需 ≥ P_MIN
      - curt[t] = 超出 P_TOTAL 的弃电量
      - 计算日合成氨产量和吨氨成本。
    """
    print("=" * 60)
    print("第一部分：离网无储能运行分析（24场景）")
    print("=" * 60)

    all_scenarios = generate_all_scenarios()  # [(w,s), ...] 24 entries
    results = []

    for w, s in all_scenarios:
        wind = config.get_wind_profile(w)
        solar = config.get_solar_profile(s)
        load = config.get_load_profile()

        p = np.zeros(24)
        curt = np.zeros(24)
        load_deficit = np.zeros(24)  # 常规负荷缺口 (MW)

        for t in range(24):
            net = wind[t] + solar[t] - load[t]  # 扣除常规负荷后的净功率
            if net < 0:
                # 风+光不足以覆盖常规负荷 → 缺电 (无储能时不可补救)
                load_deficit[t] = -net
                p[t] = 0.0
                curt[t] = 0.0
            elif net < P_MIN:
                # 常规负荷已满足, 但剩余功率不足启动设备 → 全部弃电
                p[t] = 0.0
                curt[t] = net
            else:
                # 常规负荷已满足, 剩余功率足够 → 设备运行
                p[t] = min(net, P_TOTAL)
                curt[t] = max(net - P_TOTAL, 0.0)

        runtime_hours = sum(1 for x in p if x >= P_MIN)
        daily_prod_tons = sum(p) * NH3_RATE_72 / P_TOTAL
        load_deficit_hours = int(sum(load_deficit > 1e-6))
        total_deficit_mwh = float(sum(load_deficit))

        total_el_kwh = sum(p) * 1000.0
        # 拆分电解与合成氨的运维费 (聚合变量按额定比例分配)
        el_om_cost = total_el_kwh * EL_RATIO * EL_OM_BLENDED
        nh3_om_cost = total_el_kwh * NH3_RATIO * NH3_OM_RATE
        wind_gen_kwh = sum(wind) * 1000.0
        solar_gen_kwh = sum(solar) * 1000.0
        total_curt_kwh = sum(curt) * 1000.0

        equip_dep = calc_total_depreciation(daily_prod_tons, annual_operating_days=360)

        # Off-grid: no grid buy/sell
        ton_cost = calc_ton_ammonia_cost(
            nh3_output_tons=daily_prod_tons,
            grid_buy_kwh=0.0,
            grid_sell_kwh=0.0,
            grid_buy_cost=0.0,
            grid_sell_revenue=0.0,
            wind_gen_kwh=wind_gen_kwh,
            solar_gen_kwh=solar_gen_kwh,
            el_om_cost=el_om_cost,
            nh3_om_cost=nh3_om_cost,
            equipment_depreciation=equip_dep,
            production_level_tons_per_day=72.0
        )

        results.append({
            'wind': w, 'solar': s,
            'scenario_name': f'W{w}S{s}',
            'daily_production_tons': round(daily_prod_tons, 4),
            'runtime_hours': runtime_hours,
            'total_curtailed_mwh': round(float(sum(curt)), 4),
            'curtailment_kwh': round(total_curt_kwh, 2),
            'load_deficit_hours': load_deficit_hours,
            'load_deficit_mwh': round(total_deficit_mwh, 4),
            'ton_cost_yuan': round(ton_cost, 2),
            'wind_gen_mwh': round(float(sum(wind)), 4),
            'solar_gen_mwh': round(float(sum(solar)), 4),
            'load_mwh': round(float(sum(load)), 4),
        })

        deficit_str = f" deficit={total_deficit_mwh:.1f}MWh({load_deficit_hours}h)" if load_deficit_hours > 0 else ""
        print(f"  W{w}S{s}: prod={daily_prod_tons:5.1f}t/d  "
              f"runtime={runtime_hours:2d}h  curt={sum(curt):6.1f}MWh  "
              f"cost={ton_cost:8.1f}¥/t{deficit_str}")

    # ---- 保存 CSV ----
    os.makedirs('results', exist_ok=True)
    csv_path = os.path.join(SCRIPT_DIR, 'q4_offgrid_no_storage.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  结果已保存: {csv_path}")

    # ---- 最大弃电场景 ----
    max_scen = max(results, key=lambda r: r['curtailment_kwh'])
    print(f"\n  最大弃电场景: W{max_scen['wind']}S{max_scen['solar']}, "
          f"弃电量={max_scen['curtailment_kwh']:.0f} kWh")

    # ---- 最小风光装机容量估算 ----
    # 固定设备容量（72t/d不变），缩放风光装机，寻找支撑≥36t/d的最小风光容量系数α
    WIND_RATED = config.WIND_RATED   # 40 MW
    SOLAR_RATED = config.SOLAR_RATED  # 64 MW
    print(f"\n  --- 最小风光装机容量估算 (固定设备{P_TOTAL:.1f}MW，系数 α 从 1.0 双向搜索，步长 0.05) ---")
    min_prod_target = 36.0
    found_alpha = None

    def _eval_alpha(alpha_val):
        """给定α，返回24场景最差日产量和最差场景编号。"""
        min_daily = float('inf')
        worst_ws = (0, 0)
        for w, s in all_scenarios:
            wind = config.get_wind_profile(w) * alpha_val
            solar = config.get_solar_profile(s) * alpha_val
            load = config.get_load_profile()
            p = np.zeros(24)
            for t in range(24):
                net = wind[t] + solar[t] - load[t]
                if net < 0 or net < P_MIN:
                    p[t] = 0.0
                else:
                    p[t] = min(net, P_TOTAL)
            daily_p = sum(p) * NH3_RATE_72 / P_TOTAL
            if daily_p < min_daily:
                min_daily = daily_p
                worst_ws = (w, s)
        return min_daily, worst_ws

    # 先判断α=1.0时是否能满足 → 若不能，则α_min > 1.0
    base_min, base_ws = _eval_alpha(1.0)
    print(f"    α=1.00  风电={WIND_RATED:5.1f}MW  光伏={SOLAR_RATED:5.1f}MW  "
          f"min_prod={base_min:5.1f}t/d  (W{base_ws[0]}S{base_ws[1]})")

    if base_min >= min_prod_target:
        # α=1.0已满足 → 向下搜索缩小风光
        print("    α=1.0已满足目标，向下搜索最小风光容量...")
        for alpha_try in np.arange(0.95, 0.05, -0.05):
            alpha = round(alpha_try, 2)
            min_daily, worst_ws = _eval_alpha(alpha)
            print(f"    α={alpha:.2f}  风电={alpha*WIND_RATED:5.1f}MW  光伏={alpha*SOLAR_RATED:5.1f}MW  "
                  f"min_prod={min_daily:5.1f}t/d  (W{worst_ws[0]}S{worst_ws[1]})")
            if min_daily < min_prod_target:
                # 上一步α满足，当前α不满足 → found_alpha是上一步的值
                found_alpha = round(alpha + 0.05, 2)
                break
        else:
            found_alpha = 0.10
    else:
        # α=1.0不满足 → 向上搜索增大风光
        print("    α=1.0不满足目标，向上搜索最小风光容量...")
        for alpha_try in np.arange(1.05, 20.01, 0.05):
            alpha = round(alpha_try, 2)
            min_daily, worst_ws = _eval_alpha(alpha)
            if alpha <= 3.0 or abs(alpha % 1.0) < 0.01:
                print(f"    α={alpha:.2f}  风电={alpha*WIND_RATED:5.1f}MW  光伏={alpha*SOLAR_RATED:5.1f}MW  "
                      f"min_prod={min_daily:5.1f}t/d  (W{worst_ws[0]}S{worst_ws[1]})")
            if min_daily >= min_prod_target:
                found_alpha = alpha
                break

    if found_alpha is not None:
        print(f"\n    *** 最小风光容量系数 α_min = {found_alpha:.2f} ***")
        print(f"        风电额定: {found_alpha*WIND_RATED:.1f} MW ({found_alpha:.1f}× 原始40MW)")
        print(f"        光伏额定: {found_alpha*SOLAR_RATED:.1f} MW ({found_alpha:.1f}× 原始64MW)")
        print(f"        风光合计: {found_alpha*(WIND_RATED+SOLAR_RATED):.1f} MW")
        print(f"        最差场景 W{base_ws[0]}S{base_ws[1]} 日产量 = {_eval_alpha(found_alpha)[0]:.1f} t/d >= {min_prod_target} t/d")
    else:
        print(f"    警告: α 搜索至 20.0 仍未满足 {min_prod_target}t/d 目标")
        found_alpha = None

    return results, max_scen, found_alpha


# =============================================================================
# 第二部分：储能优化（先针对最大弃电场景，再推广到全部24场景）
# =============================================================================
def _build_storage_lp(wind_arr, solar_arr, load_arr, prod_target,
                      fixed_capacity=None, fixed_power=None,
                      label="", maximize_production=False):
    """构建储能优化 LP 模型。

    参数
    ----------
    wind_arr, solar_arr, load_arr : np.array (24,)
    prod_target : float — 日合成氨最低目标产量 (吨/天)
    fixed_capacity : float 或 None — 若给定则储能容量固定
    fixed_power : float 或 None — 若给定则储能功率固定
    label : str — 问题标识
    maximize_production : bool — 若为 True，最大化产量而非最小化成本

    返回
    -------
    (结果字典, LpProblem)
    """
    if maximize_production:
        prob = LpProblem(f"Q4_MaxProd_{label}", LpMaximize)
    else:
        prob = LpProblem(f"Q4_Storage_{label}", LpMinimize)

    # 决策变量 — 24时段
    p = [LpVariable(f"p_{t}", lowBound=0) for t in range(24)]
    Pc = [LpVariable(f"Pc_{t}", lowBound=0) for t in range(24)]
    Pd = [LpVariable(f"Pd_{t}", lowBound=0) for t in range(24)]
    curt = [LpVariable(f"curt_{t}", lowBound=0) for t in range(24)]  # 弃电变量
    z = [LpVariable(f"z_{t}", cat="Binary") for t in range(24)]

    # 储能标量变量
    if fixed_capacity is not None:
        capacity = fixed_capacity
    else:
        capacity = LpVariable("capacity", lowBound=0)

    if fixed_power is not None:
        spower = fixed_power
    else:
        spower = LpVariable("spower", lowBound=0)

    # SOC 变量: SOC[0] .. SOC[24]（共25个）
    soc = [LpVariable(f"soc_{t}", lowBound=0) for t in range(25)]

    SOC_MIN = 0.10
    SOC_MAX = 0.90

    # ---- 半连续约束: p[t] ∈ {0} ∪ [P_MIN, P_TOTAL] ----
    for t in range(24):
        prob += p[t] >= P_MIN * z[t], f"Pmin_{t}"
        prob += p[t] <= P_TOTAL * z[t], f"Pmax_{t}"

    # ---- SOC 初始值与终值 ----
    if fixed_capacity is not None:
        prob += soc[0] == 0.5 * capacity, "SOC_Init"
        prob += soc[24] == soc[0], "SOC_Term"
    else:
        prob += soc[0] == 0.5 * capacity, "SOC_Init"
        prob += soc[24] == soc[0], "SOC_Term"

    # ---- SOC 上下限 ----
    for t in range(24):
        prob += soc[t] >= SOC_MIN * capacity, f"SOC_lo_{t}"
        prob += soc[t] <= SOC_MAX * capacity, f"SOC_up_{t}"

    # ---- SOC 动态方程 ----
    INV_ETA_D = 1.0 / ETA_D  # ≈ 1.111...
    for t in range(24):
        prob += (soc[t + 1] == soc[t] * SELF_DISCH + ETA_C * Pc[t] - INV_ETA_D * Pd[t],
                 f"SOC_dyn_{t}")

    # ---- 充放电 ≤ 储能功率 ----
    for t in range(24):
        prob += Pc[t] <= spower, f"Pc_bound_{t}"
        prob += Pd[t] <= spower, f"Pd_bound_{t}"

    # ---- 功率平衡 (含弃电) ----
    for t in range(24):
        prob += (wind_arr[t] + solar_arr[t] + Pd[t]
                 == Pc[t] + p[t] + load_arr[t] + curt[t], f"Bal_{t}")

    # ---- 产量目标 ----
    if maximize_production:
        # 目标: 最大化日产量
        prod_expr = lpSum(p) * NH3_RATE_72 / P_TOTAL
        prob += prod_expr
    else:
        # 约束: 产量 ≥ 目标值
        prob += lpSum(p) * NH3_RATE_72 / P_TOTAL >= prod_target, "ProdTarget"

        # 约束: 储能功率 ≤ 容量 × C倍率 (最大4C放电)
        if fixed_capacity is None or fixed_power is None:
            prob += spower <= capacity * 4.0, "PowerCapRatio"

        # 目标: 最小化储能日折旧 + 微小功率惩罚
        daily_dep = capacity * STORAGE_INVESTMENT * 1000 / STORAGE_LIFETIME / 365
        prob += daily_dep + 0.01 * spower, "MinStorageDep"

    # ---- 求解 ----
    solver = PULP_CBC_CMD(msg=False, timeLimit=600, gapRel=0.01)
    prob.solve(solver)

    p_vals = [value(p[t]) for t in range(24)]
    pc_vals = [value(Pc[t]) for t in range(24)]
    pd_vals = [value(Pd[t]) for t in range(24)]
    curt_vals = [value(curt[t]) for t in range(24)]
    z_vals = [int(round(value(z[t]))) for t in range(24)]
    soc_vals = [value(soc[t]) for t in range(25)]

    actual_prod = sum(p_vals) * NH3_RATE_72 / P_TOTAL
    cap_val = fixed_capacity if fixed_capacity is not None else value(capacity)
    pow_val = fixed_power if fixed_power is not None else value(spower)

    result = {
        'status': 'Optimal' if prob.status == 1 else f'Status_{prob.status}',
        'storage_capacity_mwh': cap_val,
        'storage_power_mw': pow_val,
        'p_schedule': p_vals,
        'charge_schedule': pc_vals,
        'discharge_schedule': pd_vals,
        'curtailment_schedule': curt_vals,
        'z_schedule': z_vals,
        'soc_schedule': soc_vals,
        'daily_production': actual_prod,
        'objective': value(prob.objective),
        'runtime_hours': sum(z_vals),
    }

    return result, prob


def run_part2(max_wind_id, max_solar_id, no_storage_results):
    """第二部分：储能优化。

    1. 针对最大弃电场景优化储能配置。
    2. 将最优配置推广到全部24场景。
    """
    print("\n" + "=" * 60)
    print("第二部分：储能优化配置")
    print("=" * 60)

    wind = config.get_wind_profile(max_wind_id)
    solar = config.get_solar_profile(max_solar_id)
    load = config.get_load_profile()

    # 该场景无储能时的产量作为基准目标
    target_prod = 36.0
    for r in no_storage_results:
        if r['wind'] == max_wind_id and r['solar'] == max_solar_id:
            target_prod = r['daily_production_tons']
            break

    print(f"  无储能时产量: {target_prod:.1f} t/d")

    # ---- 步骤1: 优化储能容量 ----
    # 目标满产(72t/d)，储能可回收弃电
    # 若不可行则按2t/d递减
    opt_target = 72.0
    result, prob = _build_storage_lp(wind, solar, load, opt_target,
                                     label=f"W{max_wind_id}S{max_solar_id}")

    while result['status'] != 'Optimal' and opt_target > 10.0:
        opt_target -= 2.0
        print(f"  产量目标 {opt_target + 2.0:.0f} t/d 不可行, "
              f"尝试 {opt_target:.0f} t/d...")
        result, prob = _build_storage_lp(wind, solar, load, opt_target,
                                         label=f"W{max_wind_id}S{max_solar_id}_t{int(opt_target)}")

    if result['status'] != 'Optimal':
        print("  错误: 无法找到可行的储能配置!")
        raise RuntimeError("储能优化不可行")

    cap_opt = result['storage_capacity_mwh']  # MWh
    pow_opt = result['storage_power_mw']       # MW
    prod_opt = result['daily_production']

    print(f"\n  最优储能配置:")
    print(f"    容量: {cap_opt:.2f} MWh")
    print(f"    功率: {pow_opt:.2f} MW")
    print(f"    日产量: {prod_opt:.2f} t/d")
    print(f"    日折旧: {result['objective']:.2f} 元/天")

    # 保存储能配置
    cfg = {
        'scenario': f'W{max_wind_id}S{max_solar_id}',
        'storage_capacity_mwh': round(cap_opt, 4),
        'storage_power_mw': round(pow_opt, 4),
        'daily_depreciation_yuan': round(result['objective'], 2),
        'production_target_tpd': round(opt_target, 1),
        'actual_production_tpd': round(prod_opt, 4),
    }
    csv_path = os.path.join(SCRIPT_DIR, 'q4_storage_config.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(cfg.keys()))
        writer.writeheader()
        writer.writerow(cfg)
    print(f"  配置已保存: {csv_path}")

    # ---- 步骤2: 推广到全部24场景 ----
    print("\n  --- 固定储能配置，推广至全部24场景 ---")
    all_scenarios = generate_all_scenarios()
    storage_results = []

    for w, s in all_scenarios:
        wind_s = config.get_wind_profile(w)
        solar_s = config.get_solar_profile(s)
        load_s = config.get_load_profile()

        # Find the no-storage production for this scenario as a target
        ns_prod = 36.0
        for r in no_storage_results:
            if r['wind'] == w and r['solar'] == s:
                ns_prod = r['daily_production_tons']
                break

        # 用该场景无储能产量作为目标，若不可行则递减
        r, _ = _build_storage_lp(wind_s, solar_s, load_s, ns_prod,
                                 fixed_capacity=cap_opt, fixed_power=pow_opt,
                                 label=f"W{w}S{s}")

        relax_r = ns_prod
        while r['status'] != 'Optimal' and relax_r > 10.0:
            relax_r -= 2.0
            r, _ = _build_storage_lp(wind_s, solar_s, load_s, relax_r,
                                     fixed_capacity=cap_opt, fixed_power=pow_opt,
                                     label=f"W{w}S{s}_r{int(relax_r)}")

        daily_prod = r['daily_production']
        total_el_kwh = sum(r['p_schedule']) * 1000.0
        el_om = total_el_kwh * EL_RATIO * EL_OM_BLENDED
        nh3_om = total_el_kwh * NH3_RATIO * NH3_OM_RATE
        wind_kwh = sum(wind_s) * 1000.0
        solar_kwh = sum(solar_s) * 1000.0
        equip_dep = calc_total_depreciation(daily_prod, annual_operating_days=360)

        # Storage daily depreciation (fixed, from optimal config)
        storage_daily_dep = cap_opt * STORAGE_INVESTMENT * 1000 / STORAGE_LIFETIME / 365

        # Ton cost including storage depreciation
        ton_cost = calc_ton_ammonia_cost(
            nh3_output_tons=daily_prod,
            grid_buy_kwh=0.0,
            grid_sell_kwh=0.0,
            grid_buy_cost=0.0,
            grid_sell_revenue=0.0,
            wind_gen_kwh=wind_kwh,
            solar_gen_kwh=solar_kwh,
            el_om_cost=el_om,
            nh3_om_cost=nh3_om,
            equipment_depreciation=equip_dep + storage_daily_dep,  # include storage dep
            production_level_tons_per_day=72.0,
        )

        curt_mwh = sum(r['curtailment_schedule'])  # MW·h from LP variable

        # Off-grid green indicators (no grid buy/sell)
        total_gen_kwh = wind_kwh + solar_kwh
        total_load_kwh = total_el_kwh + sum(load_s) * 1000.0

        curtailment_mwh = curt_mwh

        green_inds = calc_green_indicators(
            total_gen=total_gen_kwh,
            total_load=total_load_kwh,
            grid_buy=0.0,
            grid_sell=0.0,
        )

        storage_results.append({
            'scenario': f'W{w}S{s}',
            'wind_id': w, 'solar_id': s,
            'daily_production_tons': round(daily_prod, 4),
            'runtime_hours': int(r['runtime_hours']),
            'ton_cost_yuan': round(ton_cost, 2),
            'storage_capacity_mwh': round(cap_opt, 2),
            'storage_power_mw': round(pow_opt, 2),
            'storage_daily_dep_yuan': round(storage_daily_dep, 2),
            'curtailment_mwh': round(curtailment_mwh, 4),
            'status': r['status'],
        })
        print(f"    W{w}S{s}: prod={daily_prod:5.1f}t/d  cost={ton_cost:8.1f}¥/t  "
              f"curt={curtailment_mwh:6.1f}MWh  {r['status']}")

    # Save all-scenario results
    csv_path2 = os.path.join(SCRIPT_DIR, 'q4_with_storage.csv')
    with open(csv_path2, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(storage_results[0].keys()))
        writer.writeheader()
        writer.writerows(storage_results)
    print(f"\n  全部场景结果已保存: {csv_path2}")

    return cap_opt, pow_opt, storage_results


# =============================================================================
# 第三部分：经济性对比（离网+储能 vs 联网）
# =============================================================================
def run_part3(no_storage_results, storage_results, cap_opt):
    """离网+储能 vs 联网运行的年度经济性对比。

    联网数据读取 q3_annual_summary.csv（若不存在则回退到 q2_annual_summary.csv）。
    """
    print("\n" + "=" * 60)
    print("第三部分：经济性对比分析")
    print("=" * 60)

    # ---- 离网+储能：年度经济指标 ----
    storage_daily_dep = cap_opt * STORAGE_INVESTMENT * 1000 / STORAGE_LIFETIME / 365
    DAYS_PER_SCENARIO = 15

    # 年度总成本 = 24场景 × 15天/场景 求和
    total_annual_cost_offgrid = 0.0
    for r in storage_results:
        total_annual_cost_offgrid += r['ton_cost_yuan'] * r['daily_production_tons'] * DAYS_PER_SCENARIO

    total_annual_prod_offgrid = sum(r['daily_production_tons'] * DAYS_PER_SCENARIO for r in storage_results)
    avg_ton_cost_offgrid = total_annual_cost_offgrid / total_annual_prod_offgrid if total_annual_prod_offgrid > 0 else float('inf')

    # ---- 联网模式：从Q3取每场景最优产量(最低吨氨成本) ----
    grid_annual_prod = 0.0
    grid_annual_cost = 0.0
    grid_count = 0

    try:
        import pandas as pd
        q3_csv = os.path.join(os.path.dirname(__file__), '..', 'q3', 'q3_annual_summary.csv')
        df_q3 = pd.read_csv(q3_csv)
        # 每场景取吨氨成本最低的产量级别
        best_per_scenario = df_q3.loc[df_q3.groupby('场景')['吨氨成本(元/吨)'].idxmin()]
        for _, row in best_per_scenario.iterrows():
            prod = float(row['日产量(吨/天)'])
            cost = float(row['吨氨成本(元/吨)'])
            grid_annual_prod += prod * DAYS_PER_SCENARIO
            grid_annual_cost += cost * prod * DAYS_PER_SCENARIO
            grid_count += 1
        print(f"  从 Q3 读取了 {grid_count} 个场景的最优联网方案")
    except Exception as e:
        print(f"  无法读取 Q3 数据: {e}")

    avg_ton_cost_grid = grid_annual_cost / grid_annual_prod if grid_annual_prod > 0 else float('inf')

    # ---- 保存对比结果 ----
    comparison = [
        {
            'mode': '离网+储能',
            'annual_production_tons': round(total_annual_prod_offgrid, 1),
            'annual_total_cost_yuan': round(total_annual_cost_offgrid, 0),
            'average_ton_cost_yuan': round(avg_ton_cost_offgrid, 2),
            'storage_capacity_mwh': round(cap_opt, 2),
            'storage_daily_dep_yuan': round(storage_daily_dep, 2),
        },
        {
            'mode': '并网(无储能)',
            'annual_production_tons': round(grid_annual_prod, 1),
            'annual_total_cost_yuan': round(grid_annual_cost, 0),
            'average_ton_cost_yuan': round(avg_ton_cost_grid, 2),
            'storage_capacity_mwh': 0,
            'storage_daily_dep_yuan': 0,
        },
    ]

    csv_path = os.path.join(SCRIPT_DIR, 'q4_economic_comparison.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(comparison[0].keys()))
        writer.writeheader()
        writer.writerows(comparison)
    print(f"\n  对比结果已保存: {csv_path}")

    # 打印摘要
    print(f"\n  --- 经济性对比 ---")
    print(f"  {'模式':<20} {'年产量(吨)':>12} {'年总成本(元)':>15} {'吨氨成本(元/吨)':>15}")
    for c in comparison:
        print(f"  {c['mode']:<20} {c['annual_production_tons']:>12.1f} "
              f"{c['annual_total_cost_yuan']:>15.0f} {c['average_ton_cost_yuan']:>15.2f}")

    if avg_ton_cost_grid > 0:
        delta = avg_ton_cost_offgrid - avg_ton_cost_grid
        pct = delta / avg_ton_cost_grid * 100
        print(f"\n  离网+储能 vs 并网: {'+'if delta>0 else ''}{delta:.2f} 元/吨 "
              f"({'↑'if delta>0 else '↓'}{abs(pct):.1f}%)")

    return comparison


# =============================================================================
# 主程序
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("电工杯A题 问题四：离网运行分析及储能配置研究")
    print("=" * 60)

    # 第一部分: 离网无储能
    results_1, max_scen, alpha_min = run_part1()

    # 第二部分: 储能优化
    cap_opt, pow_opt, storage_results = run_part2(
        max_scen['wind'], max_scen['solar'], results_1
    )

    # 第三部分: 经济性对比
    comparison = run_part3(results_1, storage_results, cap_opt)

    print("\n" + "=" * 60)
    print("问题四完成")
    print("=" * 60)
    print(f"  第一部分结果: q4/q4_offgrid_no_storage.csv")
    print(f"  第二部分结果: q4/q4_storage_config.csv")
    print(f"  第二部分全部: q4/q4_with_storage.csv")
    print(f"  第三部分结果: q4/q4_economic_comparison.csv")
    if alpha_min is not None:
        print(f"  最小风光容量系数: α_min = {alpha_min:.2f}")
        print(f"    风电: {alpha_min*config.WIND_RATED:.1f}MW, 光伏: {alpha_min*config.SOLAR_RATED:.1f}MW")
    else:
        print(f"  最小风光容量系数: 未找到 (α ≤ 20.0 仍不满足)")
