# -*- coding: utf-8 -*-
"""
q3_sensitivity.py — Q3 OAT参数灵敏度分析 (MILP, 72t/d, 分拆ALK/PEM/NH3)

对Q3关键的8个参数进行单因子扰动 (One-At-a-Time):
  NH3_LOAD_MIN, NH3_RAMP_RATE, green_penalty_λ₁/λ₂/λ₃,
  ALK_H2_PER_MW, PEM_H2_PER_MW, tou_price_scale

注意: q3_lp.py 通过 `from config import X` 导入参数, 这些参数在 q3_lp 模块中
      成为局部变量。 因此 monkey-patch 必须同时覆盖 config 模块和 q3_lp 模块。
"""

import sys
import os
import csv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, ROOT_DIR)

import numpy as np
import config as root_config
import q3.q3_lp as q3_lp_mod
from sensitivity.config import TYPICAL_SCENARIOS

# ============================================================================
# 数据输出路径
# ============================================================================
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(DATA_DIR, 'q3_oat_results.csv')

# ============================================================================
# 参数名映射 (sensitivity config 名 → root_config 属性名)
# ============================================================================
NAME_TO_CONFIG_ATTR = {
    'green_penalty_self_use':  'GREEN_PENALTY_SELF_USE',
    'green_penalty_green_rate': 'GREEN_PENALTY_GREEN_RATE',
    'green_penalty_feed_rate':  'GREEN_PENALTY_FEED_RATE',
    'nh3_load_min':             'NH3_LOAD_MIN',
    'nh3_ramp_rate':            'NH3_RAMP_RATE',
    'alk_h2_per_mw':            'ALK_H2_PER_MW',
    'pem_h2_per_mw':            'PEM_H2_PER_MW',
    # tou_price_scale: 无直接对应 → 通过 monkey-patch get_price 实现
}

# 需要在 q3_lp 模块层面覆盖的参数 (被 from config import 过)
Q3_LP_MODULE_PARAMS = {
    'nh3_load_min', 'nh3_ramp_rate', 'alk_h2_per_mw', 'pem_h2_per_mw',
}

# 直接在 config 模块层面覆盖的参数 (被 add_green_penalty_to_problem 读取)
CONFIG_MODULE_PARAMS = {
    'green_penalty_self_use', 'green_penalty_green_rate', 'green_penalty_feed_rate',
}

# ============================================================================
# Q3 关键参数定义
# ============================================================================
Q3_CRITICAL_PARAMS = [
    {
        'name': 'nh3_load_min',
        'label': 'NH₃最低负荷率',
        'range': [0.05, 0.50],
        'unit': 'ratio',
        'n_points': 7,
    },
    {
        'name': 'nh3_ramp_rate',
        'label': 'NH₃爬坡率',
        'range': [0.05, 0.50],
        'unit': 'ratio',
        'n_points': 7,
    },
    {
        'name': 'green_penalty_self_use',
        'label': '自发自用率惩罚λ₁',
        'range': [0.1, 2.0],
        'unit': '¥/kWh',
        'n_points': 7,
    },
    {
        'name': 'green_penalty_green_rate',
        'label': '绿电比例惩罚λ₂',
        'range': [0.05, 1.5],
        'unit': '¥/kWh',
        'n_points': 7,
    },
    {
        'name': 'green_penalty_feed_rate',
        'label': '上网比例惩罚λ₃',
        'range': [0.1, 2.5],
        'unit': '¥/kWh',
        'n_points': 7,
    },
    {
        'name': 'alk_h2_per_mw',
        'label': 'ALK制氢效率',
        'range': [8.0, 20.0],
        'unit': 'kg/(h·MW)',
        'n_points': 7,
    },
    {
        'name': 'pem_h2_per_mw',
        'label': 'PEM制氢效率',
        'range': [10.0, 22.0],
        'unit': 'kg/(h·MW)',
        'n_points': 7,
    },
    {
        'name': 'tou_price_scale',
        'label': '分时电价系数',
        'range': [0.3, 1.5],
        'unit': 'scale',
        'n_points': 7,
    },
]


def generate_oat_points(param_def):
    """为给定参数生成 OAT 扰动点列表 (含基线).

    从 [range_low, range_high] 均匀采样 n_points 个点,
    包含基线值 (如果基线在范围内).
    """
    lo, hi = param_def['range']
    n = param_def['n_points']
    name = param_def['name']

    # 获取基线值
    if name == 'tou_price_scale':
        baseline = 1.0
    else:
        attr = NAME_TO_CONFIG_ATTR[name]
        baseline = getattr(root_config, attr)

    points = list(np.linspace(lo, hi, n))
    # 确保基线在列表中
    if lo <= baseline <= hi:
        rounded_baseline = round(baseline, 6)
        if not any(abs(p - baseline) < 1e-8 for p in points):
            points.append(baseline)
            points.sort()
    return points


def get_param_baseline(param_name):
    """获取参数的当前基线值."""
    if param_name == 'tou_price_scale':
        return 1.0
    attr = NAME_TO_CONFIG_ATTR[param_name]
    return getattr(root_config, attr)


def run_q3_with_override(wind, solar, load, daily_output, param_name, param_value):
    """运行 Q3 MILP, 覆盖单个参数.

    返回 dict: {'status': ..., 'ton_ammonia_cost': ..., ...} 或包含 error 字段.
    自动恢复原始参数值, 防止状态泄漏.

    猴子补丁策略:
      - q3_lp 模块级参数 (from config import X):  覆盖 q3_lp_mod.X
      - config 模块级参数 (GREEN_PENALTY_*):      覆盖 root_config.X
      - tou_price_scale:                          覆盖 root_config.get_price
    """
    # --- 保存原始状态 ---
    saved = {}

    if param_name == 'tou_price_scale':
        saved['get_price'] = root_config.get_price
    elif param_name in Q3_LP_MODULE_PARAMS:
        attr = NAME_TO_CONFIG_ATTR[param_name]
        saved['q3_lp_attr'] = getattr(q3_lp_mod, attr)
    elif param_name in CONFIG_MODULE_PARAMS:
        attr = NAME_TO_CONFIG_ATTR[param_name]
        saved['config_attr'] = getattr(root_config, attr)

    try:
        # --- 应用覆盖 ---
        if param_name == 'tou_price_scale':
            original_get_price = root_config.get_price
            root_config.get_price = lambda h, s=param_value: original_get_price(h) * s
        elif param_name in Q3_LP_MODULE_PARAMS:
            attr = NAME_TO_CONFIG_ATTR[param_name]
            setattr(q3_lp_mod, attr, param_value)
            # 同时覆盖 root_config 以保持一致
            setattr(root_config, attr, param_value)
        elif param_name in CONFIG_MODULE_PARAMS:
            attr = NAME_TO_CONFIG_ATTR[param_name]
            setattr(root_config, attr, param_value)

        # --- 求解 ---
        result = q3_lp_mod.build_and_solve_lp(wind, solar, load, daily_output)

        return {
            'status': result.get('status', 'unknown'),
            'objective': result.get('objective'),
            'ton_ammonia_cost': result.get('ton_ammonia_cost'),
            'total_p_mwh': result.get('total_p_mwh'),
            'total_alk_mwh': result.get('total_alk_mwh'),
            'total_pem_mwh': result.get('total_pem_mwh'),
            'total_nh3_mwh': result.get('total_nh3_mwh'),
            'total_buy_kwh': result.get('total_buy_kwh'),
            'total_sell_kwh': result.get('total_sell_kwh'),
            'green_penalty': result.get('green_penalty'),
            'h2_produced_kg': result.get('h2_produced_kg'),
            'h2_consumed_kg': result.get('h2_consumed_kg'),
        }

    except (RuntimeError, ValueError) as e:
        return {
            'status': 'infeasible',
            'error': str(e)[:200],
            'ton_ammonia_cost': None,
            'objective': None,
        }

    finally:
        # --- 恢复原始值 ---
        if param_name == 'tou_price_scale':
            root_config.get_price = saved['get_price']
        elif param_name in Q3_LP_MODULE_PARAMS:
            attr = NAME_TO_CONFIG_ATTR[param_name]
            setattr(q3_lp_mod, attr, saved['q3_lp_attr'])
            setattr(root_config, attr, saved['q3_lp_attr'])
        elif param_name in CONFIG_MODULE_PARAMS:
            attr = NAME_TO_CONFIG_ATTR[param_name]
            setattr(root_config, attr, saved['config_attr'])


def run_parameter_scan(wind, solar, load, scenario_name, param_def):
    """对单个参数执行完整 OAT 扫描, 返回结果列表."""
    results = []
    points = generate_oat_points(param_def)
    baseline = get_param_baseline(param_def['name'])

    for val in points:
        is_baseline = abs(val - baseline) < 1e-8
        print(f"  [{param_def['label']}] = {val:.4f}{' (baseline)' if is_baseline else ''} ...", end=' ', flush=True)

        try:
            res = run_q3_with_override(wind, solar, load, 72, param_def['name'], val)
        except Exception as e:
            print(f"CRASH: {e}")
            res = {'status': 'error', 'error': str(e)[:200], 'ton_ammonia_cost': None}

        row = {
            'scenario': scenario_name,
            'param_name': param_def['name'],
            'param_label': param_def['label'],
            'param_value': val,
            'param_unit': param_def['unit'],
            'is_baseline': is_baseline,
            'status': res.get('status', 'unknown'),
            'ton_ammonia_cost': res.get('ton_ammonia_cost'),
            'objective': res.get('objective'),
            'total_p_mwh': res.get('total_p_mwh'),
            'total_alk_mwh': res.get('total_alk_mwh'),
            'total_pem_mwh': res.get('total_pem_mwh'),
            'total_nh3_mwh': res.get('total_nh3_mwh'),
            'total_buy_kwh': res.get('total_buy_kwh'),
            'total_sell_kwh': res.get('total_sell_kwh'),
            'green_penalty': res.get('green_penalty'),
            'h2_produced_kg': res.get('h2_produced_kg'),
            'error': res.get('error', ''),
        }
        results.append(row)

        if res.get('status') == 'infeasible':
            print(f"INFEASIBLE ({res.get('error', '')[:60]})")
        elif res.get('status') == 'error':
            print(f"ERROR")
        else:
            cost = res.get('ton_ammonia_cost')
            cost_str = f"{cost:.2f}" if cost is not None else 'N/A'
            print(f"cost={cost_str} ¥/t")

    return results


def save_oat_results(results, filepath):
    """将 OAT 扫描结果保存为 CSV."""
    if not results:
        print("  (无结果, 跳过保存)")
        return

    fieldnames = [
        'scenario', 'param_name', 'param_label', 'param_value', 'param_unit',
        'is_baseline', 'status', 'ton_ammonia_cost', 'objective',
        'total_p_mwh', 'total_alk_mwh', 'total_pem_mwh', 'total_nh3_mwh',
        'total_buy_kwh', 'total_sell_kwh', 'green_penalty',
        'h2_produced_kg', 'h2_consumed_kg', 'error',
    ]

    with open(filepath, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    print(f"  结果已保存至: {filepath} ({len(results)} 行)")


def q3_sensitivity(scenarios=None, params=None):
    """Q3 OAT 灵敏度分析主入口.

    Args:
        scenarios: 场景列表 [(w_index, s_index), ...], 默认 TYPICAL_SCENARIOS 全部8个
        params:    参数列表, 默认 Q3_CRITICAL_PARAMS 全部8个
    """
    if scenarios is None:
        scenarios = TYPICAL_SCENARIOS
    if params is None:
        params = Q3_CRITICAL_PARAMS

    all_results = []
    total_runs = len(scenarios) * sum(p['n_points'] for p in params)
    print(f"Q3 OAT 灵敏度分析开始: {len(scenarios)} 场景 × {len(params)} 参数")
    print(f"  预计总运行次数: ~{total_runs}")
    print(f"  固定日产量: 72 吨/天\n")

    load = root_config.get_load_profile()

    for w_idx, s_idx in scenarios:
        wind = root_config.get_wind_profile(w_idx)
        solar = root_config.get_solar_profile(s_idx)
        scenario_name = f"W{w_idx}S{s_idx}"

        print(f"=== 场景 {scenario_name} (风电场景{w_idx}, 光伏场景{s_idx}) ===")

        for param_def in params:
            print(f"  参数: {param_def['label']} ({param_def['name']}), "
                  f"范围 [{param_def['range'][0]}, {param_def['range'][1]}]")

            results = run_parameter_scan(wind, solar, load, scenario_name, param_def)
            all_results.extend(results)

            feasible = sum(1 for r in results if r['status'] == 'Optimal')
            infeasible = sum(1 for r in results if r['status'] == 'infeasible')
            print(f"    → {len(results)} 点: {feasible} 可行, {infeasible} 不可行\n")

        print()

    # 汇总
    total_feasible = sum(1 for r in all_results if r['status'] == 'Optimal')
    total_infeasible = sum(1 for r in all_results if r['status'] == 'infeasible')
    print(f"=== 汇总 ===")
    print(f"  总运行: {len(all_results)}")
    print(f"  可行:   {total_feasible}")
    print(f"  不可行: {total_infeasible}")

    save_oat_results(all_results, OUTPUT_CSV)

    return all_results


# ============================================================================
# 主入口
# ============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("Q3 OAT 参数灵敏度分析 (MILP, 72t/d, ALK/PEM/NH3 分拆)")
    print("=" * 60)

    # 3个对比场景 (最低、中等、最高风光)
    print("\n--- 3个对比场景 (W0S0, W4S2, W5S3) ---")
    q3_sensitivity(scenarios=[(0,0), (4,2), (5,3)])

    print("\n完成.")
