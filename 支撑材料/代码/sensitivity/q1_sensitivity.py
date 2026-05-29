# -*- coding: utf-8 -*-
"""
q1_sensitivity.py — 电工杯A题 问题1 OAT参数敏感性分析

对 Q1 基准全负荷连续运行场景 (36t/d) 进行单参数扰动 (One-At-a-Time) 分析，
量化各参数对吨氨成本和绿电指标的影响程度。

测试参数: wind_lcoe, solar_lcoe, tou_price_scale, alkel_om, pemel_om,
          alkel_rated_scale, pemel_rated_scale
"""

import sys
import os

# 修复 Windows 终端中文输出乱码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 确保能导入根目录模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np

import config as root_config
from utils import calc_green_indicators, check_green_requirements, \
    classify_scenario, calc_equipment_depreciation
from sensitivity.config import PARAMETERS, generate_oat_points, DATA_DIR

# =============================================================================
# Q1 基线计算 (自包含, 支持参数覆盖)
# =============================================================================

# Q1 待分析的参数名列表 (与 PARAMETERS 列表中的 name 字段一致)
Q1_PARAM_NAMES = [
    'wind_lcoe',
    'solar_lcoe',
    'tou_price_scale',
    'alkel_om',
    'pemel_om',
    'alkel_rated_scale',
    'pemel_rated_scale',
]


def run_q1_baseline(wind_lcoe=None, solar_lcoe=None, tou_price_scale=None,
                    alkel_om=None, pemel_om=None,
                    alkel_rated_scale=None, pemel_rated_scale=None):
    """运行 Q1 基准计算, 支持参数覆盖。

    除指定参数外, 其余参数使用根 config.py 中的默认值。

    Args:
        wind_lcoe:          风电度电成本 (yuan/kWh), None=默认 0.15
        solar_lcoe:         光伏度电成本 (yuan/kWh), None=默认 0.12
        tou_price_scale:    分时电价缩放系数, None=默认 1.0
        alkel_om:           ALK电解槽运维系数 (yuan/kWh), None=默认 0.10
        pemel_om:           PEM电解槽运维系数 (yuan/kWh), None=默认 0.15
        alkel_rated_scale:  ALK电解槽额定功率缩放, None=默认 1.0
        pemel_rated_scale:  PEM电解槽额定功率缩放, None=默认 1.0

    Returns:
        dict with keys:
            ton_cost, self_use_rate, green_rate, grid_feed_rate
    """
    # ---- 解析覆盖参数 ----
    _wind_lcoe = wind_lcoe if wind_lcoe is not None else root_config.WIND_LCOE
    _solar_lcoe = solar_lcoe if solar_lcoe is not None else root_config.SOLAR_LCOE
    _alkel_om = alkel_om if alkel_om is not None else root_config.ALKEL_OM
    _pemel_om = pemel_om if pemel_om is not None else root_config.PEMEL_OM
    _alkel_rated = root_config.ALKEL_RATED * (alkel_rated_scale if alkel_rated_scale is not None else 1.0)
    _pemel_rated = root_config.PEMEL_RATED * (pemel_rated_scale if pemel_rated_scale is not None else 1.0)
    _tou_scale = tou_price_scale if tou_price_scale is not None else 1.0

    # ---- 加载数据 ----
    wind = root_config.get_typical_wind()      # (24,) MW
    solar = root_config.get_typical_solar()    # (24,) MW
    load = root_config.get_load_profile()      # (24,) MW

    TOTAL_EQ_POWER = _alkel_rated + _pemel_rated + root_config.NH3_RATED
    n_hours = 24

    # ---- 逐时功率平衡 ----
    buy = np.zeros(n_hours)
    sell = np.zeros(n_hours)

    for t in range(n_hours):
        net_power = wind[t] + solar[t] - TOTAL_EQ_POWER - load[t]
        if net_power >= 0:
            sell[t] = net_power
            buy[t] = 0.0
        else:
            buy[t] = -net_power
            sell[t] = 0.0

    # ---- 日总量 (kWh) ----
    wind_kwh = float(np.sum(wind)) * 1000
    solar_kwh = float(np.sum(solar)) * 1000
    load_kwh = float(np.sum(load)) * 1000
    buy_kwh = float(np.sum(buy)) * 1000
    sell_kwh = float(np.sum(sell)) * 1000

    total_gen = wind_kwh + solar_kwh
    total_load = load_kwh + (TOTAL_EQ_POWER * 1000 * n_hours)

    # ---- 绿电指标 ----
    indicators = calc_green_indicators(total_gen, total_load, buy_kwh, sell_kwh)

    # ---- 吨氨成本 ----
    DAILY_NH3 = 36.0          # tons
    ANNUAL_NH3 = DAILY_NH3 * 360

    # 电解槽运维费 (已覆盖参数)
    el_om_cost = (_alkel_rated * 1000 * n_hours * _alkel_om
                  + _pemel_rated * 1000 * n_hours * _pemel_om)

    # 合成氨运维费
    nh3_om_cost = root_config.NH3_RATED * 1000 * n_hours * root_config.NH3_OM

    # 设备折旧 (仅合成氨装置)
    dep_per_ton = calc_equipment_depreciation(
        root_config.NH3_INVESTMENT,
        root_config.NH3_LIFETIME,
        ANNUAL_NH3,
    )
    equipment_depreciation = dep_per_ton * DAILY_NH3

    # 上网售电收入
    grid_sell_revenue = sell_kwh * root_config.GRID_FEEDIN_PRICE

    # 网购电成本 (按分时电价 × tou_scale)
    buy_hourly_kwh = buy * 1000  # (24,) array in kWh
    grid_buy_cost = 0.0
    for t in range(n_hours):
        if buy_hourly_kwh[t] > 0:
            grid_buy_cost += buy_hourly_kwh[t] * root_config.get_price(t) * _tou_scale

    # 风光度电成本 (已覆盖参数)
    wind_cost = wind_kwh * _wind_lcoe
    solar_cost = solar_kwh * _solar_lcoe

    total_cost = (grid_buy_cost + wind_cost + solar_cost
                  + el_om_cost + nh3_om_cost
                  + equipment_depreciation
                  - grid_sell_revenue)

    ton_cost = total_cost / DAILY_NH3

    return {
        'ton_cost': ton_cost,
        'self_use_rate': indicators['self_use_rate'],
        'green_rate': indicators['green_rate'],
        'grid_feed_rate': indicators['grid_feed_rate'],
    }


# =============================================================================
# OAT 敏感性分析主函数
# =============================================================================

def save_oat_results(results: list, output_path: str):
    """将 OAT 结果列表保存为 CSV 文件。

    Args:
        results:     list of dict, 每个 dict 包含 param_name, perturbed_value,
                     ton_cost, self_use_rate, green_rate, grid_feed_rate, cost_change_pct
        output_path: 输出 CSV 文件路径
    """
    fieldnames = [
        'param_name', 'perturbed_value', 'ton_cost',
        'self_use_rate', 'green_rate', 'grid_feed_rate', 'cost_change_pct',
    ]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def q1_sensitivity():
    """执行 Q1 OAT 参数敏感性分析, 输出 CSV 结果文件。"""
    # 确保 data 目录存在
    os.makedirs(DATA_DIR, exist_ok=True)

    # ---- 基线计算 ----
    baseline = run_q1_baseline()
    print("=" * 60)
    print("Q1 OAT 参数敏感性分析")
    print("=" * 60)
    print(f"  基线吨氨成本: {baseline['ton_cost']:.2f} yuan/ton")
    print(f"  基线自发自用率: {baseline['self_use_rate']:.4f}")
    print(f"  基线绿电比例: {baseline['green_rate']:.4f}")
    print(f"  基线上网比例: {baseline['grid_feed_rate']:.4f}")
    print()

    # ---- 参数查找表 ----
    param_map = {p['name']: p for p in PARAMETERS}

    # ---- OAT 逐参数逐点扰动 ----
    results = []
    for param_name in Q1_PARAM_NAMES:
        if param_name not in param_map:
            print(f"  [WARN] 参数 '{param_name}' 未在 PARAMETERS 中定义, 跳过")
            continue

        param_def = param_map[param_name]
        oat_points = generate_oat_points(param_def)

        print(f"  分析参数: {param_def['label']} ({param_name})")
        print(f"    基线值: {param_def['baseline']} {param_def['unit']}")
        print(f"    扰动点数: {len(oat_points)}")

        for val in oat_points:
            kwargs = {param_name: val}
            r = run_q1_baseline(**kwargs)
            cost_change_pct = (
                (r['ton_cost'] - baseline['ton_cost']) / baseline['ton_cost'] * 100
            )
            results.append({
                'param_name': param_name,
                'perturbed_value': val,
                'ton_cost': round(r['ton_cost'], 4),
                'self_use_rate': round(r['self_use_rate'], 6),
                'green_rate': round(r['green_rate'], 6),
                'grid_feed_rate': round(r['grid_feed_rate'], 6),
                'cost_change_pct': round(cost_change_pct, 4),
            })

        # 显示摘要
        param_results = [r for r in results if r['param_name'] == param_name]
        costs = [r['ton_cost'] for r in param_results]
        changes = [abs(r['cost_change_pct']) for r in param_results]
        max_change = max(changes) if changes else 0
        print(f"    吨氨成本范围: {min(costs):.2f} ~ {max(costs):.2f} yuan/ton")
        print(f"    最大变化幅度: {max_change:.2f}%")
        print()

    # ---- 保存结果 ----
    output_path = os.path.join(DATA_DIR, 'q1_oat_results.csv')
    save_oat_results(results, output_path)
    print(f"结果已保存: {output_path}")
    print(f"  总参数数: {len(Q1_PARAM_NAMES)}")
    print(f"  总OAT运行次数: {len(results)}")
    print()

    # ---- 敏感性排序 (按最大成本变化幅度) ----
    param_sensitivity = {}
    for r in results:
        pn = r['param_name']
        if pn not in param_sensitivity:
            param_sensitivity[pn] = {'max_abs_change': 0, 'label': param_map[pn]['label']}
        abs_change = abs(r['cost_change_pct'])
        if abs_change > param_sensitivity[pn]['max_abs_change']:
            param_sensitivity[pn]['max_abs_change'] = abs_change

    sorted_params = sorted(param_sensitivity.items(),
                           key=lambda x: x[1]['max_abs_change'], reverse=True)

    print("--- 参数敏感性排序 (按最大吨氨成本变化幅度) ---")
    for rank, (pn, info) in enumerate(sorted_params, 1):
        print(f"  {rank}. {info['label']:20s}  max Δ = {info['max_abs_change']:.2f}%")

    return results


if __name__ == '__main__':
    q1_sensitivity()
