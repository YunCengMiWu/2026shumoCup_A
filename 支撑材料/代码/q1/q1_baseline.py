# -*- coding: utf-8 -*-
"""
q1_baseline.py — 电工杯A题 问题1: 基准全负荷连续运行分析

假设全部设备 (ALKEL 10MW, PEMEL 10MW, NH3 0.75MW) 24h 满功率连续运行，
日合成氨产量 36 吨。逐时计算网购电量与上网电量，评估绿电指标与吨氨成本。
"""

import sys
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# 修复 Windows 终端中文输出乱码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
import os
# 设置路径以导入上级目录的 config 和 utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv
import config
from utils import (calc_green_indicators, check_green_requirements,
                   classify_scenario, calc_ton_ammonia_cost,
                   calc_equipment_depreciation)


def main():
    # =========================================================================
    # 1. 加载数据
    # =========================================================================
    wind = config.get_typical_wind()      # (24,) MW
    solar = config.get_typical_solar()    # (24,) MW
    load = config.get_load_profile()      # (24,) MW

    TOTAL_EQ_POWER = config.ALKEL_RATED + config.PEMEL_RATED + config.NH3_RATED  # 20.75 MW

    # =========================================================================
    # 2. 逐时计算购电/售电
    # =========================================================================
    n_hours = 24
    buy = np.zeros(n_hours)   # MW
    sell = np.zeros(n_hours)  # MW

    for t in range(n_hours):
        net_power = wind[t] + solar[t] - TOTAL_EQ_POWER - load[t]
        if net_power >= 0:
            sell[t] = net_power
            buy[t] = 0.0
        else:
            buy[t] = -net_power
            sell[t] = 0.0

    # =========================================================================
    # 3. 日总量 (kWh)
    # =========================================================================
    wind_kwh  = float(np.sum(wind))  * 1000
    solar_kwh = float(np.sum(solar)) * 1000
    load_kwh  = float(np.sum(load))  * 1000
    buy_kwh   = float(np.sum(buy))   * 1000
    sell_kwh  = float(np.sum(sell))  * 1000

    total_gen  = wind_kwh + solar_kwh
    total_load = load_kwh + (TOTAL_EQ_POWER * 1000 * n_hours)  # 20.75*1000*24 = 498000

    # =========================================================================
    # 4. 绿电指标计算
    # =========================================================================
    indicators = calc_green_indicators(total_gen, total_load, buy_kwh, sell_kwh)
    pass_self_use, pass_green, pass_feed = check_green_requirements(indicators)
    classification = classify_scenario(indicators)

    # =========================================================================
    # 5. 吨氨成本计算
    # =========================================================================
    DAILY_NH3 = 36.0  # tons
    ANNUAL_NH3 = DAILY_NH3 * 360  # 12960 tons/year

    # 电解槽运维费: ALKEL 10MW × 1000 × 24h × 0.10 + PEMEL 10MW × 1000 × 24h × 0.15
    el_om_cost = (config.ALKEL_RATED * 1000 * 24 * config.ALKEL_OM) + (config.PEMEL_RATED * 1000 * 24 * config.PEMEL_OM)

    # 合成氨运维费: 0.75MW × 1000 × 24h × 0.002
    nh3_om_cost = config.NH3_RATED * 1000 * 24 * config.NH3_OM

    # 设备折旧 (仅合成氨装置, 电解槽折旧计入运维)
    dep_per_ton = calc_equipment_depreciation(config.NH3_INVESTMENT,
                                              config.NH3_LIFETIME,
                                              ANNUAL_NH3)
    equipment_depreciation = dep_per_ton * DAILY_NH3

    # 上网售电收入
    grid_sell_revenue = float(sell_kwh) * config.GRID_FEEDIN_PRICE

    # 逐时网购电量 kWh 数组 (传给 calc_ton_ammonia_cost 自动按分时电价计算)
    buy_hourly_kwh = buy * 1000  # (24,) array in kWh

    ton_cost = calc_ton_ammonia_cost(
        nh3_output_tons=DAILY_NH3,
        grid_buy_kwh=buy_hourly_kwh,
        grid_sell_kwh=sell_kwh,
        grid_buy_cost=0.0,                # 传入数组时此参数被忽略
        grid_sell_revenue=grid_sell_revenue,
        wind_gen_kwh=wind_kwh,
        solar_gen_kwh=solar_kwh,
        el_om_cost=el_om_cost,
        nh3_om_cost=nh3_om_cost,
        equipment_depreciation=equipment_depreciation,
        production_level_tons_per_day=DAILY_NH3,
    )

    # =========================================================================
    # 6. 功率平衡校验
    # =========================================================================
    max_err = 0.0
    for t in range(n_hours):
        # 功率平衡: wind + solar + buy = sell + el + nh3 + load
        net = wind[t] + solar[t] + buy[t] - sell[t] - TOTAL_EQ_POWER - load[t]
        err = abs(net)
        if err > max_err:
            max_err = err

    balance_ok = max_err < 1e-4

    # =========================================================================
    # 7. 输出结果到 stdout
    # =========================================================================
    print("=" * 60)
    print("问题1: 基准全负荷连续运行分析")
    print("=" * 60)
    print()
    print(f"设备总功率: {TOTAL_EQ_POWER} MW  (ALKEL {config.ALKEL_RATED} + PEMEL {config.PEMEL_RATED} + NH3 {config.NH3_RATED})")
    print(f"日合成氨产量: {DAILY_NH3} 吨")
    print()

    print("--- 日电量汇总 (kWh) ---")
    print(f"  风电发电量:     {wind_kwh:>12.2f}")
    print(f"  光伏发电量:     {solar_kwh:>12.2f}")
    print(f"  新能源总发电:   {total_gen:>12.2f}")
    print(f"  常规负荷用电:   {load_kwh:>12.2f}")
    print(f"  设备总用电:     {TOTAL_EQ_POWER * 1000 * 24:>12.2f}")
    print(f"  总用电量:       {total_load:>12.2f}")
    print(f"  网购电量:       {buy_kwh:>12.2f}")
    print(f"  上网电量:       {sell_kwh:>12.2f}")
    print()

    print("--- 绿电指标 ---")
    def mark(ok):
        return '[OK]' if ok else '[X]'

    print(f"  新能源自发自用率: {indicators['self_use_rate']:.4f}  (>0.60 {mark(pass_self_use)})")
    print(f"  总用电量绿电比例: {indicators['green_rate']:.4f}  (>0.30 {mark(pass_green)})")
    print(f"  新能源上网电量比: {indicators['grid_feed_rate']:.4f}  (<0.20 {mark(pass_feed)})")
    print(f"  场景分类: {classification}")
    print()

    print(f"--- 吨氨成本 ---")
    print(f"  吨氨生产成本: {ton_cost:.2f} 元/吨NH3")
    print()

    print(f"功率平衡检查: {'通过' if balance_ok else '失败'}  (最大误差 = {max_err:.2e})")
    print()

    # =========================================================================
    # 8. 保存结果
    # =========================================================================
    os.makedirs('results', exist_ok=True)

    # --- 8a. 时序曲线图 ---
    hours = np.arange(n_hours)
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    axes[0].plot(hours, wind, 'b-', linewidth=1.5, label='风电出力')
    axes[0].set_ylabel('风电 (MW)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hours, solar, 'orange', linewidth=1.5, label='光伏出力')
    axes[1].set_ylabel('光伏 (MW)')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(hours, buy, color='red', alpha=0.7, label='网购电')
    axes[2].set_ylabel('网购电 (MW)')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)

    axes[3].bar(hours, sell, color='green', alpha=0.7, label='上网电')
    axes[3].set_xlabel('小时')
    axes[3].set_ylabel('上网电 (MW)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)

    fig.suptitle('问题1: 基准全负荷连续运行 — 功率曲线', fontsize=14)
    plt.tight_layout()
    fig.savefig('q1_power_curves.png', dpi=150)
    plt.close()
    print(f"图片已保存: q1_power_curves.png")

    # --- 8a2. 逐小时明细 CSV 与可视化 ---
    def _save_q1_hourly_detail(csv_path):
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow(['小时', '风电(MW)', '光伏(MW)', '常规负荷(MW)', '设备用电(MW)', '买电(MW)', '卖电(MW)', '买电(kWh)', '买电成本(¥)', '卖电(kWh)', '卖电收入(¥)'])
            total_buy_cost = 0.0
            for t in range(n_hours):
                buy_mw = buy[t]
                sell_mw = sell[t]
                buy_kwh = buy_mw * 1000
                sell_kwh = sell_mw * 1000
                price = config.get_price(t)
                buy_cost = buy_kwh * price
                sell_revenue = sell_kwh * config.GRID_FEEDIN_PRICE
                total_buy_cost += buy_cost
                writer.writerow([
                    t,
                    f"{wind[t]:.4f}",
                    f"{solar[t]:.4f}",
                    f"{load[t]:.4f}",
                    f"{TOTAL_EQ_POWER:.4f}",
                    f"{buy_mw:.4f}",
                    f"{sell_mw:.4f}",
                    f"{buy_kwh:.2f}",
                    f"{buy_cost:.2f}",
                    f"{sell_kwh:.2f}",
                    f"{sell_revenue:.2f}",
                ])
            # Totals
            writer.writerow([])
            writer.writerow(['合计', '', '', '', f"{TOTAL_EQ_POWER * n_hours:.2f}", '', '', '', f"{total_buy_cost:.2f}", '', f"{float(np.sum(sell)*1000) * config.GRID_FEEDIN_PRICE:.2f}"])

    def _plot_q1_hourly_detail(png_path):
        hours = np.arange(n_hours)
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        ax[0].plot(hours, wind, label='风电 (MW)', color='tab:blue')
        ax[0].plot(hours, solar, label='光伏 (MW)', color='tab:orange')
        ax[0].plot(hours, load, label='常规负荷 (MW)', color='tab:gray')
        ax[0].plot(hours, np.full(n_hours, TOTAL_EQ_POWER), label='设备总功率 (MW)', color='tab:red', linestyle='--')
        ax[0].set_ylabel('功率 (MW)')
        ax[0].legend(loc='upper right')
        ax[0].grid(alpha=0.3)

        # buy/sell bars
        ax[1].bar(hours - 0.2, buy, width=0.4, label='网购 (MW)', color='tab:green', alpha=0.7)
        ax[1].bar(hours + 0.2, sell, width=0.4, label='上网 (MW)', color='tab:purple', alpha=0.7)
        # also show price as secondary axis
        ax2 = ax[1].twinx()
        prices = [config.get_price(t) for t in hours]
        ax2.plot(hours, prices, color='black', linestyle=':', marker='o', label='分时电价 (¥/kWh)')
        ax2.set_ylabel('电价 (¥/kWh)')

        ax[1].set_xlabel('小时')
        ax[1].set_ylabel('功率 (MW)')
        ax[1].legend(loc='upper right')
        ax2.legend(loc='upper left')
        ax[1].grid(alpha=0.3)

        fig.suptitle('问题1: 基准全负荷 — 逐小时明细')
        plt.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)

    hourly_csv = 'q1_hourly_detail.csv'
    hourly_png = 'q1_hourly_detail.png'
    _save_q1_hourly_detail(hourly_csv)
    print(f"逐小时明细已保存: {hourly_csv}")
    _plot_q1_hourly_detail(hourly_png)
    print(f"逐小时图已保存: {hourly_png}")

    # --- 8b. 指标 CSV ---
    csv_path = 'q1_indicators.csv'
    rows = [
        ['新能源自发自用率', f'{indicators["self_use_rate"]:.4f}', '-'],
        ['总用电量绿电比例', f'{indicators["green_rate"]:.4f}', '-'],
        ['新能源上网电量比', f'{indicators["grid_feed_rate"]:.4f}', '-'],
        ['自发自用率合格', '是' if pass_self_use else '否', '-'],
        ['绿电比例合格', '是' if pass_green else '否', '-'],
        ['上网比例合格', '是' if pass_feed else '否', '-'],
        ['场景分类', classification, '-'],
        ['吨氨生产成本', f'{ton_cost:.2f}', '元/吨NH3'],
    ]
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['指标', '数值', '单位'])
        writer.writerows(rows)
    print(f"CSV 已保存: {csv_path}")

    print()
    print("=" * 60)
    print("分析完成")
    print("=" * 60)


if __name__ == '__main__':
    main()
