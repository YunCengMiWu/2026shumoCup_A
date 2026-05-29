# -*- coding: utf-8 -*-
"""
q4 可视化补充脚本
从已有 CSV 读取数据，生成 5 张新图表
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import csv
import numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib import cm
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_production_comparison():
    csv1 = os.path.join(OUT_DIR, 'q4_offgrid_no_storage.csv')
    no_storage = {}
    with open(csv1, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['scenario_name'].strip()
            no_storage[name] = float(row['daily_production_tons'])

    csv2 = os.path.join(OUT_DIR, 'q4_with_storage.csv')
    with_storage = {}
    with open(csv2, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['scenario'].strip()
            with_storage[name] = float(row['daily_production_tons'])

    all_names = sorted(no_storage.keys())
    ns_vals = [no_storage.get(n, 0) for n in all_names]
    ws_vals = [with_storage.get(n, 0) for n in all_names]

    x = np.arange(len(all_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width/2, ns_vals, width, label='离网无储能',
           color='#e74c3c', alpha=0.85, edgecolor='white')
    ax.bar(x + width/2, ws_vals, width, label='离网+储能',
           color='#2ecc71', alpha=0.85, edgecolor='white')

    ax.set_ylabel('日产量 (吨/天)', fontsize=12)
    ax.set_xlabel('风光场景', fontsize=12)
    ax.set_title('离网运行 24场景日产量对比 (无储能 vs 有储能)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=36, color='#3498db', linestyle='--', linewidth=1.5, alpha=0.7, label='36吨/天基准')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    for i, (ns, ws) in enumerate(zip(ns_vals, ws_vals)):
        delta = ws - ns
        if delta > 0.5:
            ax.annotate(f'+{delta:.1f}', (x[i] + width/2, ws),
                       ha='center', va='bottom', fontsize=7,
                       color='#27ae60', fontweight='bold')

    png_path = os.path.join(OUT_DIR, 'q4_production_comparison.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'[1/5] 产量对比图已保存: {png_path}')


def plot_curtailment_heatmap():
    csv1 = os.path.join(OUT_DIR, 'q4_offgrid_no_storage.csv')
    data = []
    with open(csv1, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            w = int(row['wind'])
            s = int(row['solar'])
            curt = float(row['total_curtailed_mwh'])
            data.append((w, s, curt))

    wind_indices = sorted(set(d[0] for d in data))
    solar_indices = sorted(set(d[1] for d in data))
    NX, NY = len(wind_indices), len(solar_indices)

    Z = np.zeros((NY, NX))
    for w, s, curt in data:
        wi = wind_indices.index(w)
        si = solar_indices.index(s)
        Z[si, wi] = curt

    fig, ax = plt.subplots(figsize=(10, 7))
    z_max = Z.max() if Z.max() > 0 else 1
    im = ax.imshow(Z, aspect='auto', cmap='YlOrRd', origin='upper',
                   extent=[-0.5, NX - 0.5, NY - 0.5, -0.5])
    ax.set_xticks(range(NX))
    ax.set_xticklabels([f'W{i}' for i in wind_indices], fontsize=11)
    ax.set_xlabel('Wind Index', fontsize=13, labelpad=10)
    ax.set_yticks(range(NY))
    ax.set_yticklabels([f'S{i}' for i in solar_indices], fontsize=11)
    ax.set_ylabel('Solar Index', fontsize=13, labelpad=10)

    for si in range(NY):
        for wi in range(NX):
            val = Z[si, wi]
            color = 'white' if val > z_max * 0.4 else 'black'
            ax.text(wi, si, f'{val:.1f}', ha='center', va='center',
                    fontsize=10, fontweight='bold', color=color)

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Curtailed Energy (MWh)', fontsize=12)
    ax.set_title('Off-Grid No-Storage: Curtailed Energy by Scenario (MWh)',
                 fontsize=15, fontweight='bold', pad=15)
    plt.tight_layout()
    png_path = os.path.join(OUT_DIR, 'q4_curtailment_heatmap.png')
    fig.savefig(png_path, dpi=150)
    plt.close()
    print(f'[2/5] Curtailment heatmap saved: {png_path}')


def plot_economic_comparison():
    csv_path = os.path.join(OUT_DIR, 'q4_economic_comparison.csv')
    modes, costs, productions = [], [], []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            modes.append(row['mode'].strip())
            costs.append(float(row['average_ton_cost_yuan']))
            productions.append(float(row['annual_production_tons']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors1 = ['#2ecc71', '#3498db']
    bars1 = ax1.bar(modes, costs, color=colors1, edgecolor='white', width=0.5)
    ax1.set_ylabel('吨氨成本 (元/吨)', fontsize=12)
    ax1.set_title('吨氨成本对比', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    for bar, cost in zip(bars1, costs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                 f'{cost:.0f}', ha='center', fontsize=13, fontweight='bold')
    if len(costs) >= 2:
        delta = costs[0] - costs[1]
        pct = delta / costs[1] * 100
        sign = '+' if delta > 0 else ''
        ax1.annotate(f'Delta = {sign}{delta:.0f} 元/吨 ({sign}{pct:.1f}%)',
                    xy=(0.5, max(costs)), ha='center', fontsize=11,
                    color='#e74c3c' if delta > 0 else '#27ae60',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff9c4', alpha=0.8))

    colors2 = ['#e67e22', '#8e44ad']
    bars2 = ax2.bar(modes, productions, color=colors2, edgecolor='white', width=0.5)
    ax2.set_ylabel('年产量 (吨)', fontsize=12)
    ax2.set_title('年产量对比', fontsize=13, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    for bar, prod in zip(bars2, productions):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
                 f'{prod:,.0f}', ha='center', fontsize=13, fontweight='bold')

    fig.suptitle('离网+储能 vs 并网(无储能) 经济性对比', fontsize=14, fontweight='bold', y=1.02)

    png_path = os.path.join(OUT_DIR, 'q4_economic_comparison.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'[3/5] 经济对比图已保存: {png_path}')


def plot_cost_scatter():
    csv1 = os.path.join(OUT_DIR, 'q4_offgrid_no_storage.csv')
    ns_data = []
    with open(csv1, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ns_data.append({
                'scenario': row['scenario_name'].strip(),
                'prod': float(row['daily_production_tons']),
                'cost': float(row['ton_cost_yuan']),
            })

    csv2 = os.path.join(OUT_DIR, 'q4_with_storage.csv')
    ws_data = []
    with open(csv2, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ws_data.append({
                'scenario': row['scenario'].strip(),
                'prod': float(row['daily_production_tons']),
                'cost': float(row['ton_cost_yuan']),
            })

    fig, ax = plt.subplots(figsize=(12, 7))

    ns_x = [d['prod'] for d in ns_data]
    ns_y = [d['cost'] for d in ns_data]
    ws_x = [d['prod'] for d in ws_data]
    ws_y = [d['cost'] for d in ws_data]

    ax.scatter(ns_x, ns_y, c='#e74c3c', s=100, alpha=0.7, edgecolors='white',
              linewidth=1, label='离网无储能', zorder=3)
    ax.scatter(ws_x, ws_y, c='#2ecc71', s=100, alpha=0.7, edgecolors='white',
              linewidth=1, label='离网+储能', zorder=4)

    for ns, ws in zip(ns_data, ws_data):
        ax.plot([ns['prod'], ws['prod']], [ns['cost'], ws['cost']],
                'k-', alpha=0.15, linewidth=0.8, zorder=1)

    ax.set_xlabel('日产量 (吨/天)', fontsize=12)
    ax.set_ylabel('吨氨成本 (元/吨)', fontsize=12)
    ax.set_title('离网运行 24场景 日产量-吨氨成本分布 (无储能 vs 有储能)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(alpha=0.3)

    avg_ns_x, avg_ns_y = np.mean(ns_x), np.mean(ns_y)
    avg_ws_x, avg_ws_y = np.mean(ws_x), np.mean(ws_y)
    ax.scatter([avg_ns_x], [avg_ns_y], marker='X', c='darkred', s=200,
              zorder=5, edgecolors='white', linewidth=1.5)
    ax.scatter([avg_ws_x], [avg_ws_y], marker='X', c='darkgreen', s=200,
              zorder=5, edgecolors='white', linewidth=1.5)
    ax.annotate(f'平均 {avg_ns_y:.0f}元/吨', (avg_ns_x, avg_ns_y),
               textcoords="offset points", xytext=(10, -15), fontsize=9)
    ax.annotate(f'平均 {avg_ws_y:.0f}元/吨', (avg_ws_x, avg_ws_y),
               textcoords="offset points", xytext=(-10, 10), fontsize=9)

    png_path = os.path.join(OUT_DIR, 'q4_cost_scatter.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'[4/5] 成本散点图已保存: {png_path}')


def plot_utilization_stacked():
    csv1 = os.path.join(OUT_DIR, 'q4_offgrid_no_storage.csv')
    scenarios, utilized_pct, curtailed_pct = [], [], []

    with open(csv1, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row['scenario_name'].strip()
            curt = float(row['total_curtailed_mwh'])
            wind_gen = float(row['wind_gen_mwh'])
            solar_gen = float(row['solar_gen_mwh'])
            total_gen = wind_gen + solar_gen
            if total_gen > 0:
                curt_p = curt / total_gen * 100
                util_p = 100 - curt_p
                scenarios.append(name)
                utilized_pct.append(util_p)
                curtailed_pct.append(curt_p)

    x = np.arange(len(scenarios))
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar(x, utilized_pct, color='#2ecc71', alpha=0.85, label='有效利用', edgecolor='white')
    ax.bar(x, curtailed_pct, bottom=utilized_pct, color='#e74c3c', alpha=0.85,
           label='弃电', edgecolor='white')

    for i, cp in enumerate(curtailed_pct):
        if cp > 2:
            ax.text(i, utilized_pct[i] + cp/2, f'{cp:.1f}%', ha='center',
                   va='center', fontsize=7, color='white', fontweight='bold')

    ax.set_ylabel('占比 (%)', fontsize=12)
    ax.set_xlabel('风光场景', fontsize=12)
    ax.set_title('离网无储能 24场景 风光发电利用/弃电比例', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, rotation=45, ha='right', fontsize=9)
    ax.axhline(y=100, color='black', linewidth=0.5)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 105)

    png_path = os.path.join(OUT_DIR, 'q4_utilization_stacked.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'[5/5] 利用率堆叠图已保存: {png_path}')


if __name__ == '__main__':
    print('=' * 50)
    print('q4 可视化补充 - 从CSV生成5张新图表')
    print('=' * 50)
    plot_production_comparison()
    plot_curtailment_heatmap()
    plot_economic_comparison()
    plot_cost_scatter()
    plot_utilization_stacked()
    print('\n全部图表生成完成!')
