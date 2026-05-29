# -*- coding: utf-8 -*-
"""
q2 可视化补充脚本
从已有 CSV 读取数据，生成 4 张新图表
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
from collections import defaultdict
import os

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_classification_pie():
    csv_path = os.path.join(OUT_DIR, 'q2_annual_summary.csv')
    counts = defaultdict(int)
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            counts[row['分类'].strip()] += 1

    labels = ['全满足', '部分满足', '全不满足']
    sizes = [counts.get(l, 0) for l in labels]
    colors = ['#2ecc71', '#f39c12', '#e74c3c']
    explode = (0.02, 0.02, 0.05)

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=None, colors=colors,
        autopct='%1.1f%%', startangle=140, textprops={'fontsize': 12}
    )
    legend_labels = [f'{l} ({s} 个方案)' for l, s in zip(labels, sizes)]
    ax.legend(wedges, legend_labels, title="绿电指标分类",
              loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=11)
    ax.set_title('24场景x5产量 绿电直连指标达标分类', fontsize=14, fontweight='bold')

    png_path = os.path.join(OUT_DIR, 'q2_classification_pie.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[1/4] 分类饼图已保存: {png_path}')


def plot_production_vs_cost():
    csv_path = os.path.join(OUT_DIR, 'q2_annual_summary.csv')
    levels, costs = [], []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['场景'].strip() == 'W0S0':
                levels.append(int(float(row['日产量(吨/天)'])))
                costs.append(float(row['吨氨成本(\u00a5/ton)']))

    sorted_data = sorted(zip(levels, costs), key=lambda x: x[0])
    levels, costs = zip(*sorted_data)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors_bar = ['#3498db', '#2980b9', '#1abc9c', '#f39c12', '#e74c3c']
    bars = ax1.bar(range(len(levels)), costs, color=colors_bar, edgecolor='white', width=0.6)
    ax1.set_xticks(range(len(levels)))
    ax1.set_xticklabels([f'{l} 吨/天' for l in levels], fontsize=11)
    ax1.set_ylabel('吨氨成本 (元/吨)', fontsize=12)
    ax1.set_xlabel('日产量', fontsize=12)
    ax1.set_title('典型场景(W0S0) 不同日产量吨氨成本对比', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    for bar, cost in zip(bars, costs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                 f'{cost:.0f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    best_idx = costs.index(min(costs))
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

    runtime_hours = [24, 21, 18, 15, 12]
    ax2 = ax1.twinx()
    ax2.plot(range(len(levels)), runtime_hours, 'D-', color='#8e44ad',
             linewidth=2, markersize=10, label='运行小时数')
    ax2.set_ylabel('运行小时数 (h)', fontsize=12, color='#8e44ad')
    ax2.tick_params(axis='y', labelcolor='#8e44ad')
    ax2.set_ylim(0, 26)

    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend([bars[best_idx], lines2[0]],
               [f'最优: {levels[best_idx]}吨/天', '运行小时'],
               loc='upper left', fontsize=10)

    png_path = os.path.join(OUT_DIR, 'q2_production_vs_cost.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'[2/4] 产量-成本柱状图已保存: {png_path}')


def plot_schedule_gantt():
    csv_path = os.path.join(OUT_DIR, 'q2_typical_schedule.csv')
    sections = []
    current_section = {'title': '', 'rows': []}
    in_data = False
    header_parsed = False

    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_section['rows']:
                    sections.append(current_section)
                    current_section = {'title': '', 'rows': []}
                    in_data = False
                    header_parsed = False
                continue
            if '日产量' in line:
                current_section['title'] = line
                continue
            if '小时,ON' in line:
                header_parsed = True
                in_data = True
                continue
            if in_data and header_parsed:
                parts = line.split(',')
                if len(parts) >= 7 and parts[0].strip().isdigit():
                    current_section['rows'].append({
                        'hour': int(parts[0]), 'on': int(parts[1]),
                        'buy': float(parts[2]), 'sell': float(parts[3]),
                        'wind': float(parts[4]), 'solar': float(parts[5]),
                        'load': float(parts[6]),
                    })
        if current_section['rows']:
            sections.append(current_section)

    if not sections:
        print('[3/4] 警告: 无法解析 typical_schedule.csv, 跳过甘特图')
        return

    section_scores = []
    for sec in sections:
        title = sec['title']
        try:
            daily_out = int(title.replace('日产量', '').replace('吨/天', '').strip())
        except:
            daily_out = 0
        total_buy = sum(r['buy'] for r in sec['rows'])
        total_sell = sum(r['sell'] for r in sec['rows'])
        runtime = sum(1 for r in sec['rows'] if r['on'] == 1)
        section_scores.append((daily_out, total_buy, total_sell, runtime, sec))

    section_scores.sort(key=lambda x: x[1] - x[2] * 0.3)
    best_daily, _, _, runtime, best_sec = section_scores[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={'height_ratios': [1, 3]}, sharex=True)

    hours = [r['hour'] for r in best_sec['rows']]
    on_vals = [r['on'] for r in best_sec['rows']]
    buy_vals = [r['buy'] for r in best_sec['rows']]
    sell_vals = [r['sell'] for r in best_sec['rows']]
    wind_vals = [r['wind'] for r in best_sec['rows']]
    solar_vals = [r['solar'] for r in best_sec['rows']]
    load_vals = [r['load'] for r in best_sec['rows']]

    ax1.fill_between(hours, 0, wind_vals, alpha=0.7, color='#3498db', label='风电')
    ax1.fill_between(hours, wind_vals, np.array(wind_vals)+np.array(solar_vals),
                     alpha=0.7, color='#f39c12', label='光伏')
    ax1.fill_between(hours, np.array(wind_vals)+np.array(solar_vals),
                     np.array(wind_vals)+np.array(solar_vals)+np.array(buy_vals),
                     alpha=0.5, color='#e74c3c', label='购电', hatch='//')
    ax1.plot(hours, load_vals, 'k-', linewidth=2, label='常规负荷')
    ax1.set_ylabel('功率 (MW)', fontsize=11)
    ax1.set_title(f'典型场景 最优日产量{best_daily}吨/天 功率平衡曲线', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, ncol=2)
    ax1.grid(alpha=0.3)

    for i, (h, on) in enumerate(zip(hours, on_vals)):
        if on == 1:
            ax2.barh(0, 0.9, left=h, height=0.6, color='#2ecc71', edgecolor='white')
        else:
            ax2.barh(0, 0.9, left=h, height=0.6, color='#bdc3c7', edgecolor='white')

    for i, (h, sell) in enumerate(zip(hours, sell_vals)):
        if sell > 0:
            ax2.barh(1, 0.9, left=h, height=0.6, color='#e67e22', alpha=0.6, edgecolor='white')

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['制氨设备(ON/OFF)', '余电上网(售电>0)'], fontsize=10)
    ax2.set_xlabel('小时', fontsize=12)
    ax2.set_xlim(-0.5, 23.5)
    ax2.set_xticks(range(0, 24, 2))
    ax2.grid(axis='x', alpha=0.3)
    ax2.text(23.5, 0, f'运行{runtime}h', ha='right', va='center',
             fontsize=11, fontweight='bold', color='#2ecc71')

    png_path = os.path.join(OUT_DIR, 'q2_schedule_gantt.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'[3/4] 调度甘特图已保存: {png_path}')


def plot_scenario_heatmap():
    csv_path = os.path.join(OUT_DIR, 'q2_annual_summary.csv')
    data = []
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            daily_out = int(float(row['日产量(吨/天)']))
            if daily_out == 72:
                w = int(row['风电档位'])
                s = int(row['光伏档位'])
                cost = float(row['吨氨成本(\u00a5/ton)'])
                data.append((w, s, cost))

    if not data:
        print('[4/4] 警告: 无72t/d数据, 跳过3D图')
        return

    wind_indices = sorted(set(d[0] for d in data))
    solar_indices = sorted(set(d[1] for d in data))
    NX, NY = len(wind_indices), len(solar_indices)

    Z = np.zeros((NY, NX))
    for w, s, cost in data:
        wi = wind_indices.index(w)
        si = solar_indices.index(s)
        Z[si, wi] = cost

    xpos = np.arange(NX)
    ypos = np.arange(NY)
    X, Y = np.meshgrid(xpos, ypos)

    fig = plt.figure(figsize=(14, 9))
    ax = fig.add_subplot(111, projection='3d')

    z_min, z_max = np.min(Z), np.max(Z)
    norm = plt.Normalize(z_min, z_max)
    cmap_name = plt.cm.RdYlGn_r

    BAR = 0.55
    dz = Z.ravel()
    colors = cmap_name(norm(dz))

    ax.bar3d(X.ravel(), Y.ravel(), np.zeros_like(dz),
             BAR, BAR, dz,
             color=colors, alpha=0.88, edgecolor='#222',
             linewidth=0.2)

    ax.set_xlabel('风电档位', fontsize=12, labelpad=14)
    ax.set_ylabel('光伏档位', fontsize=12, labelpad=14)
    ax.set_zlabel('吨氨成本 (元/吨)', fontsize=12, labelpad=14)
    ax.set_title('日产量72吨/天 各风光场景吨氨成本 3D 分布 (元/吨)',
                 fontsize=15, fontweight='bold', pad=22)

    ax.set_xticks(xpos + BAR / 2)
    ax.set_xticklabels([f'W{i}' for i in wind_indices], fontsize=10)
    ax.set_yticks(ypos + BAR / 2)
    ax.set_yticklabels([f'S{i}' for i in solar_indices], fontsize=10)

    ax.set_box_aspect([NX, NY, NX * 0.55])
    ax.view_init(elev=28, azim=130)

    for si in range(NY):
        for wi in range(NX):
            val = Z[si, wi]
            ax.text(wi + BAR / 2, si + BAR / 2, val + z_max * 0.02,
                    f'{val:.0f}', ha='center', va='bottom',
                    fontsize=8, fontweight='bold', color='#333')

    sm = cm.ScalarMappable(cmap=cmap_name, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=18, pad=0.1)
    cbar.set_label('吨氨成本 (元/吨)', fontsize=11)

    best = min(data, key=lambda d: d[2])
    print(f'  最佳72t/d场景: W{best[0]}S{best[1]}, 成本={best[2]:.0f} 元/吨')

    png_path = os.path.join(OUT_DIR, 'q2_scenario_heatmap.png')
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f'[4/4] 场景3D图已保存: {png_path}')


if __name__ == '__main__':
    print('=' * 50)
    print('q2 可视化补充 - 从CSV生成4张新图表')
    print('=' * 50)
    plot_classification_pie()
    plot_production_vs_cost()
    plot_schedule_gantt()
    plot_scenario_heatmap()
    print('\n全部图表生成完成!')
