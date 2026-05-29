# -*- coding: utf-8 -*-
"""
q2 3D 吨氨成本分布图 — 方柱1:1 + 柱间距=柱宽 + 高视角
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import csv, os, numpy as np
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib import cm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(SCRIPT_DIR, 'q2_annual_summary.csv')
data = {}
all_scenarios, all_levels = set(), set()
with open(csv_path, 'r', encoding='utf-8-sig') as f:
    reader = csv.DictReader(f)
    for row in reader:
        w = int(row['风电档位']); s = int(row['光伏档位'])
        daily = int(float(row['日产量(吨/天)']))
        cost = float(row['吨氨成本(\u00a5/ton)'])
        data[(w, s, daily)] = cost
        all_scenarios.add((w, s)); all_levels.add(daily)

wind_idx = sorted(set(w for w, s in all_scenarios))
solar_idx = sorted(set(s for w, s in all_scenarios))
levels = sorted(all_levels, reverse=True)
labels = [f'W{w}S{s}' for s in solar_idx for w in wind_idx]
NX, NY = len(labels), len(levels)

Y_GAP = 1.8
xpos = np.arange(NX); ypos = np.arange(NY) * Y_GAP
X, Y = np.meshgrid(xpos, ypos)
Z = np.zeros((NY, NX))
for si, s in enumerate(solar_idx):
    for wi, w in enumerate(wind_idx):
        c = si * len(wind_idx) + wi
        for li, lv in enumerate(levels):
            Z[li, c] = data.get((w, s, lv), np.nan)

fig = plt.figure(figsize=(18, 11))
ax = fig.add_subplot(111, projection='3d')

z_valid = Z[~np.isnan(Z)]
z_min, z_max = np.min(z_valid), np.max(z_valid)
norm = plt.Normalize(z_min, z_max)
cmap = plt.cm.ocean

BAR = 0.5
dz = Z.ravel()
colors = cmap(norm(np.where(np.isnan(dz), z_min, dz)))

ax.bar3d(X.ravel(), Y.ravel(), np.zeros_like(dz),
         BAR, BAR, np.nan_to_num(dz, 0),
         color=colors, alpha=0.88, edgecolor='#222', linewidth=0.2)

ax.set_xlabel('场景 (风电 × 光伏)', fontsize=12, labelpad=14)
ax.set_ylabel('日产量 (吨/天)', fontsize=12, labelpad=14)
ax.set_zlabel('吨氨成本 (元/吨)', fontsize=12, labelpad=14)
ax.set_title('Q2 离散开关 吨氨成本三维分布\n24场景 × 5产量 = 120 MILP 求解',
             fontsize=15, fontweight='bold', pad=22)

ax.set_xticks(xpos + BAR/2)
ax.set_xticklabels(labels, rotation=55, ha='right', fontsize=7.5)
ax.set_yticks(ypos + BAR/2)
ax.set_yticklabels([f'{l} t/d' for l in levels], fontsize=10)

ax.set_box_aspect([NX, NY * Y_GAP, NX * 0.35])
ax.view_init(elev=32, azim=122)

sm = cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, shrink=0.45, aspect=18, pad=0.08)
cbar.set_label('吨氨成本 (元/吨)', fontsize=11)

costs = list(data.values())
mean_c, median_c = np.mean(costs), np.median(costs)
best = min(data.items(), key=lambda x: x[1])
(bw, bs, bl), bc = best
info = f'均值 {mean_c:.0f}    中位数 {median_c:.0f}    最优 W{bw}S{bs} {bl}t/d={bc:.0f} 元/吨'
ax.text2D(0.01, 0.97, info, transform=ax.transAxes, fontsize=10,
          bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                    edgecolor='#0f3460', alpha=0.85), color='white')

png = os.path.join(SCRIPT_DIR, 'q2_3d_cost.png')
plt.savefig(png, dpi=150, bbox_inches='tight'); plt.close()
print(f'q2 3D: {png}')
