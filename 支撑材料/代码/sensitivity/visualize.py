# -*- coding: utf-8 -*-
"""
sensitivity/visualize.py — 统一可视化脚本
读取 sensitivity/data/q*_oat_results.csv，生成龙卷风图、敏感性曲线、热力图等
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from matplotlib import cm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Parameter Chinese labels
PARAM_LABELS = {
    'wind_lcoe': '风电LCOE',
    'solar_lcoe': '光伏LCOE',
    'tou_price_scale': '分时电价系数',
    'alkel_om': 'ALK运维费',
    'pemel_om': 'PEM运维费',
    'alkel_rated_scale': 'ALK容量缩放',
    'pemel_rated_scale': 'PEM容量缩放',
    'nh3_load_min': 'NH$_3$最低负荷率',
    'nh3_ramp_rate': 'NH$_3$爬坡率',
    'green_penalty_self_use': '$\\lambda_1$自发自用率惩罚',
    'green_penalty_green_rate': '$\\lambda_2$绿电比例惩罚',
    'green_penalty_feed_rate': '$\\lambda_3$上网比例惩罚',
    'el_om_blended': '电解运维(混合)',
    'storage_investment': '储能投资成本',
    'alk_h2_per_mw': 'ALK制氢效率$\\eta_{alk}$',
    'pem_h2_per_mw': 'PEM制氢效率$\\eta_{pem}$',
}

def label(pname):
    return PARAM_LABELS.get(pname, pname)

def load_q1():
    df = pd.read_csv(os.path.join(DATA_DIR, 'q1_oat_results.csv'))
    df['question'] = 'Q1'
    df['cost_change'] = df['cost_change_pct']
    return df

def load_q2():
    df = pd.read_csv(os.path.join(DATA_DIR, 'q2_oat_results.csv'))
    df['question'] = 'Q2'
    df['cost_change'] = df['ton_cost_rel_change_pct']
    df['ton_cost'] = df['ton_ammonia_cost']
    return df

def load_q3():
    df = pd.read_csv(os.path.join(DATA_DIR, 'q3_oat_results.csv'))
    df['question'] = 'Q3'
    # Compute cost change vs baseline for each scenario/param combo
    baselines = df[df['is_baseline'] == True].groupby('scenario')['ton_ammonia_cost'].first()
    def get_change(row):
        bl = baselines.get(row['scenario'], row['ton_ammonia_cost'])
        if bl == 0: return 0
        return (row['ton_ammonia_cost'] - bl) / bl * 100
    df['cost_change'] = df.apply(get_change, axis=1)
    df['ton_cost'] = df['ton_ammonia_cost']
    return df

def load_q4():
    df = pd.read_csv(os.path.join(DATA_DIR, 'q4_oat_results.csv'))
    return df

def plot_tornado():
    """综合龙卷风图：所有问题参数按影响幅度排序"""
    dfs = []
    for loader, qname in [(load_q1, 'Q1'), (load_q2, 'Q2'), (load_q3, 'Q3')]:
        try:
            df = loader()
            summary = df.groupby('param_name').agg(
                max_change=('cost_change', lambda x: x.abs().max()),
                min_change=('cost_change', 'min'),
            ).reset_index()
            summary['question'] = qname
            dfs.append(summary)
        except Exception as e:
            print(f"  [警告] 加载{qname}数据失败: {e}")

    if not dfs:
        print("  [跳过] 无可用数据")
        return

    all_summary = pd.concat(dfs, ignore_index=True)
    # Take max absolute change per param across questions
    impact = all_summary.groupby('param_name')['max_change'].max().sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#e74c3c' if v < 0 else '#3498db' for v in impact.values]
    bars = ax.barh([label(p) for p in impact.index], impact.values, color=colors, edgecolor='white')

    ax.set_xlabel('吨氨成本最大变化幅度 (%)', fontsize=13)
    ax.set_title('参数对吨氨成本影响范围 (综合龙卷风图)', fontsize=15, fontweight='bold')
    ax.axvline(0, color='black', linewidth=0.8)

    for bar, val in zip(bars, impact.values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val:.1f}%', va='center', fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'tornado_overall.png'), dpi=150)
    plt.close()
    print("  [OK] tornado_overall.png")

def plot_q1_curves():
    """Q1敏感性曲线"""
    try:
        df = load_q1()
    except:
        return

    params = df['param_name'].unique()
    n = len(params)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten() if n > 1 else [axes]

    baseline = df[df['perturbed_value'] == df.groupby('param_name')['perturbed_value'].transform(
        lambda x: x.iloc[len(x)//2])]['ton_cost'].iloc[0] if len(df) > 0 else 0

    for i, param in enumerate(params):
        ax = axes[i]
        pdata = df[df['param_name'] == param].sort_values('perturbed_value')
        ax.plot(pdata['perturbed_value'], pdata['ton_cost'], 'o-', color='#2ecc71', linewidth=2, markersize=6)
        ax.axhline(baseline, color='gray', linestyle='--', alpha=0.7, label=f'基准 {baseline:.0f}¥/t')
        ax.set_title(label(param), fontsize=11)
        ax.set_xlabel('参数值')
        ax.set_ylabel('吨氨成本 (¥/t)')
        ax.grid(True, alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Q1 基准场景参数敏感性曲线', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q1_curves.png'), dpi=150)
    plt.close()
    print("  [OK] q1_curves.png")

def plot_q2_curves():
    """Q2敏感性曲线 (聚合场景)"""
    try:
        df = load_q2()
    except:
        return

    params = df['param_name'].unique()
    n = len(params)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, param in enumerate(params):
        ax = axes[i]
        pdata = df[df['param_name'] == param]
        grouped = pdata.groupby('perturbed_value')['ton_cost'].agg(['mean', 'std'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                    fmt='o-', color='#e67e22', linewidth=2, markersize=6, capsize=3)
        ax.set_title(label(param), fontsize=11)
        ax.set_xlabel('参数值')
        ax.set_ylabel('吨氨成本 (¥/t)')
        ax.grid(True, alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Q2 离散制氨MILP参数敏感性曲线 (均值±标准差)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q2_curves.png'), dpi=150)
    plt.close()
    print("  [OK] q2_curves.png")

def plot_q3_curves():
    """Q3敏感性曲线"""
    try:
        df = load_q3()
    except:
        return

    params = df['param_name'].unique()
    n = len(params)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, param in enumerate(params):
        ax = axes[i]
        pdata = df[df['param_name'] == param]
        grouped = pdata.groupby('param_value')['ton_cost'].agg(['mean', 'std'])
        ax.errorbar(grouped.index, grouped['mean'], yerr=grouped['std'],
                    fmt='o-', color='#9b59b6', linewidth=2, markersize=6, capsize=3)
        ax.set_title(label(param), fontsize=11)
        ax.set_xlabel('参数值')
        ax.set_ylabel('吨氨成本 (¥/t)')
        ax.grid(True, alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle('Q3 连续制氨MILP参数敏感性曲线 (均值±标准差)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q3_curves.png'), dpi=150)
    plt.close()
    print("  [OK] q3_curves.png")

def plot_q4_capacity():
    """Q4容量-产量曲线"""
    try:
        df = load_q4()
    except:
        return

    part_a = df[df['section'] == 'part_a'].copy()
    if part_a.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(part_a['alpha'], part_a['min_daily_production_tpd'],
            'o-', color='#1abc9c', linewidth=2, markersize=6)

    # Annotate worst scenarios
    for _, row in part_a.iterrows():
        if row['alpha'] in [0.2, 0.5, 1.0, 1.5, 2.0]:
            ax.annotate(row['worst_scenario'], (row['alpha'], row['min_daily_production_tpd']),
                       textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)

    ax.set_xlabel('风光容量系数 α', fontsize=13)
    ax.set_ylabel('最差场景日产量 (吨/天)', fontsize=13)
    ax.set_title('Q4 离网运行：风光容量与最小日产量关系', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(36, color='red', linestyle='--', alpha=0.5, label='最低目标 36 t/d')
    ax.legend()

    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q4_capacity_curve.png'), dpi=150)
    plt.close()
    print("  [OK] q4_capacity_curve.png")

def plot_importance():
    """参数重要性分组柱状图"""
    dfs = []
    for loader, qname in [(load_q1, 'Q1'), (load_q2, 'Q2'), (load_q3, 'Q3')]:
        try:
            df = loader()
            impact = df.groupby('param_name')['cost_change'].apply(lambda x: x.abs().max()).reset_index()
            impact['question'] = qname
            dfs.append(impact)
        except:
            pass

    if not dfs:
        return

    all_impact = pd.concat(dfs, ignore_index=True)
    pivot = all_impact.pivot_table(values='cost_change', index='param_name', columns='question', aggfunc='max').fillna(0)

    # Sort by total impact
    pivot['total'] = pivot.sum(axis=1)
    pivot = pivot.sort_values('total', ascending=True)

    n_params = len(pivot)
    fig, ax = plt.subplots(figsize=(14, max(6, 0.35 * n_params)))
    questions = [c for c in ['Q1', 'Q2', 'Q3'] if c in pivot.columns]
    colors_q = {'Q1': '#2ecc71', 'Q2': '#e67e22', 'Q3': '#9b59b6'}

    x = np.arange(len(pivot))
    width = min(0.25, 0.8 / max(len(questions), 1))
    for j, q in enumerate(questions):
        vals = pivot[q].values
        ax.barh(x + j*width, vals, width, label=q, color=colors_q.get(q, '#95a5a6'))

    ax.set_yticks(x + width)
    ax.set_yticklabels([label(p) for p in pivot.index])
    ax.set_xlabel('吨氨成本最大变化幅度 (%)', fontsize=13)
    ax.set_title('参数重要性对比 (各问题最大影响幅度)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    plt.subplots_adjust(left=0.30)
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'importance_barchart.png'), dpi=150)
    plt.close()
    print("  [OK] importance_barchart.png")

def plot_green_heatmap_q3():
    """Q3参数敏感性3D柱状图（仿Q2 3D风格：bar3d + 场景分组 + 成本着色）"""
    try:
        df = load_q3()
    except:
        return

    scenarios = sorted(df['scenario'].unique())
    all_params = list(df['param_name'].unique())
    
    params = []
    for p in all_params:
        for sc in scenarios:
            costs = df[(df['param_name'] == p) & (df['scenario'] == sc)]['ton_cost'].dropna()
            if costs.nunique() > 1:
                params.append(p)
                break
    params = list(dict.fromkeys(params))
    
    if not params:
        print("  [SKIP] No discriminating Q3 parameters")
        return
    
    Y_GAP = 1.8
    BAR = 0.5
    n_show = min(len(scenarios), 3)
    
    # Collect all data first to compute global z range
    all_z = []
    scenario_data = []
    for sc_idx, scenario in enumerate(scenarios[:n_show]):
        sdata = df[df['scenario'] == scenario]
        all_levels = []
        for param in params:
            pdata = sdata[sdata['param_name'] == param].sort_values('param_value')
            costs = pdata['ton_cost'].dropna().values
            if len(costs) > 0: all_levels.append((param, costs))
        n_params = len(all_levels)
        if n_params == 0: continue
        max_pts = max(len(c) for _, c in all_levels)
        Z_local = np.full((n_params, max_pts), np.nan)
        labels_local = []
        for pi, (param, costs) in enumerate(all_levels):
            Z_local[pi, :len(costs)] = costs
            labels_local.append(label(param))
        scenario_data.append((scenario, Z_local, labels_local, n_params, max_pts))
        all_z.extend(Z_local[~np.isnan(Z_local)].tolist())
    
    if not scenario_data:
        print("  [SKIP] No Q3 data")
        return
    
    z_min, z_max = np.min(all_z), np.max(all_z)
    norm = plt.Normalize(z_min, z_max)
    cmap_name = plt.cm.RdYlGn_r
    
    # Single combined chart: stack scenarios on Y-axis
    SCENARIO_GAP = 3.0    # gap between scenario groups
    PARAM_GAP = 1.6       # gap between params within a scenario
    
    all_X, all_Y, all_Z = [], [], []
    all_labels = []
    y_offset = 0.0
    max_pts_overall = max(mp for _, _, _, _, mp in scenario_data)
    
    for scenario, Z, labels, n_params, max_pts in scenario_data:
        xpos = np.arange(max_pts)
        ypos = np.arange(n_params) * PARAM_GAP + y_offset
        X, Y = np.meshgrid(xpos, ypos)
        all_X.append(X.ravel())
        all_Y.append(Y.ravel())
        all_Z.append(Z.ravel())
        all_labels.extend([f'{scenario} | {l}' for l in labels])
        y_offset = ypos[-1] + SCENARIO_GAP
    
    X_flat = np.concatenate(all_X)
    Y_flat = np.concatenate(all_Y)
    Z_flat = np.concatenate(all_Z)
    
    fig = plt.figure(figsize=(16, max(10, 4 + 0.8 * len(all_labels))))
    ax = fig.add_subplot(111, projection='3d')
    
    dz = Z_flat
    colors = cmap_name(norm(np.where(np.isnan(dz), z_min, dz)))
    ax.bar3d(X_flat, Y_flat, np.zeros_like(dz),
             BAR, BAR, np.nan_to_num(dz, 0),
             color=colors, alpha=0.85, edgecolor='#333', linewidth=0.15)
    
    ax.set_xlabel('Perturbation Step', fontsize=12, labelpad=12)
    ax.set_ylabel('Scenario | Parameter', fontsize=12, labelpad=12)
    ax.set_zlabel('Ton NH3 Cost (yuan/t)', fontsize=12, labelpad=12)
    ax.set_title('Q3 Sensitivity — All Scenarios Combined', fontsize=14, fontweight='bold', pad=18)
    
    ax.set_xticks(np.arange(max_pts_overall) + BAR / 2)
    ax.set_xticklabels([f'{i+1}' for i in range(max_pts_overall)], fontsize=9)
    
    # Y ticks at center of each parameter row
    y_tick_positions = []
    y_off = 0.0
    for scenario, _, _, n_params, _ in scenario_data:
        for pi in range(n_params):
            y_tick_positions.append(y_off + pi * PARAM_GAP + BAR / 2)
        y_off += n_params * PARAM_GAP + SCENARIO_GAP
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(all_labels, fontsize=7)
    
    ax.set_box_aspect([max_pts_overall, len(all_labels), max_pts_overall * 0.5])
    ax.view_init(elev=28, azim=315)
    
    sm = cm.ScalarMappable(cmap=cmap_name, norm=norm); sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.75, aspect=22, pad=0.08)
    cbar.set_label('Ton NH3 Cost (yuan/t)', fontsize=11)
    
    info = f'3 scenarios | {len(params)} params | global cost {z_min:.0f}–{z_max:.0f} yuan/t'
    ax.text2D(0.01, 0.97, info, transform=ax.transAxes, fontsize=9,
              bbox=dict(boxstyle='round,pad=0.35', facecolor='#1a1a2e',
                        edgecolor='#0f3460', alpha=0.82), color='white')
    
    plt.tight_layout()
    outpath = os.path.join(FIG_DIR, 'q3_green_heatmap.png')
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print("  [OK] q3_green_heatmap.png (combined 3 scenarios)")

def plot_q4_storage():
    """Q4储能参数敏感性"""
    try:
        df = load_q4()
    except:
        return

    part_b = df[df['section'] == 'part_b'].copy()
    if part_b.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Storage cost vs capacity
    cost_data = part_b[part_b['param_name'] == 'storage_investment']
    if not cost_data.empty:
        axes[0].plot(cost_data['perturbed_value'], cost_data['storage_capacity_mwh'],
                     'o-', color='#3498db', linewidth=2)
        axes[0].set_xlabel('储能投资成本 (¥/kWh)')
        axes[0].set_ylabel('最优储能容量 (MWh)')
        axes[0].set_title('储能投资成本敏感性')
        axes[0].grid(True, alpha=0.3)

    # Storage cost vs daily depreciation
    if not cost_data.empty:
        axes[1].plot(cost_data['perturbed_value'], cost_data['daily_depreciation_yuan'],
                     's-', color='#e74c3c', linewidth=2)
        axes[1].set_xlabel('储能投资成本 (¥/kWh)')
        axes[1].set_ylabel('日折旧费 (¥/天)')
        axes[1].set_title('储能日折旧敏感性')
        axes[1].grid(True, alpha=0.3)

    fig.suptitle('Q4 储能参数敏感性分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'q4_storage_curves.png'), dpi=150)
    plt.close()
    print("  [OK] q4_storage_curves.png")


if __name__ == '__main__':
    print("=" * 50)
    print("生成敏感性分析图表...")
    print("=" * 50)

    plot_tornado()
    plot_q1_curves()
    plot_q2_curves()
    plot_q3_curves()
    plot_q4_capacity()
    plot_importance()
    plot_green_heatmap_q3()
    plot_q4_storage()

    print("\n图表生成完毕。")
