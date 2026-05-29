"""Q5 可视化: 参数α (售电权重) 敏感性分析
=====================================
生成两张图表 (全部中文标签):
  1. 敏感性曲线 (q5_sensitivity_curves.png)  — 3 子图: ton_cost / eta_self / (eta_green + eta_grid) vs α
  2. 政策对比柱状图 (q5_policy_comparison.png) — 保守策略 vs 激进策略 五指标并排

运行:
    cd A题/code_solution_v2
    python visualization/charts_q5.py

数据来源: results/q5_sensitivity.csv (31 行, utf-8-sig)

α 参数含义:
    α 放大了售电收入在目标函数中的权重。α 越大 → 求解器越倾向"多售电" →
    η_self 下降 (自发自用率降低) / η_grid 上升 (上网比例增大) → 吨氨成本上升。
    当 α ≥ 2.5 时 η_self 变负，表明系统过度追求售电收益。
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.plotting import (
    create_figure, save_figure, get_color_scheme,
    style_axis, apply_paper_style,
)

# ── 结果目录 ─────────────────────────────────────────────────────────────────
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJ_ROOT, 'results')
CSV_PATH = os.path.join(RESULTS_DIR, 'q5_sensitivity.csv')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Data loading                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_q5_data():
    """加载 Q5 敏感性 CSV (utf-8-sig, 31 行, α=1.0~4.0, 全部 Optimal)。"""
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
    for col in ['alpha', 'ton_cost', 'eta_self', 'eta_green', 'eta_grid',
                'operating_cost']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Figure 1: 敏感性曲线 — 3 子图纵向排列                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_sensitivity_curves(df):
    """3 子图垂直堆叠: 吨氨成本 / η_self / (η_green + η_grid) 随 α 变化曲线。

    特性:
        - 共享 X 轴 (α)
        - 标注关键转折点: α=1.5, 1.7, 2.2, 2.5, 2.7
        - η_self 子图标注零穿越水平线
        - 底部子图双曲线 (η_green + η_grid, 含图例)
        - 各阶段阴影背景

    Returns:
        str: 保存的 PNG 路径
    """
    alpha = df['alpha'].values
    ton_cost = df['ton_cost'].values
    eta_self = df['eta_self'].values
    eta_green = df['eta_green'].values
    eta_grid = df['eta_grid'].values

    fig, axes = plt.subplots(3, 1, figsize=(13, 17), sharex=True)
    ax1, ax2, ax3 = axes

    # ── 配色 ──────────────────────────────────────────────────────────
    c_cost  = '#D32F2F'   # 吨氨成本 — 红
    c_self  = '#1565C0'   # η_self   — 蓝
    c_green = '#2E7D32'   # η_green  — 绿
    c_grid  = '#E65100'   # η_grid   — 橙
    c_annot = '#6A1B9A'   # 标注色  — 紫

    # ═══ 子图 (a): 吨氨成本 vs α ═══════════════════════════════════════════
    ax1.plot(alpha, ton_cost, '-o', color=c_cost, linewidth=2.5,
             markersize=7, markerfacecolor='white', markeredgewidth=1.8,
             markeredgecolor=c_cost, zorder=4)
    ax1.fill_between(alpha, ton_cost, 0, alpha=0.06, color=c_cost, zorder=1)

    # 阶段阴影 (稳定区 → 跳变区 → 饱和区)
    stages = [
        (1.0, 1.4, '#E8F5E9', '稳定区\n(4550)'),
        (1.5, 1.6, '#FFFDE7', '第一跳变\n(4656)'),
        (1.7, 1.8, '#FFEBEE', '大幅跳变\n(5063)'),
        (1.9, 2.1, '#F3E5F5', '过渡区\n(5087)'),
        (2.2, 2.2, '#E3F2FD', ''),
        (2.3, 2.4, '#FFF8E1', '持续上升\n(5395)'),
        (2.5, 2.6, '#FCE4EC', '零穿越区\n(5888)'),
        (2.7, 4.0, '#ECEFF1', '饱和区\n(5995)'),
    ]
    for x0, x1, bg, lbl in stages:
        ax1.axvspan(x0, x1, alpha=0.30, color=bg, zorder=0)
        if lbl and x1 - x0 > 0.25:
            ax1.text((x0 + x1) / 2, ton_cost[0] - 50, lbl,
                     ha='center', va='top', fontsize=7.5,
                     color='#757575', style='italic')

    # 关键转折点标注
    key_pts = [
        (1.5, 250, 0.15),
        (1.7, 200, 0.18),
        (2.2, 220, 0.15),
        (2.5, 180, 0.12),
        (2.7, 80,  0.10),
    ]
    for aa, yoff, xoff in key_pts:
        idx = np.where(alpha == aa)[0][0]
        tc = ton_cost[idx]
        ax1.annotate(
            f'α={aa:.1f}\n{tc:.0f} 元/吨',
            xy=(aa, tc), xytext=(aa + xoff, tc + yoff),
            arrowprops=dict(arrowstyle='->', color='#757575', lw=1.3),
            fontsize=8.5, fontweight='bold', color=c_cost,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      edgecolor=c_cost, alpha=0.92),
            zorder=10,
        )

    ax1.set_ylabel('吨氨成本 (元/吨)', fontsize=13, color=c_cost)
    ax1.tick_params(axis='y', labelcolor=c_cost)
    ax1.grid(alpha=0.22, zorder=2)
    ax1.set_title('(a) 吨氨成本随 α 变化', fontsize=14, fontweight='bold',
                  loc='left', color='#333333')

    # ═══ 子图 (b): η_self vs α ══════════════════════════════════════════════
    ax2.plot(alpha, eta_self, '-s', color=c_self, linewidth=2.5,
             markersize=7, markerfacecolor='white', markeredgewidth=1.8,
             markeredgecolor=c_self, zorder=4)
    # 正/负区域填充
    ax2.fill_between(alpha, 0, eta_self, where=(eta_self >= 0),
                     alpha=0.06, color=c_self, interpolate=True, zorder=1)
    ax2.fill_between(alpha, 0, eta_self, where=(eta_self < 0),
                     alpha=0.10, color='#D32F2F', interpolate=True,
                     label='η_self < 0 (变负)', zorder=1)

    # 零值参考线
    ax2.axhline(y=0, color='#D32F2F', linestyle='--', linewidth=1.6,
                alpha=0.70, zorder=2)

    # 初始值标注
    ax2.annotate(
        f'η_self = {eta_self[0]:.3f}',
        xy=(1.0, eta_self[0]),
        xytext=(1.15, eta_self[0] + 0.09),
        arrowprops=dict(arrowstyle='->', color='#757575', lw=1.2),
        fontsize=9.5, fontweight='bold', color=c_self,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
        zorder=10,
    )

    # 最终值标注
    ax2.annotate(
        f'η_self = {eta_self[-1]:.3f}',
        xy=(4.0, eta_self[-1]),
        xytext=(3.2, eta_self[-1] - 0.15),
        arrowprops=dict(arrowstyle='->', color='#757575', lw=1.2),
        fontsize=9.5, fontweight='bold', color='#D32F2F',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9),
        zorder=10,
    )

    # 零穿越标注
    ax2.annotate(
        'η_self 零穿越\n(α≈2.3 → 2.5)\n负值: 过度售电',
        xy=(2.5, -0.060),
        xytext=(3.3, 0.22),
        arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5),
        fontsize=9.5, fontweight='bold', color='#D32F2F',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE',
                  edgecolor='#D32F2F', alpha=0.90),
        zorder=10,
    )

    ax2.set_ylabel('η_self (自发自用率)', fontsize=13, color=c_self)
    ax2.tick_params(axis='y', labelcolor=c_self)
    ax2.grid(alpha=0.22, zorder=2)
    ax2.set_title('(b) 自发自用率 η_self 随 α 变化', fontsize=14,
                  fontweight='bold', loc='left', color='#333333')
    ax2.legend(loc='lower left', fontsize=10, framealpha=0.9)

    # ═══ 子图 (c): η_green + η_grid vs α (双曲线) ═══════════════════════════
    ax3.plot(alpha, eta_green, '-D', color=c_green, linewidth=2.5,
             markersize=7, markerfacecolor='white', markeredgewidth=1.8,
             markeredgecolor=c_green, label='η_green (绿电比例)', zorder=4)
    ax3.fill_between(alpha, eta_green, 0, alpha=0.05, color=c_green, zorder=1)

    ax3.plot(alpha, eta_grid, '-^', color=c_grid, linewidth=2.5,
             markersize=7, markerfacecolor='white', markeredgewidth=1.8,
             markeredgecolor=c_grid, label='η_grid (上网比例)', zorder=4)
    ax3.fill_between(alpha, eta_grid, 0, alpha=0.05, color=c_grid, zorder=1)

    # 标注单调趋势
    ax3.annotate(
        f'η_green↓: {eta_green[0]:.3f}→{eta_green[-1]:.3f}',
        xy=(2.5, 0.38), fontsize=9, color=c_green, fontweight='bold',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85),
        zorder=10,
    )
    ax3.annotate(
        f'η_grid↑: {eta_grid[0]:.3f}→{eta_grid[-1]:.3f}',
        xy=(2.5, 0.48), fontsize=9, color=c_grid, fontweight='bold',
        ha='center',
        bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.85),
        zorder=10,
    )

    # 两线交汇点
    cross_idx = np.argmin(np.abs(eta_green - eta_grid))
    ax3.annotate(
        f'交汇 α≈{alpha[cross_idx]:.1f}',
        xy=(alpha[cross_idx], (eta_green[cross_idx] + eta_grid[cross_idx]) / 2),
        xytext=(alpha[cross_idx] + 0.5, (eta_green[cross_idx] + eta_grid[cross_idx]) / 2 - 0.05),
        arrowprops=dict(arrowstyle='->', color='#757575', lw=1.3),
        fontsize=9, fontweight='bold', color='#555555',
        bbox=dict(boxstyle='round,pad=0.35', facecolor='#FFF8E1',
                  edgecolor='#FFB300', alpha=0.92),
        zorder=10,
    )

    ax3.set_ylabel('绿电指标', fontsize=13)
    ax3.set_xlabel('α (售电收入权重系数)', fontsize=13)
    ax3.grid(alpha=0.22, zorder=2)
    ax3.set_title('(c) 绿电指标 η_green / η_grid 随 α 变化', fontsize=14,
                  fontweight='bold', loc='left', color='#333333')
    ax3.legend(loc='center right', fontsize=10.5, framealpha=0.92)

    # ── 共享 X 轴 ──────────────────────────────────────────────────────
    ax3.set_xlim(0.85, 4.15)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=11)

    # ── 全局标题 ──────────────────────────────────────────────────────
    fig.suptitle('Q5 参数α (售电权重) 敏感性分析',
                 fontsize=18, fontweight='bold', y=1.007)

    apply_paper_style(fig)
    fig.tight_layout()
    out_path = save_figure(fig, 'q5_sensitivity_curves.png',
                           results_dir=RESULTS_DIR, dpi=200)
    print(f"  [敏感性曲线] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Figure 2: 政策对比柱状图 — 5 指标分组对比                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_policy_comparison(df):
    """分组柱状图: α=1 (保守策略) vs α=4 (激进策略)。

    5 组指标 (吨氨成本用左轴, 其余用右轴):
        1. 吨氨成本 (元/吨)
        2. η_self (自发自用率)
        3. η_green (绿电比例)
        4. η_grid (上网比例)
        5. 售电量 (MWh) — 由 η_grid × E_RE 估算, E_RE 为常数取相对值

    Returns:
        str: 保存的 PNG 路径
    """
    row_1 = df[df['alpha'] == 1.0].iloc[0]
    row_4 = df[df['alpha'] == 4.0].iloc[0]

    # ── 指标索引 ──────────────────────────────────────────────────────
    metric_labels = [
        '吨氨成本\n(元/吨)',
        'η_self',
        'η_green',
        'η_grid',
        '售电量\n(MWh)',
    ]

    # 保守策略 (α=1)
    vals_c = [
        row_1['ton_cost'],    # 4550.41
        row_1['eta_self'],    # 0.723
        row_1['eta_green'],   # 0.614
        row_1['eta_grid'],    # 0.139
        row_1['eta_grid'],    # 售电比例 = η_grid (E_RE 常数)
    ]
    # 激进策略 (α=4)
    vals_a = [
        row_4['ton_cost'],    # 5995.22
        row_4['eta_self'],    # -0.092
        row_4['eta_green'],   # 0.321
        row_4['eta_grid'],    # 0.546
        row_4['eta_grid'],    # 售电比例 = η_grid
    ]

    n_metrics = len(metric_labels)
    x = np.arange(n_metrics)
    bar_w = 0.32

    # ── 配色 ──────────────────────────────────────────────────────────
    c_con = '#1565C0'   # 保守 — 蓝
    c_agg = '#F57C00'   # 激进 — 橙

    fig, ax1 = plt.subplots(figsize=(15, 8))

    # ═══ 绘制柱状图 ══════════════════════════════════════════════════════
    bars_con = ax1.bar(x - bar_w / 2, vals_c, bar_w,
                       color=c_con, alpha=0.88,
                       edgecolor='white', linewidth=0.8,
                       label='保守策略 (α=1)', zorder=3)
    bars_agg = ax1.bar(x + bar_w / 2, vals_a, bar_w,
                       color=c_agg, alpha=0.88,
                       edgecolor='white', linewidth=0.8,
                       label='激进策略 (α=4)', zorder=3)

    # ── 右轴 (仅 η 指标参考) ──────────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.set_ylim(ax1.get_ylim())

    # 吨氨成本区域 (左轴主导)
    ax1.axvspan(-0.6, 0.6, alpha=0.06, color='#F3E5F5', zorder=0)
    # η 指标区域
    for i in range(1, n_metrics):
        ax1.axvspan(i - 0.6, i + 0.6, alpha=0.04, color='#E3F2FD', zorder=0)

    # ── 数值标注 ──────────────────────────────────────────────────────
    for i in range(n_metrics):
        vc, va = vals_c[i], vals_a[i]
        is_cost = (i == 0)

        # 保守策略值
        if is_cost:
            ax1.text(x[i] - bar_w / 2, vc - vc * 0.08,
                     f'{vc:.0f}', ha='center', va='top',
                     fontsize=10, color='white', fontweight='bold')
        elif vc >= 0:
            ax1.text(x[i] - bar_w / 2, vc + 0.02,
                     f'{vc:.3f}', ha='center', va='bottom',
                     fontsize=9.5, color=c_con, fontweight='bold')
        else:
            ax1.text(x[i] - bar_w / 2, vc - 0.03,
                     f'{vc:.3f}', ha='center', va='top',
                     fontsize=9.5, color=c_con, fontweight='bold')

        # 激进策略值
        if is_cost:
            ax1.text(x[i] + bar_w / 2, va - va * 0.08,
                     f'{va:.0f}', ha='center', va='top',
                     fontsize=10, color='white', fontweight='bold')
        elif va >= 0:
            ax1.text(x[i] + bar_w / 2, va + 0.02,
                     f'{va:.3f}', ha='center', va='bottom',
                     fontsize=9.5, color=c_agg, fontweight='bold')
        else:
            ax1.text(x[i] + bar_w / 2, va - 0.03,
                     f'{va:.3f}', ha='center', va='top',
                     fontsize=9.5, color=c_agg, fontweight='bold')

        # 差值标注
        diff = va - vc
        if is_cost:
            diff_txt = f'Δ={diff:+.0f}'
        else:
            diff_txt = f'Δ={diff:+.3f}'
        diff_clr = '#D32F2F' if abs(diff) > 0.08 else '#757575'
        y_max = max(vc, va)
        ax1.annotate(
            diff_txt,
            xy=(x[i], y_max),
            xytext=(x[i], y_max + 0.07),
            ha='center', fontsize=8.5, fontweight='bold',
            color=diff_clr,
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                      alpha=0.85, edgecolor=diff_clr, linewidth=0.7),
            zorder=10,
        )

    # ── 负值警告 ──────────────────────────────────────────────────────
    ax1.axhline(y=0, color='#BDBDBD', linestyle='-', linewidth=0.8, zorder=1)
    ax1.annotate(
        'η_self < 0\n自发自用率跌破零!',
        xy=(1, -0.092),
        xytext=(2.0, -0.28),
        arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5),
        fontsize=10, fontweight='bold', color='#D32F2F',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE',
                  edgecolor='#D32F2F', alpha=0.92),
        zorder=10,
    )

    # ── 轴与刻度 ──────────────────────────────────────────────────────
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, fontsize=11.5)
    ax1.set_ylabel('指标值', fontsize=12)
    ax1.grid(axis='y', alpha=0.18, zorder=1)

    y_min = min(min(vals_c), min(vals_a))
    y_max_abs = max(abs(v) for v in vals_c + vals_a)
    ax1.set_ylim(y_min - 0.40, y_max_abs * 1.25)
    ax2.set_ylim(ax1.get_ylim())
    ax2.set_ylabel('η 指标 / 比例', fontsize=11, color='#9E9E9E')
    ax2.tick_params(axis='y', labelcolor='#9E9E9E')

    # ── 图例 ──────────────────────────────────────────────────────────
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, facecolor=c_con, alpha=0.88,
                      edgecolor='white', linewidth=0.8,
                      label='保守策略 (α=1)'),
        plt.Rectangle((0, 0), 1, 1, facecolor=c_agg, alpha=0.88,
                      edgecolor='white', linewidth=0.8,
                      label='激进策略 (α=4)'),
    ]
    ax1.legend(handles=legend_patches, loc='upper right',
               framealpha=0.92, fontsize=11.5, title='售电权重策略',
               title_fontsize=12)

    # ── 标题 ──────────────────────────────────────────────────────────
    ax1.set_title('Q5 政策对比：保守策略 (α=1) vs 激进策略 (α=4)',
                  fontsize=17, fontweight='bold', pad=20)

    # 底部注释
    fig.text(0.5, 0.008,
             '注: α 放大售电收入权重; 激进策略 (α=4) 导致吨氨成本上升 31.7%, η_self 变负, 绿电本地消纳能力大幅减弱',
             ha='center', fontsize=9.5, color='#757575', style='italic')

    apply_paper_style(fig)
    fig.tight_layout(rect=[0, 0.035, 1, 0.98])
    out_path = save_figure(fig, 'q5_policy_comparison.png',
                           results_dir=RESULTS_DIR, dpi=200)
    print(f"  [政策对比图] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    """加载 Q5 敏感性数据, 生成 2 张图表, 输出中文政策建议摘要。"""
    print("=" * 60)
    print("Q5 可视化 — 参数 α (售电权重) 敏感性分析")
    print("=" * 60)

    # ── 1. 加载数据 ──────────────────────────────────────────────────
    print("\n[1/3] 加载 Q5 敏感性数据...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Q5 敏感性数据文件不存在: {CSV_PATH}\n"
            f"请先运行 models/q5_sensitivity.py 生成结果。"
        )

    df = load_q5_data()
    print(f"  数据行数: {len(df)} (α = {df['alpha'].min():.1f} ~ {df['alpha'].max():.1f})")
    print(f"  状态: 全部 {df['status'].iloc[0]}")
    print(f"  吨氨成本范围: {df['ton_cost'].min():.0f} ~ {df['ton_cost'].max():.0f} 元/吨")

    # ── 2. 提取关键行 ────────────────────────────────────────────────
    row_1 = df[df['alpha'] == 1.0].iloc[0]
    row_15 = df[df['alpha'] == 1.5].iloc[0]
    row_4 = df[df['alpha'] == 4.0].iloc[0]

    # ── 3. Figure 1: 敏感性曲线 ──────────────────────────────────────
    print("\n[2/3] 生成敏感性曲线 (3 子图)...")
    curves_path = make_sensitivity_curves(df)

    # ── 4. Figure 2: 政策对比柱状图 ──────────────────────────────────
    print("\n[3/3] 生成政策对比柱状图 (α=1 vs α=4)...")
    compare_path = make_policy_comparison(df)

    # ── 5. 验证输出 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("输出文件大小验证:")
    print("=" * 60)
    all_ok = True
    for label, path in [
        ('敏感性曲线 (3子图)', curves_path),
        ('政策对比柱状图',     compare_path),
    ]:
        if path and os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            ok = size_kb > 20
            marker = 'OK' if ok else 'WARN'
            print(f"  [{marker}] {label}: {os.path.basename(path)}")
            print(f"         size: {size_kb:.1f} KB {'(>20KB)' if ok else '(<20KB!)'}")
            if not ok:
                all_ok = False
        else:
            print(f"  [FAIL] {label}: 文件未找到!")
            all_ok = False

    # ── 6. 中文政策建议摘要 ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("========== Q5 政策建议摘要 ==========")
    print("=" * 60)

    d_ton = row_4['ton_cost'] - row_1['ton_cost']
    d_self = row_4['eta_self'] - row_1['eta_self']
    d_green = row_4['eta_green'] - row_1['eta_green']
    d_grid = row_4['eta_grid'] - row_1['eta_grid']

    print(f"""
  α 参数的经济含义:
    α 是售电收入在目标函数中的权重系数。增大 α 意味着放大售电收益,
    使优化模型更倾向于"多售电、少自用"。随着 α 从 1.0 增至 4.0,
    售电行为的优先级不断提高, 导致自发自用率持续下降。

  关键转折点分析:
    --> α=1.0 (基准策略): 吨氨成本 {row_1['ton_cost']:.0f} 元/吨, η_self={row_1['eta_self']:.3f}, η_green={row_1['eta_green']:.3f}, η_grid={row_1['eta_grid']:.3f}
    --> α=1.5 (第一跳变): η_self 从 0.723 骤降至 0.615, η_grid 从 0.139 跃升至 0.193, 吨氨成本增加 {row_15['ton_cost'] - row_1['ton_cost']:.0f} 元
    --> α=1.7 (大幅跳变): η_self 暴跌至 0.279, η_grid 翻倍至 0.360, 吨氨成本跳增约 400 元/吨 (→5063)
    --> α=2.5 (零穿越点): η_self 首次变负 (-0.060), 自发自用率跌破零!
    --> α=4.0 (激进策略): 吨氨成本 {row_4['ton_cost']:.0f} 元/吨, η_self={row_4['eta_self']:.3f}, η_green={row_4['eta_green']:.3f}, η_grid={row_4['eta_grid']:.3f}

  绿电指标变化趋势:
    --> η_self 单调下降: {row_1['eta_self']:.3f} → {row_4['eta_self']:.3f} (降幅 {abs(d_self):.1%})
    --> η_green 单调下降: {row_1['eta_green']:.3f} → {row_4['eta_green']:.3f} (降幅 {abs(d_green):.1%})
    --> η_grid 单调上升:  {row_1['eta_grid']:.3f} → {row_4['eta_grid']:.3f} (增幅 {abs(d_grid):.1%})
    --> 三者呈"跷跷板"关系: 多售电 → 少自用 → 绿电利用率下降 → 吨氨成本上升

  政策建议:
    [+] 推荐 α 控制在 [1.0, 1.5] 区间内, 可兼顾经济性与绿色指标:
      - 吨氨成本: 4550~4656 元/吨 (经济性最优)
      - η_self >= 0.6 (满足自发自用率要求)
      - η_green >= 0.58 (绿电比例保持高位)
    [+] 若政策强制 η_self >= 0.6, 则 α 不得高于 1.4
    [+] 若允许 η_self >= 0.3, α 可放宽至 1.6, 但吨氨成本上升约 10%
    [+] 当 α >= 2.5 时 η_self 变负, 系统过度追求售电收益, 不建议采用
    [+] 从绿色低碳角度出发, 保守策略 (α=1) 的绿电自用效率最高,
      可在保证经济效益的同时最大化绿电就地消纳

  总结:
    适度提高 α 可激励新能源上网, 但过度激励会导致自发自用率急剧下降与
    绿电本地消纳能力减弱, 吨氨成本大幅上升 (+31.7%)。建议政策制定者在
    1.0 <= α <= 1.5 范围内寻找最优平衡点, 实现经济性与绿色性的协同优化。
    α=1 (保守策略) 综合表现最佳: 吨氨成本 4550 元/吨, η_self=0.723, η_green=0.614。
""")
    print("=====================================")

    if all_ok:
        print("\nQ5 可视化全部完成 — 2 张图表已生成, 政策建议已输出。")
    else:
        print("\n部分图表验证未通过 — 请检查上方 WARN 项。")

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
