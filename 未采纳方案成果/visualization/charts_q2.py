"""Q2 可视化: 离散制氨调节优化
=====================================
生成三张图表 (全部中文标签):
  1. Gantt chart (q2_gantt.png)              — 24 场景设备启停甘特图
  2. Boxplot (q2_boxplot.png)                — 4 面板: 日购电量/日售电量/吨氨成本/绿色指标
  3. Cost distribution (q2_cost_distribution.png) — PDF + CDF 吨氨成本分布

运行:
    cd A题/code_solution_v2
    python visualization/charts_q2.py

数据来源: results/q2_results.csv (120 行, utf-8-sig)
  - 仅 D_target=36 有 24 可行解 (行 98-121); 其他目标全部不可行
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.stats import gaussian_kde

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization.plotting import (
    create_figure, save_figure, get_color_scheme,
    style_axis, apply_paper_style,
)

# ── 项目根路径 ───────────────────────────────────────────────────────────────
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJ_ROOT, 'results')
CSV_PATH = os.path.join(RESULTS_DIR, 'q2_results.csv')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Data loading helpers                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_q2_data():
    """Load Q2 results CSV and split into feasible / infeasible subsets.

    Returns:
        tuple: (df_all, df_feasible, df_infeasible, targets)
    """
    df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')

    # Convert numeric columns (empty strings → NaN)
    numeric_cols = [
        '运行成本(元)', '吨氨成本(元/吨)',
        '日购电量(MWh)', '日售电量(MWh)',
        'ALK总功率(MW)', 'PEM总功率(MW)', 'NH3总功率(MW)',
        '新能源自发自用电量占比(%)', '总用电量绿电比例(%)', '新能源上网电量比例(%)',
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Split by status
    df_feasible = df[df['状态'] == '最优'].copy()
    df_infeasible = df[df['状态'] != '最优'].copy()

    targets = sorted(df['日产量目标(吨)'].unique(), reverse=True)

    return df, df_feasible, df_infeasible, targets


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 1: Gantt chart — 24 场景设备启停                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_gantt(df_feasible, colors):
    """甘特图: 24 个可行场景 (D=36吨/日) 的 ALK/PEM/NH3 启停状态。

    每场景 1 行 (y轴 = 场景标签)，内部 3 条水平色条 (ALK/PEM/NH3)。
    横轴 24 小时。ALK总功率=240MW (10MW×24h) → 全部小时 ON。
    """
    n_scenarios = len(df_feasible)  # 24

    # ── Device rated powers (单机) ──────────────────────────────────────
    ALK_RATED = 10.0   # MW
    PEM_RATED = 10.0   # MW
    NH3_RATED = 0.75   # MW
    HOURS = 24

    fig, ax = create_figure(figsize=(22, 14))

    # ── Color mapping ────────────────────────────────────────────────────
    device_colors = {
        'ALK': colors.get('d63', '#1565C0'),        # deep blue
        'PEM': colors.get('box_fill', '#42A5F5'),   # medium blue
        'NH3': colors.get('accent', '#FF7043'),     # orange-red
    }
    device_names_cn = {
        'ALK': 'ALK 碱性电解槽',
        'PEM': 'PEM 质子交换膜',
        'NH3': '合成氨装置',
    }

    bar_height = 0.20
    row_spacing = 1.0
    y_tick_positions = []
    y_tick_labels = []

    for idx, (_, row) in enumerate(df_feasible.iterrows()):
        sid = int(row['场景编号'])
        w = int(row['风电场景'])
        pv = int(row['光伏场景'])
        label = f"W{w}P{pv} (#{sid:02d})"

        y_center = idx * row_spacing

        # Derive hourly on/off from daily totals
        alk_tot = row['ALK总功率(MW)']
        pem_tot = row['PEM总功率(MW)']
        nh3_tot = row['NH3总功率(MW)']

        alk_h = int(alk_tot / ALK_RATED) if pd.notna(alk_tot) else 0
        pem_h = int(pem_tot / PEM_RATED) if pd.notna(pem_tot) else 0
        nh3_h = int(nh3_tot / NH3_RATED) if pd.notna(nh3_tot) else 0

        # Draw 3 horizontal bars per scenario (offset vertically)
        for dev, on_h, y_off in [
            ('ALK', alk_h, +0.30),
            ('PEM', pem_h,  0.00),
            ('NH3', nh3_h, -0.30),
        ]:
            if on_h > 0:
                rect = Rectangle(
                    (0, y_center + y_off - bar_height / 2),
                    on_h, bar_height,
                    facecolor=device_colors[dev],
                    edgecolor='white',
                    linewidth=0.4,
                    alpha=0.90,
                )
                ax.add_patch(rect)
                ax.text(
                    on_h + 0.35, y_center + y_off,
                    f'{on_h}h',
                    va='center', fontsize=6.5,
                    color=device_colors[dev],
                    fontweight='bold',
                )

        y_tick_positions.append(y_center)
        y_tick_labels.append(label)

    # ── Axes / grid ──────────────────────────────────────────────────────
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels(y_tick_labels, fontsize=7.5)
    ax.set_xlim(-0.8, 26.5)
    ax.set_ylim(-1.2, n_scenarios * row_spacing - 0.5)
    ax.set_xlabel('小时 (h)', fontsize=12)
    ax.set_ylabel('风光场景组合', fontsize=12)
    ax.set_title('Q2 离散调度甘特图 (D=36吨/日)', fontsize=16, fontweight='bold')

    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xticklabels([f'{h}:00' for h in range(0, 25, 2)])
    ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis='x', which='major', alpha=0.35, linewidth=0.7)
    ax.grid(axis='x', which='minor', alpha=0.12, linewidth=0.3)

    ax.invert_yaxis()

    # ── Legend ───────────────────────────────────────────────────────────
    legend_patches = [
        Rectangle((0, 0), 1, 1, facecolor=device_colors[k],
                  edgecolor='white', linewidth=0.4,
                  label=device_names_cn[k])
        for k in ['ALK', 'PEM', 'NH3']
    ]
    ax.legend(handles=legend_patches, loc='upper right',
              framealpha=0.92, fontsize=10,
              title='设备', title_fontsize=11)

    # ── Annotation ───────────────────────────────────────────────────────
    ax.text(
        26.8, n_scenarios * row_spacing * 0.97,
        '注: 24 场景设备\n均全时段 24h 运行',
        fontsize=10, color=colors.get('primary', '#1565C0'),
        ha='left', va='top',
        bbox=dict(boxstyle='round,pad=0.55',
                  facecolor='#E3F2FD',
                  edgecolor=colors.get('primary', '#1565C0'),
                  alpha=0.85),
    )

    apply_paper_style(fig)
    fig.tight_layout()
    out_path = save_figure(fig, 'q2_gantt.png',
                           results_dir=RESULTS_DIR, dpi=300)
    print(f"  [甘特图] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 2: Boxplot — 4 面板对比 (日购电量 + 日售电量 + 吨氨成本 + 绿色) ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _collect_boxplot_data(df, targets, col_name):
    """Collect boxplot data arrays per D_target. Returns list of arrays."""
    result = []
    for d in targets:
        sub = df[df['日产量目标(吨)'] == d][col_name].dropna()
        result.append(sub.values if len(sub) > 0 else np.array([]))
    return result


def _annotate_infeasible(ax, x_pos, text='不可行'):
    """Place '不可行' annotation above an empty boxplot position."""
    ax.annotate(
        text, xy=(x_pos, 0.5), xycoords=('data', 'axes fraction'),
        fontsize=12, color='#D32F2F', fontweight='bold',
        ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE',
                  edgecolor='#D32F2F', alpha=0.90),
    )


def make_boxplot(df, df_feasible, targets, colors):
    """4 面板箱线图: 日购电量 / 日售电量 / 吨氨成本 / 绿色指标。

    对比 5 种日产量目标 (72/63/54/45/36 吨/日)。
    不可行目标显示 '不可行' 标注。
    """
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    ax1, ax2, ax3, ax4 = axes.flatten()

    target_labels = [f'{int(d)}吨/日' for d in targets]
    x_positions = np.arange(1, len(targets) + 1)

    # ── Common boxplot style ──────────────────────────────────────────────
    box_fill = colors.get('box_fill', '#42A5F5')
    box_edge = colors.get('box_edge', '#1565C0')
    flier_clr = colors.get('flier', '#FF7043')

    def _draw_boxplot(ax, data_list, label, title, subplot_label, fill_clr=None):
        """Draw a single boxplot subpanel with infeasible annotations."""
        fill_clr = fill_clr or box_fill
        # Only plot non-empty groups
        non_empty = [(i, d) for i, d in enumerate(data_list) if len(d) > 0]
        if non_empty:
            pos_list = [x_positions[i] for i, _ in non_empty]
            data_to_plot = [d for _, d in non_empty]
            bp = ax.boxplot(
                data_to_plot, positions=pos_list,
                patch_artist=True, showfliers=True, widths=0.5,
            )
            for patch in bp['boxes']:
                patch.set_facecolor(fill_clr)
                patch.set_alpha(0.60)
                patch.set_edgecolor(box_edge)
                patch.set_linewidth(1.3)
            for whisker in bp['whiskers']:
                whisker.set_color(box_edge)
            for cap in bp['caps']:
                cap.set_color(box_edge)
            for flier in bp['fliers']:
                flier.set_markeredgecolor(flier_clr)
                flier.set_markerfacecolor(flier_clr)
                flier.set_alpha(0.7)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(target_labels, fontsize=9)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(subplot_label, fontsize=13, fontweight='bold', loc='left')
        ax.grid(axis='y', alpha=0.30)

        # Annotate infeasible targets
        for i, d in enumerate(data_list):
            if len(d) == 0:
                _annotate_infeasible(ax, x_positions[i])

    # ── Panel (a): 日购电量 (MWh) ────────────────────────────────────────
    buy_data = _collect_boxplot_data(df, targets, '日购电量(MWh)')
    _draw_boxplot(ax1, buy_data, '日购电量 (MWh)',
                  '日购电量分布', '(a) 日购电量分布',
                  fill_clr=colors.get('d63', '#1565C0'))

    # ── Panel (b): 日售电量 (MWh) ────────────────────────────────────────
    sell_data = _collect_boxplot_data(df, targets, '日售电量(MWh)')
    _draw_boxplot(ax2, sell_data, '日售电量 (MWh)',
                  '日售电量分布', '(b) 日售电量分布',
                  fill_clr='#66BB6A')

    # ── Panel (c): 吨氨成本 (元/吨) ──────────────────────────────────────
    ton_data = _collect_boxplot_data(df, targets, '吨氨成本(元/吨)')
    _draw_boxplot(ax3, ton_data, '吨氨成本 (元/吨)',
                  '吨氨成本分布', '(c) 吨氨成本分布',
                  fill_clr=colors.get('accent', '#FF7043'))

    # ── Panel (d): 绿色指标 (仅 D_target=36 有数据) ─────────────────────
    fdf = df_feasible
    green_indicators = {
        'η_self\n(自发自用率)': (fdf['新能源自发自用电量占比(%)'] / 100.0).dropna().values,
        'η_green\n(绿电比例)':   (fdf['总用电量绿电比例(%)'] / 100.0).dropna().values,
        'η_grid\n(上网比例)':    (fdf['新能源上网电量比例(%)'] / 100.0).dropna().values,
    }
    green_names = list(green_indicators.keys())
    green_vals = list(green_indicators.values())

    # Use only items with data
    bp4 = ax4.boxplot(
        green_vals, patch_artist=True, showfliers=True, widths=0.4,
    )
    green_local_colors = [colors.get('d72', '#0D47A1'), '#66BB6A', colors.get('accent', '#FF7043')]
    for patch, gclr in zip(bp4['boxes'], green_local_colors):
        patch.set_facecolor(gclr)
        patch.set_alpha(0.55)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1.3)
    for flier in bp4['fliers']:
        flier.set_markeredgecolor(flier_clr)

    ax4.set_xticklabels(green_names, fontsize=9)
    ax4.set_ylabel('比例', fontsize=11)
    ax4.set_title('(d) 绿色指标分布 (D=36吨/日)', fontsize=13,
                  fontweight='bold', loc='left')
    ax4.grid(axis='y', alpha=0.30)

    # Requirement thresholds
    ax4.axhline(y=0.60, color='#2E7D32', linestyle='--', alpha=0.50, linewidth=1.0)
    ax4.axhline(y=0.30, color='#2E7D32', linestyle='--', alpha=0.50, linewidth=1.0)
    ax4.text(3.45, 0.61, 'η_self≥0.6', fontsize=7.5, color='#2E7D32', va='bottom', ha='right')
    ax4.text(3.45, 0.31, 'η_green≥0.3', fontsize=7.5, color='#2E7D32', va='bottom', ha='right')

    # ── Global title ──────────────────────────────────────────────────────
    fig.suptitle('Q2 不同日产量目标对比', fontsize=16, fontweight='bold', y=1.01)

    apply_paper_style(fig)
    fig.tight_layout()
    out_path = save_figure(fig, 'q2_boxplot.png',
                           results_dir=RESULTS_DIR, dpi=300)
    print(f"  [箱线图] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 3: Cost distribution — PDF + CDF 双面板                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_cost_distribution(df_feasible, colors):
    """吨氨成本分布: 左面板 PDF (直方图 + KDE), 右面板 CDF。

    仅使用 D_target=36 的 24 个可行场景。
    """
    ton_costs = df_feasible['吨氨成本(元/吨)'].dropna().values

    if len(ton_costs) == 0:
        print("  [WARN] 无可行吨氨成本数据，跳过成本分布图")
        return None

    fig = plt.figure(figsize=(17, 7))

    primary = colors.get('primary', '#1565C0')
    secondary = colors.get('secondary', '#90CAF9')
    accent = colors.get('accent', '#FF7043')

    # ═══ Left: PDF (Histogram + KDE + rug) ════════════════════════════════
    ax1 = fig.add_subplot(1, 2, 1)

    n_bins = min(10, max(5, int(np.sqrt(len(ton_costs))) + 2))
    ax1.hist(
        ton_costs, bins=n_bins,
        color=secondary, edgecolor=primary,
        linewidth=1.2, alpha=0.65, density=True,
        label='直方图 (归一化)',
    )

    # KDE
    kde = gaussian_kde(ton_costs)
    x_kde = np.linspace(ton_costs.min() * 0.88, ton_costs.max() * 1.12, 350)
    y_kde = kde(x_kde)
    ax1.plot(x_kde, y_kde, color=accent, linewidth=2.8,
             label='KDE 密度曲线', zorder=5)

    # Rug
    ylim1 = ax1.get_ylim()
    ax1.plot(ton_costs, np.full_like(ton_costs, -0.00002 * ylim1[1]),
             '|', color=primary, markersize=9, markeredgewidth=1.5)

    # Mean / median
    mean_v = np.mean(ton_costs)
    med_v = np.median(ton_costs)
    ax1.axvline(mean_v, color=colors.get('d72', '#0D47A1'),
                linestyle='-', linewidth=1.8, alpha=0.85,
                label=f'均值: {mean_v:.0f} 元/吨')
    ax1.axvline(med_v, color=colors.get('d63', '#1565C0'),
                linestyle='--', linewidth=1.8, alpha=0.85,
                label=f'中位数: {med_v:.0f} 元/吨')

    ax1.set_xlabel('吨氨成本 (元/吨)', fontsize=11)
    ax1.set_ylabel('概率密度', fontsize=11)
    ax1.set_title('概率密度分布 (PDF)', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, framealpha=0.88)
    ax1.grid(alpha=0.22)

    # ═══ Right: CDF (经验累积分布) ═══════════════════════════════════════
    ax2 = fig.add_subplot(1, 2, 2)

    sorted_costs = np.sort(ton_costs)
    cdf_y = np.arange(1, len(sorted_costs) + 1) / len(sorted_costs)

    ax2.step(sorted_costs, cdf_y, where='post',
             color=primary, linewidth=2.5,
             label='经验累积分布 (CDF)', zorder=4)
    ax2.fill_between(sorted_costs, 0, cdf_y,
                     color=secondary, alpha=0.22, step='post')

    ax2.scatter(sorted_costs, cdf_y, color=accent,
                s=32, zorder=5, edgecolors='white', linewidth=0.4)

    # Percentile annotations
    for pct in [25, 50, 75]:
        pv = np.percentile(ton_costs, pct)
        ax2.axhline(y=pct / 100, color='#999999', linestyle=':', alpha=0.45)
        ax2.axvline(x=pv, color='#999999', linestyle=':', alpha=0.45)
        ax2.text(
            pv, pct / 100 + 0.035,
            f'P{pct}\n{pv:.0f}',
            fontsize=7.5, color='#555555',
            ha='center', va='bottom',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.65),
        )

    ax2.set_xlabel('吨氨成本 (元/吨)', fontsize=11)
    ax2.set_ylabel('累积概率', fontsize=11)
    ax2.set_title('累积分布函数 (CDF)', fontsize=13, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=9, framealpha=0.88)
    ax2.set_ylim(-0.03, 1.05)
    ax2.grid(alpha=0.22)

    # ── Global suptitle with stats ────────────────────────────────────────
    std_v = np.std(ton_costs, ddof=1)
    stats_line = (
        f"均值={mean_v:.0f}  中位数={med_v:.0f}  "
        f"标准差={std_v:.0f}  范围=[{ton_costs.min():.0f}, {ton_costs.max():.0f}] 元/吨"
    )
    fig.suptitle(
        f'Q2 吨氨成本分布 (D=36吨/日, 24场景)\n{stats_line}',
        fontsize=14, fontweight='bold', y=1.03,
    )

    apply_paper_style(fig)
    fig.tight_layout()
    out_path = save_figure(fig, 'q2_cost_distribution.png',
                           results_dir=RESULTS_DIR, dpi=300)
    print(f"  [成本分布图] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 4: Cost vs Green ratio scatter — 5 档产量着色                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def plot_q2_cost_green(df: pd.DataFrame, save_path: str):
    """
    Q2 散点图：吨氨成本 vs. 总用电量绿电比例
    按 5 档日产量着色。排除不可行行。
    """
    df_feasible = df[df['状态'] == '最优'].copy()
    if len(df_feasible) == 0:
        print("[charts_q2] 警告：无可行行，散点图为空白")
        return

    # 读取百分比值并转为小数以供散点图轴使用
    df_feasible['eta_green_frac'] = df_feasible['总用电量绿电比例(%)'] / 100.0

    fig, ax = plt.subplots(figsize=(8, 6))
    targets = sorted(df_feasible['日产量目标(吨)'].unique())

    # 5 种颜色用于 5 档产量
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    for i, d_target in enumerate(targets):
        mask = df_feasible['日产量目标(吨)'] == d_target
        color = colors[i % len(colors)]
        ax.scatter(
            df_feasible.loc[mask, '吨氨成本(元/吨)'],
            df_feasible.loc[mask, 'eta_green_frac'] * 100,  # 百分比形式展示更直观
            c=color, label=f'日产 {d_target} t', alpha=0.7, edgecolors='white', s=40
        )

    ax.set_xlabel('吨氨成本 (元/吨)')
    ax.set_ylabel('总用电量绿电比例 η_green (%)')
    ax.set_title('Q2: 吨氨成本 vs 绿电比例（按产量着色）')
    ax.legend(title='日氨产量')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=30, color='gray', linestyle='--', alpha=0.5, label='η_green=30% 门槛')

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"[charts_q2] 散点图已保存至 {save_path}")
    return save_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    """加载 Q2 结果 CSV，生成全部 3 张图表。"""
    print("=" * 60)
    print("Q2 可视化 — 生成四张图表 (甘特图 / 箱线图 / 成本分布 / 成本-绿电散点)")
    print("=" * 60)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/4] 加载 Q2 结果数据...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"Q2 结果文件不存在: {CSV_PATH}\n"
            f"请先运行 models/q2_milp.py 生成结果。"
        )

    df, df_feasible, df_infeasible, targets = load_q2_data()
    print(f"  总行数: {len(df)} (5 目标 × 24 场景)")
    print(f"  可行解: {len(df_feasible)} 行 (仅 D_target=36)")
    print(f"  不可行: {len(df_infeasible)} 行")
    print(f"  日产量目标: {[int(t) for t in targets]}")

    if len(df_feasible) == 0:
        raise RuntimeError("无可行场景数据，无法生成图表！")

    colors = get_color_scheme('Q2')

    # ── 2. Chart 1: Gantt ─────────────────────────────────────────────────
    print("\n[2/5] 生成甘特图 (24 场景设备启停)...")
    gantt_path = make_gantt(df_feasible, colors)

    # ── 3. Chart 2: Boxplot ───────────────────────────────────────────────
    print("\n[3/5] 生成箱线图 (4 面板 — 对比 5 种日产量目标)...")
    boxplot_path = make_boxplot(df, df_feasible, targets, colors)

    # ── 4. Chart 3: Cost distribution ─────────────────────────────────────
    print("\n[4/5] 生成吨氨成本分布图 (PDF + CDF)...")
    dist_path = make_cost_distribution(df_feasible, colors)

    # ── 5. Chart 4: Cost vs Green ratio scatter ───────────────────────────
    print("\n[5/5] 生成吨氨成本 vs 绿电比例散点图...")
    scatter_path = plot_q2_cost_green(df, os.path.join(RESULTS_DIR, 'q2_cost_green_scatter.png'))

    # ── 6. Verify outputs ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("输出文件大小验证:")
    print("=" * 60)
    all_ok = True
    for label, path in [
        ('甘特图 (24场景)',     gantt_path),
        ('箱线图 (4面板)',      boxplot_path),
        ('成本分布图',          dist_path),
        ('成本-绿电散点图',     scatter_path),
    ]:
        if path and os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            ok = size_kb > 15
            marker = 'OK' if ok else 'WARN'
            print(f"  [{marker}] {label}: {os.path.basename(path)}")
            print(f"         size: {size_kb:.1f} KB {'(>15KB)' if ok else '(<15KB!)'}")
            if not ok:
                all_ok = False
        else:
            print(f"  [FAIL] {label}: file not found!")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("All Q2 charts generated successfully.")
    else:
        print("Some charts failed validation - check above.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
