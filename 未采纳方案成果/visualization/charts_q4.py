"""Q4 可视化: 离网储能优化 — SOC热力图 + Q3 vs Q4对比 + 成本散点图
=====================================================================
生成三张图表 (全部中文标签):
  1. SOC 逐时热力图 (q4_soc_heatmap.png)       — 24场景 × 24小时, infeasible灰色
  2. Q3 vs Q4 双柱对比图 (q4_vs_q3_comparison.png) — 吨氨成本/eta_self/eta_green
  3. 成本散点图 (q4_cost_scatter.png)           — Q3 vs Q4 吨氨成本 + y=x 参考线

运行:
    cd A题/code_solution_v2
    python visualization/charts_q4.py
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

# ── 结果目录 (项目根下的 results/) ─────────────────────────────────────────
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJ_ROOT, 'results')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  数据加载辅助                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _has_soc_columns(df):
    """检查 DataFrame 是否包含 SOC_0~SOC_23 列。"""
    return sum(1 for c in df.columns if c.startswith('SOC_')) == 24


def _get_feasible_mask(df):
    """返回 bool 数组: True = 状态列为 Optimal。"""
    status_col = df.columns[-1]
    return (df[status_col] == 'Optimal').values


def _common_feasible_sids(df_q4, df_q3):
    """返回 Q3 和 Q4 中状态均为 Optimal 的场景编号交集 (已排序)。"""
    mask4 = _get_feasible_mask(df_q4)
    mask3 = _get_feasible_mask(df_q3)
    sids4 = set(df_q4.loc[mask4, df_q4.columns[0]].values)
    sids3 = set(df_q3.loc[mask3, df_q3.columns[0]].values)
    return sorted(int(s) for s in (sids4 & sids3))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 1: SOC 逐时热力图 — 24场景 × 24小时                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_soc_heatmap(df, colors):
    """SOC 热力图: x=24 小时, y=24 场景, 颜色=SOC 值。

    Optimal 行按 colormap 着色, infeasible 行灰色填充标注"不可行"。
    若无 SOC_* 列, 跳过并提示。
    """
    if not _has_soc_columns(df):
        print("  [WARN] CSV 中无逐时 SOC 数据 (SOC_0~SOC_23), 跳过热力图")
        return None

    feasible_mask = _get_feasible_mask(df)
    n_total = len(df)
    n_feas = int(np.sum(feasible_mask))

    # ── 提取 SOC 矩阵 ─────────────────────────────────────────────────
    soc_cols = sorted([c for c in df.columns if c.startswith('SOC_')],
                      key=lambda x: int(x.split('_')[1]))
    soc_matrix = df[soc_cols].values.astype(float)

    # ── 全部行 (含 infeasible) ────────────────────────────────────────
    y_labels = []
    for i in range(n_total):
        sid = int(df.iloc[i, 0])
        wind = int(df.iloc[i, 1])
        pv = int(df.iloc[i, 2])
        mark = ' [可行]' if feasible_mask[i] else ' [不可行]'
        y_labels.append(f'场景{sid} (风{wind}光{pv}){mark}')

    x_labels = [f'{h}:00' for h in range(24)]

    # ── 绘图 ──────────────────────────────────────────────────────────
    fig, ax = create_figure(figsize=(16, max(6, n_total * 0.42)))

    # 灰色底层 (infeasible 区域显示灰色)
    gray_bg = np.full((n_total, 24), 0.5)
    ax.imshow(gray_bg, cmap='Greys', aspect='auto', vmin=0, vmax=1,
              alpha=0.12, origin='upper')

    # SOC 数据层 — infeasible 行 mask 掉
    masked_soc = np.ma.array(soc_matrix,
                             mask=np.tile(~feasible_mask[:, None], (1, 24)))
    cmap_name = colors.get('heatmap_cmap', 'YlGn')
    soc_max = max(np.nanmax(soc_matrix), 0.05)
    im = ax.imshow(masked_soc, cmap=cmap_name, aspect='auto', origin='upper',
                   vmin=0.0, vmax=max(1.0, np.ceil(soc_max * 10) / 10))

    # ── infeasible 行标注 ─────────────────────────────────────────────
    infeas_idx = np.where(~feasible_mask)[0]
    for ri in infeas_idx:
        ax.text(11.5, ri, '不可行 (离网无解)',
                ha='center', va='center', fontsize=10,
                color='#999999', fontweight='bold', fontstyle='italic')

    # ── 数值标注 (仅 feasible 行) ──────────────────────────────────────
    feas_idx = np.where(feasible_mask)[0]
    for ri in feas_idx:
        for ci in range(24):
            val = soc_matrix[ri, ci]
            txt_clr = 'white' if val > 0.55 else 'black'
            ax.text(ci, ri, f'{val:.2f}',
                    ha='center', va='center', fontsize=6,
                    color=txt_clr, fontweight='bold')

    # ── 坐标轴 ────────────────────────────────────────────────────────
    ax.set_xticks(np.arange(24))
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(np.arange(n_total))
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('场景', fontsize=12)

    # ── colorbar ──────────────────────────────────────────────────────
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('SOC (荷电状态)', fontsize=11)

    # ── 标题 ──────────────────────────────────────────────────────────
    ax.set_title(
        f'Q4 离网储能 SOC 逐时热力图 ({n_feas}/{n_total} 场景可行)',
        fontsize=14, fontweight='bold'
    )

    apply_paper_style(fig)
    fig.tight_layout()

    out_path = save_figure(fig, 'q4_soc_heatmap.png',
                           results_dir=RESULTS_DIR, dpi=200)
    print(f"  [SOC热力图] 已保存: {out_path} ({n_feas}/{n_total} feasible)")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 2: Q3 vs Q4 双柱对比图 — 吨氨成本 / eta_self / eta_green        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_q3_vs_q4_comparison(df_q4, df_q3, colors):
    """三子图并排柱状图: 仅对比双方都 feasible 的相同场景。

    指标: 吨氨成本(元/吨) | η_self | η_green
    Q3=蓝柱, Q4=橙/绿柱。
    """
    common_sids = _common_feasible_sids(df_q4, df_q3)
    if not common_sids:
        print("  [WARN] Q3 和 Q4 无共同 feasible 场景, 跳过对比图")
        return None

    n = len(common_sids)
    sid_col4 = df_q4.columns[0]

    # ── 提取指标 (位置索引, 避免编码问题) ──────────────────────────────
    # Q4: 吨氨成本=col11, eta_self=col8, eta_green=col9
    # Q3: 吨氨成本=col4,  eta_self=col10, eta_green=col11
    ton_q4 = np.array([
        df_q4[df_q4[sid_col4] == sid].iloc[0, 11] for sid in common_sids
    ], dtype=float)
    ton_q3 = np.array([
        df_q3[df_q3[df_q3.columns[0]] == sid].iloc[0, 4] for sid in common_sids
    ], dtype=float)
    etas_q4 = np.array([
        df_q4[df_q4[sid_col4] == sid].iloc[0, 8] for sid in common_sids
    ], dtype=float)
    etas_q3 = np.array([
        df_q3[df_q3[df_q3.columns[0]] == sid].iloc[0, 10] for sid in common_sids
    ], dtype=float)
    etag_q4 = np.array([
        df_q4[df_q4[sid_col4] == sid].iloc[0, 9] for sid in common_sids
    ], dtype=float)
    etag_q3 = np.array([
        df_q3[df_q3[df_q3.columns[0]] == sid].iloc[0, 11] for sid in common_sids
    ], dtype=float)

    # ── 3 子图 ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(n)
    bar_w = 0.35
    q3_clr = '#1565C0'  # 蓝 — 并网
    q4_clr = colors.get('primary', '#2E7D32')  # 绿 — 离网

    chart_specs = [
        ('吨氨成本 (元/吨)',   ton_q3, ton_q4, '{:.0f}'),
        ('η_self (自给率)',    etas_q3, etas_q4, '{:.3f}'),
        ('η_green (绿电占比)', etag_q3, etag_q4, '{:.3f}'),
    ]

    for ax, (title, dq3, dq4, vfmt) in zip(axes, chart_specs):
        b3 = ax.bar(x - bar_w/2, dq3, bar_w,
                    color=q3_clr, alpha=0.88, edgecolor='white', linewidth=0.5,
                    label='Q3 并网运行')
        b4 = ax.bar(x + bar_w/2, dq4, bar_w,
                    color=q4_clr, alpha=0.88, edgecolor='white', linewidth=0.5,
                    label='Q4 离网+储能')

        # 数值标注
        for bar in b3:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + (h * 0.01),
                    vfmt.format(h), ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold', color=q3_clr, rotation=90)
        for bar in b4:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + (h * 0.01),
                    vfmt.format(h), ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold', color=q4_clr, rotation=90)

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'S{sid}' for sid in common_sids], fontsize=9)
        ax.set_xlabel('场景编号', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')

        # η 指标固定 y 轴 0~1.1
        if 'η' in title:
            ax.set_ylim(0, 1.15)
            ax.axhline(y=1.0, color='#333333', linewidth=0.8,
                       linestyle='--', alpha=0.3)

    # ── 总标题与注释 ──────────────────────────────────────────────────
    fig.suptitle(
        f'Q3 (并网) vs Q4 (离网+储能) 关键指标对比 — {n} 个共同可行场景',
        fontsize=15, fontweight='bold', y=1.01
    )
    if np.mean(ton_q4) > np.mean(ton_q3):
        fig.text(0.5, -0.02,
                 '注: 离网储能方案吨氨成本略高，但实现了能源完全自主 (η_self≈1.0)',
                 ha='center', fontsize=10, color='#888888', style='italic')

    apply_paper_style(fig)
    fig.tight_layout()

    out_path = save_figure(fig, 'q4_vs_q3_comparison.png',
                           results_dir=RESULTS_DIR, dpi=200)
    print(f"  [Q3vsQ4对比] 已保存: {out_path} ({n} scenarios)")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 3: 成本散点图 — Q3 vs Q4 吨氨成本 + y=x 参考对角线              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_cost_scatter(df_q4, df_q3, colors):
    """散点图: x=Q3吨氨成本, y=Q4吨氨成本。

    仅绘制双方都 feasible 的相同场景, 加 y=x 参考线。
    点在线上方 = Q4成本更高 (离网代价), 点在线下方 = Q4成本更低。
    """
    common_sids = _common_feasible_sids(df_q4, df_q3)
    if not common_sids:
        print("  [WARN] Q3 和 Q4 无共同 feasible 场景, 跳过散点图")
        return None

    n = len(common_sids)
    sid_col4 = df_q4.columns[0]

    # 吨氨成本: Q4=col11, Q3=col4
    ton_q4 = np.array([
        df_q4[df_q4[sid_col4] == sid].iloc[0, 11] for sid in common_sids
    ], dtype=float)
    ton_q3 = np.array([
        df_q3[df_q3[df_q3.columns[0]] == sid].iloc[0, 4] for sid in common_sids
    ], dtype=float)

    fig, ax = create_figure(figsize=(10, 9))

    # ── y=x 参考线 ────────────────────────────────────────────────────
    all_vals = np.concatenate([ton_q3, ton_q4])
    margin = (all_vals.max() - all_vals.min()) * 0.10
    ax_min = max(0, all_vals.min() - margin)
    ax_max = all_vals.max() + margin

    ax.plot([ax_min, ax_max], [ax_min, ax_max],
            'k--', linewidth=1.5, alpha=0.45, zorder=1,
            label='y=x (并网成本 = 离网成本)')

    # ── 散点 ──────────────────────────────────────────────────────────
    primary = colors.get('primary', '#2E7D32')
    accent = colors.get('accent', '#FF5252')
    ax.scatter(ton_q3, ton_q4,
               c=primary, s=120, alpha=0.85,
               edgecolors='white', linewidth=1.2,
               zorder=5, label=f'Q3 vs Q4 ({n} 场景)')

    # ── 场景标签 ──────────────────────────────────────────────────────
    for i, sid in enumerate(common_sids):
        ax.annotate(
            f'S{sid}',
            (ton_q3[i], ton_q4[i]),
            textcoords='offset points',
            xytext=(9, 9),
            fontsize=9, fontweight='bold', color='#333333',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                      alpha=0.75, edgecolor='#CCCCCC', linewidth=0.8),
        )

    # ── 区域标注 ──────────────────────────────────────────────────────
    mid_x = (ax_min + ax_max) / 2
    ax.text(ax_max - margin * 1.2, ax_min + margin * 1.5,
            '▼ 离网成本更低', fontsize=9, color=primary,
            fontweight='bold', ha='right', va='bottom', alpha=0.6)
    ax.text(ax_min + margin * 1.2, ax_max - margin * 1.5,
            '▲ 离网成本更高\n  (储能增加成本但提升自主性)',
            fontsize=9, color=accent, fontweight='bold',
            ha='left', va='top', alpha=0.7)

    # ── 装饰 ──────────────────────────────────────────────────────────
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)
    ax.set_aspect('equal')
    ax.set_xlabel('Q3 吨氨成本 (元/吨) — 并网运行', fontsize=12)
    ax.set_ylabel('Q4 吨氨成本 (元/吨) — 离网+储能', fontsize=12)
    ax.set_title(
        f'Q3 vs Q4 吨氨成本散点图 ({n} 个共同可行场景)',
        fontsize=14, fontweight='bold'
    )
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(alpha=0.25, zorder=0)

    apply_paper_style(fig)
    fig.tight_layout()

    out_path = save_figure(fig, 'q4_cost_scatter.png',
                           results_dir=RESULTS_DIR, dpi=200)
    print(f"  [成本散点] 已保存: {out_path} ({n} scenarios)")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    """加载 Q4/Q3 CSV 数据并生成全部 3 张图表。"""
    print("=" * 60)
    print("Q4 可视化 — 离网储能优化结果")
    print("=" * 60)

    colors = get_color_scheme(4)

    # ── 加载数据 ──────────────────────────────────────────────────────
    q4_path = os.path.join(RESULTS_DIR, 'q4_results.csv')
    q3_path = os.path.join(RESULTS_DIR, 'q3_results.csv')

    print("\n[加载数据]")
    df_q4 = pd.read_csv(q4_path, encoding='utf-8-sig')
    df_q3 = pd.read_csv(q3_path, encoding='utf-8-sig')

    n_feas4 = int(np.sum(_get_feasible_mask(df_q4)))
    n_feas3 = int(np.sum(_get_feasible_mask(df_q3)))
    print(f"  Q4: {n_feas4}/{len(df_q4)} 场景 Optimal")
    print(f"  Q3: {n_feas3}/{len(df_q3)} 场景 Optimal")

    # ── Chart 1: SOC 热力图 ────────────────────────────────────────────
    print("\n[1/3] 生成 SOC 逐时热力图...")
    heatmap_path = make_soc_heatmap(df_q4, colors)

    # ── Chart 2: Q3 vs Q4 对比柱图 ─────────────────────────────────────
    print("\n[2/3] 生成 Q3 vs Q4 对比柱图...")
    comparison_path = make_q3_vs_q4_comparison(df_q4, df_q3, colors)

    # ── Chart 3: 成本散点图 ───────────────────────────────────────────
    print("\n[3/3] 生成成本散点图...")
    scatter_path = make_cost_scatter(df_q4, df_q3, colors)

    # ── 验证输出文件 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("输出文件大小验证:")
    print("=" * 60)
    all_ok = True
    has_soc = _has_soc_columns(df_q4)
    for label, path in [
        ('SOC热力图',       heatmap_path),
        ('Q3vsQ4对比图',    comparison_path),
        ('成本散点图',      scatter_path),
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
            if label == 'SOC热力图' and not has_soc:
                print(f"  [SKIP] {label}: 无逐时SOC数据, 已跳过")
            else:
                print(f"  [FAIL] {label}: file not found!")
                all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("All Q4 charts generated successfully.")
    else:
        print("Some charts failed validation - check above.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
