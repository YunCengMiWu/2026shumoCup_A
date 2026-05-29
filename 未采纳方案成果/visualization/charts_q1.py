"""Q1 可视化: 典型日绿电直连指标评估
=====================================
生成三张图表 (全部中文标签):
  1. Plotly 桑基图 (q1_sankey.html)        — 风电/光伏/网购电 → 基荷/制氢/合成氨/售电
   2. 多指标折线图 (q1_multi_metric.png)     — 6 曲线 + 吨氨成本副轴
  3. 功率平衡堆叠柱状图 (q1_power_balance.png) — 源-荷双侧堆叠

运行:
    cd A题/code_solution_v2
    python visualization/charts_q1.py
"""
import sys
import os
import numpy as np
import plotly.graph_objects as go

# ── Path setup ──────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.run_q1 import run_q1
from utils.constants import DELTA_T, HOURS_PER_DAY
from visualization.plotting import (
    create_figure, save_figure, get_color_scheme,
    style_axis, set_hour_xticks, add_legend_outside,
    add_value_labels, apply_paper_style,
)

# ── 结果目录 (项目根下的 results/) ─────────────────────────────────────────
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_PROJ_ROOT, 'results')


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 1: Plotly 桑基图 — 能源流比例分配                              ║
# ║  3 源 (风电/光伏/网购电) → 4 汇 (基荷/制氢/合成氨/售电)               ║
# ║  按源侧占比比例分配到各汇                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_sankey(result, colors):
    """生成交互式 Plotly 桑基图，保存为 HTML。

    按源侧发电占比，将日总能量比例分配到各负荷/售电侧。
    """
    p_wind = result['p_wind']
    p_pv   = result['p_pv']
    p_buy  = result['p_buy']
    p_base = result['p_base']
    p_h2   = result['p_alk'] + result['p_pem']   # 制氢合计
    p_nh3  = result['p_nh3']
    p_sell = result['p_sell']

    # 日总能量 (MWh)
    E_wind = float(np.sum(p_wind)) * DELTA_T
    E_pv   = float(np.sum(p_pv))   * DELTA_T
    E_buy  = float(np.sum(p_buy))  * DELTA_T
    E_base = float(np.sum(p_base)) * DELTA_T
    E_h2   = float(np.sum(p_h2))   * DELTA_T
    E_nh3  = float(np.sum(p_nh3))  * DELTA_T
    E_sell = float(np.sum(p_sell)) * DELTA_T

    E_source = E_wind + E_pv + E_buy
    if E_source <= 0:
        print("  [WARN] 源侧总能量为零，跳过桑基图")
        return None

    # 各源占比
    share_wind = E_wind / E_source
    share_pv   = E_pv   / E_source
    share_buy  = E_buy  / E_source
    shares = [share_wind, share_pv, share_buy]

    # 各汇能量
    sink_vals = [E_base, E_h2, E_nh3, E_sell]

    # ── 节点 ──────────────────────────────────────────────────────────
    labels = ['风电', '光伏', '网购电', '基荷', '制氢', '合成氨', '售电']
    node_colors = [
        colors['wind'],       # 风电
        colors['pv'],         # 光伏
        colors['grid_buy'],   # 网购电
        colors['load'],       # 基荷
        colors['h2'],         # 制氢
        colors['nh3'],        # 合成氨
        colors['grid_sell'],  # 售电
    ]

    # ── 链接: 3 源 × 4 汇 = 12 条 (比例分配) ──────────────────────────
    source_idx = []
    target_idx = []
    link_vals  = []
    link_colors_list = []
    link_labels_list  = []

    source_names = ['风电', '光伏', '网购电']
    sink_names   = ['基荷', '制氢', '合成氨', '售电']
    # 源色 → rgba 半透明用于链接
    source_hex = [colors['wind'], colors['pv'], colors['grid_buy']]

    for si in range(3):
        sc = shares[si]
        r, g, b = int(source_hex[si][1:3], 16), int(source_hex[si][3:5], 16), int(source_hex[si][5:7], 16)
        link_rgba = f'rgba({r},{g},{b},0.35)'
        for ti in range(4):
            val = round(sink_vals[ti] * sc, 1)
            if val < 0.05:
                continue
            source_idx.append(si)
            target_idx.append(3 + ti)
            link_vals.append(val)
            link_colors_list.append(link_rgba)
            link_labels_list.append(
                f'{source_names[si]} → {sink_names[ti]}: {val:.1f} MWh'
            )

    # ── 构建 Figure ────────────────────────────────────────────────────
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=22,
            thickness=28,
            line=dict(color='black', width=1.0),
            label=labels,
            color=node_colors,
        ),
        link=dict(
            source=source_idx,
            target=target_idx,
            value=link_vals,
            color=link_colors_list,
            label=link_labels_list,
            hovertemplate='%{label}<extra></extra>',
        ),
    )])

    fig.update_layout(
        title=dict(
            text=f'Q1 典型日能源流分配桑基图（总用电 {E_base+E_h2+E_nh3+E_sell:.0f} MWh）',
            font=dict(size=17, family='SimHei, Microsoft YaHei, sans-serif'),
        ),
        font=dict(size=13, family='SimHei, Microsoft YaHei, sans-serif'),
        width=1100,
        height=600,
        margin=dict(l=30, r=30, t=60, b=30),
    )

    # ── 保存 HTML ──────────────────────────────────────────────────────
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, 'q1_sankey.html')
    fig.write_html(out_path, auto_open=False)
    print(f"  [桑基图] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 2: 多指标折线图 — 6 曲线 + 吨氨成本副轴                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_multi_metric(result, indicators, colors):
    """折线图: 总用电量 / 新能源发电合计 / 风力发电 / 光伏发电 / 网购电量 / 上网电量 / 吨氨成本(副轴)。

     左 Y 轴: 功率/电量 (MW / MWh per hour, DELTA_T=1h 故数值相等)
     右 Y 轴: 吨氨成本 (元/吨) — 日均水平参考线
     """
    hours = np.arange(HOURS_PER_DAY)

    p_load = result['p_load']
    p_gen  = result['p_gen']
    p_wind = result['p_wind']
    p_pv   = result['p_pv']
    p_buy  = result['p_buy']
    p_sell = result['p_sell']
    ton_cost_daily = indicators['ton_cost']

    fig, ax1 = create_figure(figsize=(14, 7))

    # ── 左轴: 6 条功率曲线 ────────────────────────────────────────────
    ax1.plot(hours, p_load, color=colors['load'],
             marker='s', markersize=5, linewidth=2.2,
             label='总用电量 (MWh)', zorder=3)
    ax1.plot(hours, p_gen, color=colors['primary'],
             marker='o', markersize=5, linewidth=2.2,
             label='新能源发电量 (MWh)', zorder=3)
    ax1.plot(hours, p_wind, color=colors['wind'],
             marker='D', markersize=5, linewidth=1.8,
             label='风电发电量 (MWh)', zorder=3)
    ax1.plot(hours, p_pv, color=colors['pv'],
             marker='h', markersize=5, linewidth=1.8,
             label='光伏发电量 (MWh)', zorder=3)
    ax1.plot(hours, p_buy, color=colors['grid_buy'],
             marker='^', markersize=5, linewidth=2.0,
             linestyle='--', label='网购电量 (MWh)', zorder=3)
    ax1.plot(hours, p_sell, color=colors['grid_sell'],
             marker='v', markersize=5, linewidth=2.0,
             linestyle='--', label='上网电量 (MWh)', zorder=3)

    ax1.set_ylabel('功率 / 电量 (MW / MWh)', fontsize=12)
    ax1.grid(alpha=0.3, zorder=0)

    # ── 右轴: 吨氨成本 (日均参考线) ────────────────────────────────────
    ax2 = ax1.twinx()
    ax2.axhline(y=ton_cost_daily, color=colors['accent'],
                linewidth=2.5, linestyle='-.', alpha=0.85, zorder=4,
                label=f'吨氨成本 = {ton_cost_daily:.1f} 元/吨')
    ax2.set_ylabel('吨氨成本 (元/吨)', fontsize=12, color=colors['accent'])
    ax2.tick_params(axis='y', labelcolor=colors['accent'])
    # 使参考线可视范围合理
    pad_y = max(ton_cost_daily * 0.4, 80)
    ax2.set_ylim(ton_cost_daily - pad_y, ton_cost_daily + pad_y)

    # ── 装饰 ──────────────────────────────────────────────────────────
    set_hour_xticks(ax1)
    style_axis(ax1, title='Q1 典型日多指标运行曲线')

    # 合并双轴图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    add_legend_outside(ax2, ncol=1)
    # 手动设合并后的图例
    ax2.legend(h1 + h2, l1 + l2,
               loc='upper left', bbox_to_anchor=(1.01, 1.0),
               framealpha=0.85, fontsize=10)

    fig.tight_layout()
    out_path = save_figure(fig, 'q1_multi_metric.png',
                           results_dir=RESULTS_DIR, dpi=200)
    print(f"  [多指标图] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 3: 功率平衡堆叠柱状图 — 源-荷双侧                              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _add_stacked_value_labels(ax, bars_list, bottoms_list):
    """为堆叠柱状图各段添加数值标签 (含正/负方向)。"""
    for bars, bottoms in zip(bars_list, bottoms_list):
        for bar, bot in zip(bars, bottoms):
            h = bar.get_height()
            if abs(h) < 0.15:
                continue
            xc = bar.get_x() + bar.get_width() / 2
            yc = bot + h
            va = 'bottom' if h >= 0 else 'top'
            off = 0.3 if h >= 0 else -0.3
            ax.text(xc, yc + off, f'{abs(h):.1f}',
                    ha='center', va=va, fontsize=7,
                    color='#333333', fontweight='bold')


def make_power_balance(result, colors):
    """堆叠柱状图: 源侧(风电+光伏+网购电) ↑ vs 荷侧(基荷+制氢+合成氨+售电) ↓。

    正半轴 (source):    风电 + 光伏 + 网购电
    负半轴 (load):      基荷 + 制氢(ALK+PEM) + 合成氨 + 售电
    """
    hours = np.arange(HOURS_PER_DAY)
    bar_w = 0.55
    zeros = np.zeros(HOURS_PER_DAY)

    p_wind = result['p_wind']
    p_pv   = result['p_pv']
    p_buy  = result['p_buy']
    p_base = result['p_base']
    p_h2   = result['p_alk'] + result['p_pem']
    p_nh3  = result['p_nh3']
    p_sell = result['p_sell']

    fig, ax = create_figure(figsize=(18, 8))

    # ── 零线 ──────────────────────────────────────────────────────────
    ax.axhline(y=0, color='black', linewidth=1.3, zorder=0)

    # ═══ 正半轴: 源侧 (发电 + 购电) ═══
    b_wind = ax.bar(hours, p_wind, bar_w,
                    color=colors['wind'], alpha=0.92,
                    edgecolor='white', linewidth=0.3, label='风电')
    b_pv = ax.bar(hours, p_pv, bar_w, bottom=p_wind,
                  color=colors['pv'], alpha=0.92,
                  edgecolor='white', linewidth=0.3, label='光伏')
    b_buy = ax.bar(hours, p_buy, bar_w, bottom=p_wind + p_pv,
                   color=colors['grid_buy'], alpha=0.92,
                   edgecolor='white', linewidth=0.3, label='网购电')

    # ═══ 负半轴: 荷侧 (用电 + 售电) ═══
    neg_base = -p_base
    b_base = ax.bar(hours, neg_base, bar_w,
                    color=colors['load'], alpha=0.88,
                    edgecolor='white', linewidth=0.3, label='基荷')
    neg_h2 = -p_h2
    b_h2 = ax.bar(hours, neg_h2, bar_w, bottom=neg_base,
                  color=colors['h2'], alpha=0.88,
                  edgecolor='white', linewidth=0.3, label='制氢 (ALK+PEM)')
    neg_nh3 = -p_nh3
    b_nh3 = ax.bar(hours, neg_nh3, bar_w, bottom=neg_base + neg_h2,
                   color=colors['nh3'], alpha=0.88,
                   edgecolor='white', linewidth=0.3, label='合成氨')
    neg_sell = -p_sell
    b_sell = ax.bar(hours, neg_sell, bar_w, bottom=neg_base + neg_h2 + neg_nh3,
                    color=colors['grid_sell'], alpha=0.88,
                    edgecolor='white', linewidth=0.3, label='售电')

    # ── add_value_labels: 为每个非堆叠段标注 ─────────────────────────
    _add_stacked_value_labels(
        ax,
        [b_wind, b_pv, b_buy],
        [zeros, p_wind, p_wind + p_pv],
    )
    _add_stacked_value_labels(
        ax,
        [b_base, b_h2, b_nh3, b_sell],
        [zeros, neg_base, neg_base + neg_h2, neg_base + neg_h2 + neg_nh3],
    )

    # ── 源/荷标注 ────────────────────────────────────────────────────
    src_max = np.max(p_wind + p_pv + p_buy)
    load_max = np.max(p_base + p_h2 + p_nh3 + p_sell)
    y_max = max(src_max, 15)
    y_min = -max(load_max, 15)
    ax.set_ylim(y_min - 4, y_max + 4)

    # 源侧 / 荷侧 标签
    ax.text(23.7, y_max * 0.55, '源 侧', fontsize=13,
            fontweight='bold', color=colors['primary'],
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      alpha=0.75, edgecolor=colors['primary'], linewidth=1.2))
    ax.text(23.7, y_min * 0.55, '荷 侧', fontsize=13,
            fontweight='bold', color=colors['load'],
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      alpha=0.75, edgecolor=colors['load'], linewidth=1.2))

    # ── 装饰 ──────────────────────────────────────────────────────────
    set_hour_xticks(ax)
    style_axis(ax, title='Q1 典型日功率平衡堆叠柱状图',
               ylabel='功率 (MW)')
    ax.grid(axis='y', alpha=0.25, zorder=-1)

    # 分组图例: 源侧 / 荷侧
    legend_src = ax.legend(
        [b_wind, b_pv, b_buy],
        ['风电', '光伏', '网购电'],
        loc='upper right', framealpha=0.9,
        title='源侧 (发电+购电)', title_fontsize=10, fontsize=9,
    )
    ax.add_artist(legend_src)
    ax.legend(
        [b_base, b_h2, b_nh3, b_sell],
        ['基荷', '制氢 (ALK+PEM)', '合成氨', '售电'],
        loc='lower right', framealpha=0.9,
        title='荷侧 (用电+售电)', title_fontsize=10, fontsize=9,
    )

    # ── 论文风格 ──────────────────────────────────────────────────────
    apply_paper_style(fig)

    fig.tight_layout()
    out_path = save_figure(fig, 'q1_power_balance.png',
                           results_dir=RESULTS_DIR, dpi=200)
    print(f"  [功率平衡图] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    """运行 Q1 模型并生成全部 3 张图表。"""
    print("=" * 60)
    print("Q1 可视化 — 生成三张图表")
    print("=" * 60)

    # ── 加载 Q1 计算结果 ──────────────────────────────────────────────
    print("\n[1/4] 运行 Q1 模型...")
    result, indicators = run_q1()
    colors = get_color_scheme(1)

    # ── Chart 1: 桑基图 ───────────────────────────────────────────────
    print("\n[2/4] 生成桑基图 (Plotly 交互式 HTML)...")
    sankey_path = make_sankey(result, colors)

    # ── Chart 2: 多指标折线图 ─────────────────────────────────────────
    print("\n[3/4] 生成多指标折线图...")
    multi_path = make_multi_metric(result, indicators, colors)

    # ── Chart 3: 功率平衡堆叠柱状图 ───────────────────────────────────
    print("\n[4/4] 生成功率平衡堆叠柱状图...")
    balance_path = make_power_balance(result, colors)

    # ── 验证输出文件 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("输出文件大小验证:")
    print("=" * 60)
    all_ok = True
    for label, path in [
        ('桑基图 HTML', sankey_path),
        ('多指标折线图', multi_path),
        ('功率平衡图',  balance_path),
    ]:
        if path and os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            ok = size_kb > 15
            marker = 'OK' if ok else 'WARN'
            print(f"  [{marker}] {label}: {path}")
            print(f"         size: {size_kb:.1f} KB {'(>15KB)' if ok else '(<15KB!)'}")
            if not ok:
                all_ok = False
        else:
            print(f"  [FAIL] {label}: file not found!")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("All Q1 charts generated successfully.")
    else:
        print("Some charts failed validation - check above.")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
