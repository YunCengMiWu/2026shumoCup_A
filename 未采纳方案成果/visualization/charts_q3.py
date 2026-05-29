"""Q3 聚类可视化 — 饼图 / 雷达图 / 箱线图 / 散点图
=================================================
从 results/q3_results.csv + q3_cluster_labels.csv 读取聚类结果，
与 q3_cluster_report.txt 的最优 k / silhouette 信息，
生成 4 张中文标注图表。

输出:
    results/q3_pie.png       — 簇比例饼图
    results/q3_radar.png     — 6 维雷达图 (簇均值)
    results/q3_boxplot.png   — 吨氨成本箱线图 (按簇分组)
    results/q3_scatter.png   — 吨氨成本 vs eta_green 散点图

运行:
    cd A题/code_solution_v2
    python visualization/charts_q3.py
"""

import sys
import os
import re

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 项目根路径 ─────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

# ── 中文字体 (按任务要求导入) ────────────────────────────────────────────
from visualization.chinese_font import setup_chinese_font
setup_chinese_font()

from visualization.plotting import (
    create_figure, save_figure, get_color_scheme,
    apply_paper_style,
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  数据加载                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_data():
    """加载并合并 Q3 结果、聚类标签, 并从 report 解析最优 k 与 silhouette。

    Returns:
        df_merged  (DataFrame):  合并后含 cluster + 模态名称列
        k_opt      (int):        最优聚类数
        sil_score  (float):      Silhouette Score
    """
    # ── CSV ──────────────────────────────────────────────────────────
    df_results = pd.read_csv(
        os.path.join(RESULTS_DIR, 'q3_results.csv'), encoding='utf-8-sig'
    )
    df_labels = pd.read_csv(
        os.path.join(RESULTS_DIR, 'q3_cluster_labels.csv'), encoding='utf-8-sig'
    )

    # 合并 on 场景编号 (确保列类型一致, 跳过汇总行)
    df_results['场景编号'] = pd.to_numeric(df_results['场景编号'], errors='coerce')
    df_results = df_results.dropna(subset=['场景编号'])
    df_results['场景编号'] = df_results['场景编号'].astype(int)
    df_labels['场景编号'] = df_labels['场景编号'].astype(int)
    df = df_results.merge(df_labels, on='场景编号', how='left')

    # ── 解析 q3_cluster_report.txt ───────────────────────────────────
    report_path = os.path.join(RESULTS_DIR, 'q3_cluster_report.txt')
    with open(report_path, 'r', encoding='utf-8-sig') as f:
        report_text = f.read()

    m = re.search(r'最优 k = (\d+), Silhouette Score = ([\d.]+)', report_text)
    if not m:
        raise ValueError('无法从 q3_cluster_report.txt 解析 最优 k 与 Silhouette Score')
    k_opt = int(m.group(1))
    sil_score = float(m.group(2))

    return df, k_opt, sil_score


def _get_cluster_info(df):
    """从合并后的 DataFrame 提取簇元数据 (不硬编码)。

    Returns:
        cluster_ids   (list):  排序后的簇编号
        cluster_names (dict):  {cid: 模态名称}
        cluster_sizes (dict):  {cid: 场景数}
    """
    cluster_ids = sorted(df['聚类标签'].unique())
    cluster_names = {}
    cluster_sizes = {}
    for cid in cluster_ids:
        mask = df['聚类标签'] == cid
        cluster_names[cid] = df.loc[mask, '模态名称'].iloc[0]
        cluster_sizes[cid] = int(mask.sum())
    return cluster_ids, cluster_names, cluster_sizes


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 1: 簇比例饼图                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_pie_chart(df, k_opt, sil_score, colors):
    """饼图 — 各簇场景数量占比 (中文标签 + 百分比)。"""
    cluster_ids, cluster_names, cluster_sizes = _get_cluster_info(df)

    sizes = [cluster_sizes[cid] for cid in cluster_ids]
    labels = [cluster_names[cid] for cid in cluster_ids]
    pie_colors = [colors['primary'], colors['accent']]

    # explode 较小簇
    min_size = min(sizes)
    explode = [0.08 if s == min_size else 0.0 for s in sizes]

    fig, ax = create_figure(figsize=(10, 8))

    wedges, texts, autotexts = ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=pie_colors,
        autopct='%1.1f%%',
        startangle=140,
        pctdistance=0.62,
        labeldistance=1.12,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.8, 'alpha': 0.92},
        textprops={'fontsize': 13},
    )

    for at in autotexts:
        at.set_fontsize(15)
        at.set_fontweight('bold')
        at.set_color('white')

    ax.set_title(
        f'Q3 场景聚类占比 (k={k_opt}, Silhouette={sil_score:.4f})',
        fontsize=16, fontweight='bold', pad=22,
    )

    legend_labels = [
        f'{cluster_names[cid]} (n={cluster_sizes[cid]})'
        for cid in cluster_ids
    ]
    ax.legend(wedges, legend_labels, loc='lower center',
              ncol=2, fontsize=11, framealpha=0.85,
              bbox_to_anchor=(0.5, -0.06))

    apply_paper_style(fig)
    fig.tight_layout()
    out_path = save_figure(fig, 'q3_pie.png', results_dir=RESULTS_DIR, dpi=200)
    print(f"  [饼图] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 2: 6 维雷达图 (簇均值极坐标)                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_radar_chart(df, k_opt, colors):
    """雷达图 — 每簇一条多边形, 6 维特征 (簇均值, Min-Max 归一化到 [0,1])。

    6 个聚类维度 (与 q3_cluster_report.txt 一致):
        吨氨成本(元/吨), eta_self, eta_green, eta_grid, 日购电量(MWh), 日售电量(MWh)
    """
    cluster_ids, cluster_names, cluster_sizes = _get_cluster_info(df)
    cluster_colors = [colors['primary'], colors['accent']]

    feature_cols = [
        '吨氨成本(元/吨)', '新能源自发自用电量占比(%)', '总用电量绿电比例(%)', '新能源上网电量比例(%)',
        '日购电量(MWh)', '日售电量(MWh)',
    ]
    axis_labels = [
        '吨氨成本\n(元/吨)', 'η_self', 'η_green', 'η_grid',
        '日购电量\n(MWh)', '日售电量\n(MWh)',
    ]

    # 簇均值 (原始单位)
    centroids_raw = {}
    for cid in cluster_ids:
        sub = df[df['聚类标签'] == cid][feature_cols]
        centroids_raw[cid] = sub.mean().values.copy()

    # 百分比列 (索引 1-3) 从 0-100 转换为 0-1
    for cid in cluster_ids:
        centroids_raw[cid][1] /= 100.0
        centroids_raw[cid][2] /= 100.0
        centroids_raw[cid][3] /= 100.0

    all_vals = np.array(list(centroids_raw.values()))
    mins = all_vals.min(axis=0)
    maxs = all_vals.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1.0

    N = len(feature_cols)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(9, 9),
                           subplot_kw=dict(projection='polar'))

    for i, cid in enumerate(cluster_ids):
        raw = centroids_raw[cid]
        norm = ((raw - mins) / ranges).tolist()
        norm_closed = norm + norm[:1]

        ax.fill(angles_closed, norm_closed,
                color=cluster_colors[i], alpha=0.14)
        ax.plot(angles_closed, norm_closed,
                color=cluster_colors[i], linewidth=2.5,
                marker='o', markersize=8,
                markerfacecolor=cluster_colors[i],
                markeredgecolor='white', markeredgewidth=1.2,
                label=f'{cluster_names[cid]} ({cluster_sizes[cid]} 场景)')

    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'],
                        fontsize=8, color='#888888')
    ax.set_rlabel_position(15)
    ax.set_title(
        f'Q3 聚类特征雷达图 (k={k_opt})',
        fontsize=16, fontweight='bold', pad=28,
    )
    ax.legend(loc='upper right', bbox_to_anchor=(1.18, 1.12),
              fontsize=11, framealpha=0.88)

    ax.text(0.5, -0.16,
            '注: 各特征经 Min-Max 归一化至 [0,1], 线值为簇均值',
            transform=ax.transAxes, ha='center', fontsize=9,
            color='#666666', style='italic')

    fig.tight_layout()
    out_path = save_figure(fig, 'q3_radar.png', results_dir=RESULTS_DIR, dpi=200)
    print(f"  [雷达图] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 3: 吨氨成本分簇箱线图                                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_boxplot(df, k_opt, sil_score, colors):
    """分组箱线图 — x=簇名 (中文), y=吨氨成本(元/吨)。"""
    cluster_ids, cluster_names, cluster_sizes = _get_cluster_info(df)
    cluster_colors = [colors['primary'], colors['accent']]

    box_data = [
        df[df['聚类标签'] == cid]['吨氨成本(元/吨)'].dropna().values
        for cid in cluster_ids
    ]
    box_labels = [cluster_names[cid] for cid in cluster_ids]

    fig, ax = create_figure(figsize=(10, 7))

    bp = ax.boxplot(
        box_data,
        tick_labels=box_labels,
        patch_artist=True,
        showfliers=True,
        widths=0.45,
        showmeans=True,
        meanprops=dict(marker='D', markerfacecolor='#FFEB3B',
                       markeredgecolor='#333333', markersize=9),
    )

    for patch, clr in zip(bp['boxes'], cluster_colors):
        patch.set_facecolor(clr)
        patch.set_alpha(0.55)
        patch.set_edgecolor('#333333')
        patch.set_linewidth(1.5)
    for whisker in bp['whiskers']:
        whisker.set_color('#555555')
        whisker.set_linewidth(1.4)
    for cap in bp['caps']:
        cap.set_color('#555555')
        cap.set_linewidth(1.4)
    for flier in bp['fliers']:
        flier.set_markeredgecolor('#D32F2F')
        flier.set_markerfacecolor('#FF8A80')
        flier.set_markersize(8)
        flier.set_alpha(0.8)
    for median in bp['medians']:
        median.set_color('#111111')
        median.set_linewidth(2.2)

    ax.set_ylabel('吨氨成本 (元/吨)', fontsize=13)
    ax.set_title(
        f'Q3 各模态吨氨成本分布 (k={k_opt}, Silhouette={sil_score:.4f})',
        fontsize=15, fontweight='bold',
    )
    ax.grid(axis='y', alpha=0.30)
    ax.tick_params(axis='x', labelsize=13)

    # 均值标注
    for i, (data, clr) in enumerate(zip(box_data, cluster_colors)):
        if len(data) > 0:
            mean_val = np.mean(data)
            ax.text(i + 1, mean_val, f'  μ={mean_val:.0f}',
                    fontsize=10, color=clr, fontweight='bold',
                    va='center', ha='left')

    apply_paper_style(fig)
    fig.tight_layout()
    out_path = save_figure(fig, 'q3_boxplot.png', results_dir=RESULTS_DIR, dpi=200)
    print(f"  [箱线图] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Chart 4: 吨氨成本 vs eta_green 散点图 (含 Silhouette 标注)             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def make_scatter(df, k_opt, sil_score, colors):
    """簇着色散点图 — x=吨氨成本(元/吨), y=eta_green, 标题含 silhouette。"""
    cluster_ids, cluster_names, cluster_sizes = _get_cluster_info(df)
    cluster_colors = [colors['primary'], colors['accent']]

    fig, ax = create_figure(figsize=(12, 8))

    for i, cid in enumerate(cluster_ids):
        sub = df[df['聚类标签'] == cid]
        ax.scatter(
            sub['吨氨成本(元/吨)'], sub['总用电量绿电比例(%)'] / 100.0,
            c=cluster_colors[i],
            s=130, alpha=0.80,
            edgecolors='white', linewidth=0.8,
            zorder=3,
            label=f'{cluster_names[cid]} ({cluster_sizes[cid]} 场景)',
        )

        # 簇重心 (大十字)
        cx = sub['吨氨成本(元/吨)'].mean()
        cy = sub['总用电量绿电比例(%)'].mean() / 100.0
        ax.scatter([cx], [cy], marker='X', s=280,
                   c=cluster_colors[i],
                   edgecolors='#111111', linewidth=1.5,
                   zorder=5)

    # 全局均值参考线
    global_ton = df['吨氨成本(元/吨)'].mean()
    global_eta = df['总用电量绿电比例(%)'].mean() / 100.0
    ax.axvline(x=global_ton, color='#999999', linestyle=':', alpha=0.55, linewidth=1.2)
    ax.axhline(y=global_eta, color='#999999', linestyle=':', alpha=0.55, linewidth=1.2)

    # Silhouette 标注 (右上角)
    ax.text(
        0.97, 0.06,
        f'Silhouette = {sil_score:.4f}',
        transform=ax.transAxes,
        fontsize=13, color='#1565C0', fontweight='bold',
        ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.45',
                  facecolor='#E3F2FD',
                  edgecolor='#1565C0', alpha=0.88),
    )

    ax.set_xlabel('吨氨成本 (元/吨)', fontsize=13)
    ax.set_ylabel('η_green (绿电占比)', fontsize=13)
    ax.set_title(
        f'Q3 场景成本-绿电指标散点图 (k={k_opt}, Silhouette={sil_score:.4f})',
        fontsize=16, fontweight='bold',
    )
    ax.legend(loc='upper left', fontsize=11, framealpha=0.90, markerscale=0.9)
    ax.grid(alpha=0.25)

    apply_paper_style(fig)
    fig.tight_layout()
    out_path = save_figure(fig, 'q3_scatter.png', results_dir=RESULTS_DIR, dpi=200)
    print(f"  [散点图] 已保存: {out_path}")
    return out_path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Main                                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    """加载 Q3 数据并生成全部 4 张聚类可视化图表。"""
    print("=" * 60)
    print("Q3 聚类可视化 — 饼图 / 雷达图 / 箱线图 / 散点图")
    print("=" * 60)

    colors = get_color_scheme(3)

    # ── 加载数据 ──────────────────────────────────────────────────────
    print("\n[加载数据]")
    df, k_opt, sil_score = load_data()
    cluster_ids, cluster_names, cluster_sizes = _get_cluster_info(df)
    print(f"  场景总数: {len(df)}")
    print(f"  最优 k = {k_opt}, Silhouette Score = {sil_score:.4f}")
    for cid in cluster_ids:
        print(f"    簇 {cid}: {cluster_names[cid]} ({cluster_sizes[cid]} 场景)")

    # ── 生成 4 张图表 ─────────────────────────────────────────────────
    print("\n[1/4] 生成簇比例饼图...")
    pie_path = make_pie_chart(df, k_opt, sil_score, colors)

    print("\n[2/4] 生成六维雷达图...")
    radar_path = make_radar_chart(df, k_opt, colors)

    print("\n[3/4] 生成吨氨成本箱线图...")
    box_path = make_boxplot(df, k_opt, sil_score, colors)

    print("\n[4/4] 生成成本 vs 绿电占比散点图...")
    scatter_path = make_scatter(df, k_opt, sil_score, colors)

    # ── 验证输出文件 ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("输出文件大小验证 (均需 >15KB):")
    print("=" * 60)
    all_ok = True
    for label, path in [
        ('饼图',   pie_path),
        ('雷达图', radar_path),
        ('箱线图', box_path),
        ('散点图', scatter_path),
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
        print("Q3 聚类可视化完成: 4张图表已保存至 results/")
    else:
        print("部分图表未通过验证 — 请检查上方日志。")
    print("=" * 60)

    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
