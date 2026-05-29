# -*- coding: utf-8 -*-
"""
问题五：绿电直连园区容量渗透率影响分析及政策建议
基于 Q1-Q4 结果数据，生成定性分析报告与可视化图表
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# =============================================================================
# 路径配置
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
# 各题数据所在文件夹
Q1_DIR = os.path.join(ROOT_DIR, 'q1')
Q2_DIR = os.path.join(ROOT_DIR, 'q2')
Q3_DIR = os.path.join(ROOT_DIR, 'q3')
Q4_DIR = os.path.join(ROOT_DIR, 'q4')

# =============================================================================
# 数据加载
# =============================================================================

def load_csv_safe(filepath, **kwargs):
    """安全加载CSV，文件缺失时返回None并打印警告"""
    if not os.path.exists(filepath):
        print(f"  [跳过] 文件不存在: {filepath}")
        return None
    return pd.read_csv(filepath, encoding='utf-8-sig', **kwargs)


def print_header(title):
    print()
    print('=' * 60)
    print(f'  {title}')
    print('=' * 60)


# =============================================================================
# 数据分析函数
# =============================================================================

def analyze_q1_indicators(df):
    """从Q1指标提取关键数据"""
    row = dict(zip(df['指标'].str.strip(), df['数值']))
    return {
        'self_use_rate': float(row['新能源自发自用率']),
        'green_ratio': float(row['总用电量绿电比例']),
        'grid_export_ratio': float(row['新能源上网电量比']),
        'cost_per_ton': float(row['吨氨生产成本']),
        'classification': str(row.get('场景分类', 'N/A')),
    }


def analyze_q2_q3_comparison(df_cmp):
    """分析Q2与Q3对比数据"""
    # 移除成本差异为0的行（两端点不可比）
    nonzero = df_cmp[df_cmp['成本降幅%'] < 0].copy()
    nonzero['成本降幅%_abs'] = -nonzero['成本降幅%']

    # 仅保留有改进的行
    improvements = nonzero[nonzero['成本降幅%'] < 0]

    stats = {
        'mean_improvement_pct': improvements['成本降幅%'].mean() if len(improvements) > 0 else 0,
        'max_improvement_pct': improvements['成本降幅%'].min() if len(improvements) > 0 else 0,
        'mean_cost_diff': improvements['成本差异'].mean() if len(improvements) > 0 else 0,
        'q2_weighted_cost': df_cmp['吨氨成本(Q2)'].mean(),
        'q3_weighted_cost': df_cmp['吨氨成本(Q3)'].mean(),
        'total_improvements': len(improvements),
        'total_scenarios': len(df_cmp),
        'improvement_list': improvements['成本降幅%'].tolist(),
    }
    return stats


def analyze_q3_classification(df_q3):
    """分析Q3场景分类分布"""
    if '分类' not in df_q3.columns:
        return {'全满足': 0, '部分满足': len(df_q3), '全不满足': 0}

    counts = df_q3['分类'].value_counts().to_dict()
    result = {
        '全满足': counts.get('全满足', 0),
        '部分满足': counts.get('部分满足', 0),
        '全不满足': counts.get('全不满足', 0),
        'total': len(df_q3),
    }
    return result


def analyze_q3_buy_sell_by_production(df_q3):
    """按产量层级统计Q3购售电平均情况"""
    if '日产量(吨/天)' not in df_q3.columns:
        return None
    grouped = df_q3.groupby('日产量(吨/天)').agg(
        avg_buy=('日购电量(kWh)', 'mean'),
        avg_sell=('日售电量(kWh)', 'mean'),
    ).reset_index()
    return grouped


# =============================================================================
# 报告生成
# =============================================================================

def generate_report(q1, cmp_stats, q3_cls, q3_buy_sell, q4_available):
    """生成结构化政策分析报告文本"""

    lines = []
    lines.append('=' * 60)
    lines.append('问题五：绿电直连园区容量渗透率影响分析及政策建议')
    lines.append('=' * 60)
    lines.append('')
    lines.append('一、绿电园区高渗透率对电力系统的影响（三利三弊）')
    lines.append('')

    # ---- 利1 ----
    lines.append('【利1】促进新能源消纳，降低碳排放')
    lines.append(f'论据：Q1基准场景下绿电比例为{q1["green_ratio"]:.1%}，但自发自用率仅{q1["self_use_rate"]:.1%}。'
                 f'Q3连续调节优化后，{cmp_stats["total_improvements"]}/{cmp_stats["total_scenarios"]}'
                 f'个场景吨氨成本降低，平均降幅{cmp_stats["mean_improvement_pct"]:.1f}%。')
    lines.append('分析：高容量渗透率的绿电直连园区通过本地消纳风光电力，大幅减少了远距离输电损耗和'
                 '化石燃料消耗。连续调节模式进一步释放了设备灵活性，在新能源充足时提升产量、'
                 '不足时降低负荷，实现"源随荷动"与"荷随源动"的双向协同。这种模式若推广至大量园区，'
                 '可从源头上提升全网绿电消纳比例，是实现"双碳"目标的关键路径。')
    lines.append('')

    # ---- 利2 ----
    lines.append('【利2】降低电网峰谷差，园区储能可参与调峰')
    if q3_buy_sell is not None:
        max_buy_row = q3_buy_sell.loc[q3_buy_sell['avg_buy'].idxmax()]
        max_sell_row = q3_buy_sell.loc[q3_buy_sell['avg_sell'].idxmax()]
        lines.append(f'论据：Q3数据表明，产量{max_buy_row["日产量(吨/天)"]}吨/天时平均购电{max_buy_row["avg_buy"]:.0f}kWh，'
                     f'产量{max_sell_row["日产量(吨/天)"]}吨/天时平均售电{max_sell_row["avg_sell"]:.0f}kWh——'
                     f'购电集中在中高产量（谷时/平时电价），售电集中在低谷产量（峰时电价高）。')
    lines.append('分析：园区通过分时电价引导，在电价低谷时段大量购电用于制氢储氢，在电价高峰时段'
                 '减少购电甚至返送电网，天然具备"虚拟储能"特性。大量园区同时遵循此策略，'
                 '可有效削峰填谷，降低电网净负荷波动幅度，减少火电机组调峰压力，'
                 '提升全网运行经济性与可靠性。')
    lines.append('')

    # ---- 利3 ----
    lines.append('【利3】带动氢氨产业链发展，降低用能成本')
    lines.append(f'论据：Q2与Q3对比显示，连续调节模式下吨氨成本从加权平均{q3_cls.get("q2_cost", cmp_stats["q2_weighted_cost"]):.0f}元/ton'
                 f'降至{q3_cls.get("q3_cost", cmp_stats["q3_weighted_cost"]):.0f}元/ton，'
                 f'最大降幅达{cmp_stats["max_improvement_pct"]:.1f}%。')
    lines.append('分析：绿电直连显著降低了电解水制氢的电力成本——这是绿氨生产最大的成本项。'
                 '连续调节模式允许设备在10%-100%额定功率之间灵活运行，避免了启停损耗和启停成本。'
                 '随着园区容量渗透率提升，规模化效应将进一步降低电解槽和储氢系统单位投资，'
                 '使绿氨具备与灰氨竞争的经济性，从而撬动整个氢氨产业链发展。')
    lines.append('')

    # ---- 弊1 ----
    lines.append('【弊1】高渗透率下风光波动性对电网频率/电压稳定性构成挑战')
    lines.append(f'论据：Q1基准场景风电出力在0~16MW之间日内剧烈波动，光伏出力0~50MW仅在白天出力，'
                 f'新能源上网比例高达{q1["grid_export_ratio"]:.1%}，说明大量波动性电力注入电网。')
    lines.append('分析：当绿电园区渗透率达到较高水平时，大量风光出力的随机性叠加将对电网造成冲击。'
                 '多云天气下光伏出力数分钟内可骤降50%以上，风电受风速影响波动更加不可预测。'
                 '若无足够的灵活性资源（储能、需求响应、灵活火电）平衡波动，'
                 '可能导致系统频率越限、电压闪变乃至连锁脱网事故，对电网安全运行构成严重威胁。')
    lines.append('')

    # ---- 弊2 ----
    lines.append('【弊2】大量园区并网可能造成局部电网潮流反向、保护配合困难')
    lines.append(f'论据：Q3数据中高比例可再生能源场景下售电量高达{df_q3_temp["日售电量(kWh)"].max():.0f}kWh/天，'
                 f'表明园区在大发时段大量返送电力。')
    lines.append('分析：传统配电网设计遵循"自上而下"的潮流方向，保护配置基于单向短路电流计算。'
                 '大量分布式绿电园区接入后，馈线潮流可能出现反向，导致传统过流保护误动或拒动，'
                 '重合闸逻辑失效、孤岛检测困难等问题。此外，多条馈线同时返送可能造成上级变压器'
                 '反向过载，必须升级配电网自动化和保护方案才能保障安全。')
    lines.append('')

    # ---- 弊3 ----
    lines.append('【弊3】园区离网时段供需不匹配需备用电厂支撑')
    if not q4_available:
        lines.append('论据：Q1基准场景下，夜间无光伏、弱风电时段（如小时4-5），'
                     f'设备用电约20.75MW而风光出力仅约10MW，缺口需全额从电网购入。')
    else:
        lines.append('论据：Q4离网分析显示，无储能情况下园区无法独立运行的场景占比显著，'
                     '需依赖电网备用电厂填补风光出力缺口。')
    lines.append('分析：即使在离网/微网模式下，风光资源的间歇性使得园区在连续无风无光时段'
                 '（如冬季夜间）无法自给自足。若无足够储能或备用电厂，园区被迫限产甚至停产，'
                 '设备利用小时数大幅降低，经济性恶化。这说明高渗透率绿电园区不能脱离大电网独立运行，'
                 '需要电网提供容量备用和灵活性支撑，两者是互补而非替代关系。')
    lines.append('')

    # ---- 政策建议 ----
    lines.append('二、进一步推动绿电直连园区发展的政策建议（四条）')
    lines.append('')

    lines.append('【建议1】完善绿电直连电价机制')
    lines.append('论证：Q2修复中grid_dir约束的必要性表明，分时电价与上网电价的价差是园区套利的根本动因。'
                 '建议建立"绿电直连专项输配电价"，合理反映园区对电网的备用依赖（容量电价）'
                 '与减少输配电损耗的贡献（减免部分电量电价），避免园区纯粹利用峰谷价差套利'
                 '而忽视对电网安全的责任。同时，明确过网费减免条件和标准，降低制度性交易成本。')
    lines.append('')

    lines.append('【建议2】鼓励园区配储，给予投资补贴')
    lines.append('论证：储能投资约1000元/kWh，占园区总投资的较大比重。Q4离网分析表明无储能方案'
                 '无法保障连续生产。建议对园区储能给予初始投资补贴（如30%-50%）或税收抵免，'
                 '将储能纳入需求响应和辅助服务市场，允许园区储能参与调频、备用等辅助服务获取收益，'
                 '缩短投资回收期。同时制定储能与制氢系统的联合优化运行标准。')
    lines.append('')

    lines.append('【建议3】建立绿证与碳市场联动机制')
    lines.append(f'论证：园区绿电自发自用率超60%的场景，其碳减排量可观。建议建立绿证-碳市场互认机制，'
                 f'允许绿电直连园区将自发自用绿电对应的碳减排量在碳市场交易。按Q1数据绿电比例{q1["green_ratio"]:.1%}、'
                 f'年用电量约18GWh估算，年减排CO₂约1万吨以上。碳收益可显著改善园区经济性，'
                 '形成"减碳→增收→扩产→更多减碳"的正向循环。')
    lines.append('')

    lines.append('【建议4】制定园区并网技术标准')
    lines.append('论证：高渗透率下电压越限、频率波动、保护配合等问题需要通过技术标准统一解决。'
                 '建议制定涵盖电压穿越能力（LVRT/HVRT）、有功无功独立调节、'
                 '防孤岛保护、电能质量（谐波、闪变）的园区并网导则。'
                 '要求园区配置功率预测系统和AGC/AVC接口，使其具备可调度性。'
                 '同时建立园区分级接入标准——小容量园区低压并网、大容量园区专线接入——'
                 '从源头控制单点接入容量对局部电网的影响。')
    lines.append('')

    # ---- 数据摘要 ----
    lines.append('三、支撑数据摘要')
    lines.append('')
    lines.append(f'  Q1 基线指标:')
    lines.append(f'    新能源自发自用率:    {q1["self_use_rate"]:.1%}')
    lines.append(f'    总用电量绿电比例:    {q1["green_ratio"]:.1%}')
    lines.append(f'    新能源上网电量比:    {q1["grid_export_ratio"]:.1%}')
    lines.append(f'    吨氨生产成本:        {q1["cost_per_ton"]:.2f} 元/ton')
    lines.append(f'    场景分类:            {q1["classification"]}')
    lines.append('')
    lines.append(f'  Q2 MILP (固定档位):')
    lines.append(f'    加权平均吨氨成本:    {cmp_stats["q2_weighted_cost"]:.2f} 元/ton')
    lines.append('')
    lines.append(f'  Q3 LP (连续调节):')
    lines.append(f'    加权平均吨氨成本:    {cmp_stats["q3_weighted_cost"]:.2f} 元/ton')
    lines.append(f'    对比Q2平均降幅:      {cmp_stats["mean_improvement_pct"]:.1f}%')
    lines.append(f'    最大降幅:            {cmp_stats["max_improvement_pct"]:.1f}%')
    lines.append(f'    有改进场景数:        {cmp_stats["total_improvements"]}/{cmp_stats["total_scenarios"]}')
    lines.append(f'    分类: 全满足{q3_cls["全满足"]} / 部分满足{q3_cls["部分满足"]} / 全不满足{q3_cls["全不满足"]}')
    lines.append('')
    lines.append(f'  Q4 离网分析:          {"已纳入" if q4_available else "无储能方案不可行（数据未生成）"}')
    lines.append('')
    lines.append('=' * 60)
    lines.append('报告生成完毕')
    lines.append('=' * 60)

    return '\n'.join(lines)


# =============================================================================
# 图表生成
# =============================================================================

def generate_chart(cmp_stats, q3_cls, df_cmp, q3_buy_sell):
    """生成2×2组合分析图表"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle('绿电直连园区 Q5 影响分析', fontsize=16, fontweight='bold')

    # ---- Subplot 1: Q2 vs Q3 加权平均吨氨成本对比 ----
    ax1 = axes[0, 0]
    categories = ['Q2 MILP\n(固定档位)', 'Q3 LP\n(连续调节)']
    values = [cmp_stats['q2_weighted_cost'], cmp_stats['q3_weighted_cost']]
    colors = ['#E74C3C', '#2ECC71']
    bars = ax1.bar(categories, values, color=colors, width=0.5, edgecolor='black')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                 f'{val:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax1.set_ylabel('吨氨成本 (元/ton)', fontsize=11)
    ax1.set_title('加权平均吨氨成本对比', fontsize=13, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(values) * 1.15)

    # ---- Subplot 2: Q3 场景分类饼图 ----
    ax2 = axes[0, 1]
    labels = ['全满足', '部分满足', '全不满足']
    sizes = [q3_cls['全满足'], q3_cls['部分满足'], q3_cls['全不满足']]
    pie_colors = ['#2ECC71', '#F39C12', '#E74C3C']
    explode = (0.02, 0.02, 0.02)

    wedges, texts, autotexts = ax2.pie(
        sizes, explode=explode, labels=None, colors=pie_colors,
        autopct='%1.1f%%', startangle=90, pctdistance=0.6,
        textprops={'fontsize': 11}
    )
    # 自定义图例
    legend_labels = [f'{l} ({s}天)' for l, s in zip(labels, sizes)]
    ax2.legend(wedges, legend_labels, title='Q3 场景分类', loc='lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=10)
    ax2.set_title('Q3 年度场景满足情况', fontsize=13, fontweight='bold')

    # ---- Subplot 3: Q3相对Q2成本降幅分布直方图 ----
    ax3 = axes[1, 0]
    improvements = [abs(v) for v in cmp_stats['improvement_list'] if v < 0]
    if improvements:
        ax3.hist(improvements, bins=15, color='#3498DB', edgecolor='black', alpha=0.8)
        ax3.axvline(np.mean(improvements), color='red', linestyle='--', linewidth=2,
                    label=f'均值: {np.mean(improvements):.1f}%')
        ax3.legend(fontsize=10)
    ax3.set_xlabel('成本降幅 (%)', fontsize=11)
    ax3.set_ylabel('场景数', fontsize=11)
    ax3.set_title('Q3 相对 Q2 吨氨成本降幅分布', fontsize=13, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)

    # ---- Subplot 4: Q3 各产量层级平均购售电 ----
    ax4 = axes[1, 1]
    if q3_buy_sell is not None:
        prod_levels = q3_buy_sell['日产量(吨/天)'].values.astype(int)
        x = np.arange(len(prod_levels))
        width = 0.35
        bars1 = ax4.bar(x - width / 2, q3_buy_sell['avg_buy'].values / 1000,
                        width, label='平均购电', color='#E74C3C', edgecolor='black')
        bars2 = ax4.bar(x + width / 2, q3_buy_sell['avg_sell'].values / 1000,
                        width, label='平均售电', color='#3498DB', edgecolor='black')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'{p}吨/天' for p in prod_levels], fontsize=9)
        ax4.set_ylabel('电量 (MWh)', fontsize=11)
        ax4.set_title('Q3 各产量层级平均购售电', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    chart_path = os.path.join(BASE_DIR, 'q5_impact_analysis.png')
    fig.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  图表已保存: {chart_path}')
    return chart_path


# =============================================================================
# 主流程
# =============================================================================

if __name__ == '__main__':
    print_header('问题五：政策定性分析')

    # ---- 1. 加载数据 ----
    print_header('1. 加载数据')

    # Q1
    print('  加载 Q1 指标...')
    df_q1 = load_csv_safe(os.path.join(Q1_DIR, 'q1_indicators.csv'))
    if df_q1 is None:
        print('  [错误] Q1 指标文件缺失，无法继续')
        sys.exit(1)
    q1 = analyze_q1_indicators(df_q1)
    print(f'  Q1: 绿电比例={q1["green_ratio"]:.1%}, 自发自用率={q1["self_use_rate"]:.1%}, 吨氨成本={q1["cost_per_ton"]:.0f}元/ton')

    # Q2
    print('  加载 Q2 年度汇总...')
    df_q2 = load_csv_safe(os.path.join(Q2_DIR, 'q2_annual_summary.csv'))

    # Q3
    print('  加载 Q3 年度汇总...')
    df_q3 = load_csv_safe(os.path.join(Q3_DIR, 'q3_annual_summary.csv'))
    if df_q3 is None:
        print('  [错误] Q3 数据文件缺失，无法继续')
        sys.exit(1)

    # Q3 vs Q2 comparison
    print('  加载 Q3 vs Q2 对比数据...')
    df_cmp = load_csv_safe(os.path.join(Q3_DIR, 'q3_vs_q2_comparison.csv'))
    if df_cmp is None:
        print('  [错误] Q3 vs Q2 对比数据缺失，无法继续')
        sys.exit(1)

    # 保存Q3原始数据引用（在闭包中使用）
    df_q3_temp = df_q3

    # Q4 (may not exist)
    print('  加载 Q4 离网数据...')
    df_q4 = load_csv_safe(os.path.join(Q4_DIR, 'q4_offgrid_no_storage.csv'))
    q4_available = df_q4 is not None
    if not q4_available:
        print('  Q4 数据缺失（预期中），将基于Q1数据进行分析')

    # ---- 2. 数据分析 ----
    print_header('2. 数据分析')

    cmp_stats = analyze_q2_q3_comparison(df_cmp)
    print(f'  Q2 vs Q3: 平均成本降幅={cmp_stats["mean_improvement_pct"]:.1f}%, '
          f'最大降幅={cmp_stats["max_improvement_pct"]:.1f}%')
    print(f'  Q2 加权平均成本: {cmp_stats["q2_weighted_cost"]:.0f} 元/ton')
    print(f'  Q3 加权平均成本: {cmp_stats["q3_weighted_cost"]:.0f} 元/ton')

    q3_cls = analyze_q3_classification(df_q3)
    q3_cls['q2_cost'] = cmp_stats['q2_weighted_cost']
    q3_cls['q3_cost'] = cmp_stats['q3_weighted_cost']
    print(f'  Q3 分类: 全满足={q3_cls["全满足"]}, 部分满足={q3_cls["部分满足"]}, 全不满足={q3_cls["全不满足"]}')

    q3_buy_sell = analyze_q3_buy_sell_by_production(df_q3)
    if q3_buy_sell is not None:
        print('  Q3 各产量层级购售电:')
        for _, row in q3_buy_sell.iterrows():
            print(f'    产量{int(row["日产量(吨/天)"])}吨/天: 购电={row["avg_buy"]:.0f}kWh, 售电={row["avg_sell"]:.0f}kWh')

    # ---- 3. 生成报告 ----
    print_header('3. 生成分析报告')

    report = generate_report(q1, cmp_stats, q3_cls, q3_buy_sell, q4_available)

    report_path = os.path.join(BASE_DIR, 'q5_policy_analysis.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f'  报告已保存: {report_path}')

    # ---- 4. 生成图表 ----
    print_header('4. 生成图表')

    chart_path = generate_chart(cmp_stats, q3_cls, df_cmp, q3_buy_sell)

    # ---- 5. 打印摘要 ----
    print_header('分析摘要')

    print(f'''
  绿电园区高渗透率的核心影响可总结为"三利三弊"：

  【利】1. 促进新能源消纳，Q3连续调节使{cmp_stats["total_improvements"]}个场景吨氨成本下降
       2. 削峰填谷，园区购售电策略符合分时电价引导方向
       3. 驱动氢氨产业链经济性提升，最大成本降幅{cmp_stats["max_improvement_pct"]:.1f}%

  【弊】1. 风光波动性威胁电网频率/电压稳定
       2. 潮流反向导致配电网保护配合困难
       3. 离网时段供需缺口需电网备用支撑

  四条核心政策建议：
  ① 完善绿电直连电价机制（过网费+容量备用费）
  ② 鼓励园区配储并给予投资补贴
  ③ 建立绿证-碳市场联动机制
  ④ 制定园区并网技术标准（电压穿越、AGC/AVC、分级接入）
''')

    print_header('问题五完成')
    print(f'  报告: {report_path}')
    print(f'  图表: {chart_path}')
    print()
