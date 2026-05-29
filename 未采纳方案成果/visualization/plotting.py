"""
统一绘图工具模块 — Unified Plotting Utilities
==============================================
为 Q1-Q5 可视化任务提供所有图表绘制函数。
集中管理中文字体配置、颜色方案、图窗布局与持久化。

特性:
    - 中文字体自动检测与配置，避免"豆腐块/乱码" (导入即生效)
    - 五题独立配色方案 (Q1: 青 teal, Q2: 蓝 blue, Q3: 橙 orange, Q4: 绿 green, Q5: 紫 purple)
    - 工厂函数 create_figure / save_figure，统一图窗样式
    - 常用图表快捷函数 (堆叠柱状图、双曲线对比、箱线图、热力图等)

用法:
    from visualization.plotting import create_figure, save_figure, get_color_scheme
    fig, ax = create_figure(figsize=(14, 6))
    colors = get_color_scheme(1)  # Q1 配色
    ax.plot(x, y, color=colors['primary'], label='风电功率 (MW)')
    save_figure(fig, 'q1_power_balance.png')
"""

import os
import warnings
import logging

import matplotlib
# ── 非交互式后端 (必须在 import pyplot 之前) ─────────────────
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np


# ╔══════════════════════════════════════════════════════════════╗
# ║  1. 中文字体配置 — 导入时自动执行                           ║
# ╚══════════════════════════════════════════════════════════════╝

def setup_chinese_font() -> str:
    """
    自动检测并配置 matplotlib 中文字体。

    按优先级依次尝试:
        SimHei → Microsoft YaHei → WenQuanYi Micro Hei →
        Noto Sans CJK SC → STHeiti → Arial Unicode MS → DejaVu Sans (fallback)

    同时设置 axes.unicode_minus = False 以正确显示负号。

    Returns:
        str: 实际生效的字体名称

    Note:
        此函数在模块导入时自动调用，无需手动执行。
    """
    chinese_fonts = [
        'SimHei',                # Windows 黑体 (最常见)
        'Microsoft YaHei',       # Windows 微软雅黑
        'WenQuanYi Micro Hei',   # Linux 文泉驿微米黑
        'Noto Sans CJK SC',      # Linux Noto 简体中文
        'STHeiti',               # macOS 华文黑体
        'Arial Unicode MS',      # macOS Arial 全字符集
    ]

    available = {f.name for f in fm.fontManager.ttflist}

    for font in chinese_fonts:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return font

    # 最终回退 — 中文字体不可用时仍能正常绘图 (但中文显示为方框)
    warnings.warn(
        "未检测到任何中文字体，中文标签可能显示为方框。"
        "请安装 SimHei / Microsoft YaHei / WenQuanYi Micro Hei 等字体。"
    )
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return 'DejaVu Sans'


# ── 导入时自动配置 ─────────────────────────────────────────────
_CURRENT_FONT = setup_chinese_font()

# 全局 rcParams 默认值 (在字体配置之后设置)
plt.rcParams.update({
    'figure.dpi':         100,
    'savefig.dpi':        150,
    'font.size':          11,
    'axes.titlesize':     14,
    'axes.labelsize':     12,
    'xtick.labelsize':    10,
    'ytick.labelsize':    10,
    'legend.fontsize':    10,
    'lines.linewidth':    1.8,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'axes.facecolor':     '#FAFAFA',
})


def get_current_font() -> str:
    """返回当前生效的字体名称。"""
    return _CURRENT_FONT


# ╔══════════════════════════════════════════════════════════════╗
# ║  2. 配色方案 (Color Schemes)                                ║
# ╚══════════════════════════════════════════════════════════════╝

COLOR_SCHEMES = {
    # ── Q1: 典型风光场景运行分析 (Teal / 青绿) ──────────────────
    'Q1': {
        'primary':    '#00897B',   # 主色调 — 标题、主曲线
        'secondary':  '#80CBC4',   # 辅助色 — 背景填充
        'accent':     '#F57C00',   # 强调色 — 标注、高亮
        # 能源流分类色
        'wind':       '#42A5F5',   # 风电 — 蓝
        'pv':         '#FFCA28',   # 光伏 — 黄
        'load':       '#EF5350',   # 常规负荷 — 红
        'grid_buy':   '#AB47BC',   # 网购电 — 紫
        'grid_sell':  '#66BB6A',   # 上网电 — 绿
        'h2':         '#EC407A',   # 制氢 — 粉
        'nh3':        '#8D6E63',   # 合成氨 — 棕
    },

    # ── Q2: 离散制氨调节优化 (Blue / 蓝) ────────────────────────
    'Q2': {
        'primary':    '#1565C0',
        'secondary':  '#90CAF9',
        'accent':     '#FF7043',
        # 5 种日产量方案色 (由浅到深)
        'd36':        '#BBDEFB',
        'd45':        '#64B5F6',
        'd54':        '#1E88E5',
        'd63':        '#1565C0',
        'd72':        '#0D47A1',
        # 箱线图配色
        'box_fill':   '#42A5F5',
        'box_edge':   '#1565C0',
        'flier':      '#FF7043',
    },

    # ── Q3: 连续制氨调节运行 (Orange / 橙) ──────────────────────
    'Q3': {
        'primary':    '#E65100',
        'secondary':  '#FFB74D',
        'accent':     '#29B6F6',
        # Q3 vs Q2 对比色
        'q3_color':   '#FF6F00',   # Q3 自身曲线
        'q2_color':   '#42A5F5',   # Q2 对比曲线
        # 堆叠分量色
        'wind':       '#4FC3F7',
        'pv':         '#FFD54F',
        'buy':        '#EF5350',
        'base_load':  '#BDBDBD',
        'alk':        '#BA68C8',
        'nh3':        '#A1887F',
        'sell':       '#81C784',
    },

    # ── Q4: 离网运行与储能配置 (Green / 绿) ─────────────────────
    'Q4': {
        'primary':      '#2E7D32',
        'secondary':    '#A5D6A7',
        'accent':       '#FF5252',
        'no_storage':   '#FF8A80',   # 无储能
        'with_storage': '#69F0AE',   # 有储能
        'soc_low':      '#FFEB3B',   # SOC 低区间
        'soc_high':     '#4CAF50',   # SOC 高区间
        'curtailment':  '#FF5252',   # 弃电
        'heatmap_cmap': 'YlGn',      # 热力图 colormap
    },

    # ── Q5: 宏观政策与敏感性分析 (Purple / 紫) ──────────────────
    'Q5': {
        'primary':      '#6A1B9A',
        'secondary':    '#CE93D8',
        'accent':       '#00BCD4',
        'alpha_curve':  '#7B1FA2',   # 装机倍数 α 曲线
        'policy_a':     '#E040FB',   # 政策方案 A
        'policy_b':     '#00BCD4',   # 政策方案 B
        'surface_cmap': 'plasma',    # 三维曲面 colormap
    },

    # ── 通用语义色 (所有题目公用) ────────────────────────────────
    '_semantic': {
        'pass':    '#4CAF50',   # 合规
        'fail':    '#F44336',   # 不合规
        'warn':    '#FF9800',   # 警告
        'info':    '#2196F3',   # 信息
        'neutral': '#9E9E9E',   # 中性
    },
}


def get_color_scheme(question_num: int):
    """
    返回指定题号的配色字典。

    Args:
        question_num: 题号 (1, 2, 3, 4, 5)

    Returns:
        dict: 对应的 COLOR_SCHEMES['QX']，不存在时回退到 Q1

    Examples:
        >>> colors = get_color_scheme(1)
        >>> colors['primary']
        '#00897B'
        >>> colors['wind']
        '#42A5F5'

        >>> q3 = get_color_scheme(3)
        >>> ax.plot(x, y3, color=q3['q3_color'], label='Q3 连续调节')
        >>> ax.plot(x, y2, color=q3['q2_color'], label='Q2 离散调节')
    """
    key = f'Q{question_num}'
    return COLOR_SCHEMES.get(key, COLOR_SCHEMES['Q1'])


# ╔══════════════════════════════════════════════════════════════╗
# ║  3. 图窗工厂与持久化                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def create_figure(nrows: int = 1, ncols: int = 1,
                  figsize: tuple = (12, 6)) -> tuple:
    """
    图窗工厂 — 统一创建带默认样式的 Figure 和 Axes。

    Args:
        nrows:   子图行数
        ncols:   子图列数
        figsize: 图窗尺寸 (宽, 高)，单位英寸

    Returns:
        tuple: (fig, axes)
            - 单子图时 axes 为单个 Axes 对象
            - 多子图时 axes 为 ndarray

    Examples:
        >>> fig, ax = create_figure(figsize=(14, 7))
        >>> fig, axes = create_figure(2, 2, figsize=(16, 12))
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def save_figure(fig, filename: str, dpi: int = 150,
                results_dir: str = 'results') -> str:
    """
    统一保存图窗 — 自动创建目录，标准化设置。

    Args:
        fig:         matplotlib.figure.Figure 对象
        filename:    文件名 (支持子目录，如 'q1/power_balance.png')
        dpi:         输出分辨率 (默认 150 dpi)
        results_dir: 结果根目录 (默认 'results/')

    Returns:
        str: 保存的绝对路径

    Examples:
        >>> path = save_figure(fig, 'q1_energy_flow.png')
        >>> path = save_figure(fig, 'q4/heatmap.png', dpi=300)
    """
    dir_path = os.path.join(results_dir, os.path.dirname(filename))
    os.makedirs(dir_path, exist_ok=True)

    save_path = os.path.join(results_dir, filename)
    fig.savefig(
        save_path,
        dpi=dpi,
        bbox_inches='tight',
        pad_inches=0.2,
    )
    plt.close(fig)
    return os.path.abspath(save_path)


# ╔══════════════════════════════════════════════════════════════╗
# ║  4. 图表装饰快捷函数                                        ║
# ╚══════════════════════════════════════════════════════════════╝

def style_axis(ax, title: str = None, xlabel: str = None,
               ylabel: str = None, **title_kw):
    """
    快速设置轴标签和标题 (支持中文)。

    Args:
        ax:       matplotlib.axes.Axes 对象
        title:    图表标题
        xlabel:   X 轴标签
        ylabel:   Y 轴标签
        **title_kw: 标题字体参数字典 (如 fontsize=16, fontweight='bold')
    """
    if title:
        kw = {'fontsize': 14, 'fontweight': 'bold'}
        kw.update(title_kw)
        ax.set_title(title, **kw)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def set_hour_xticks(ax, hours=None):
    """
    设置 X 轴为 0-23 小时刻度 (Q1-Q4 高频操作)。

    Args:
        ax:    matplotlib.axes.Axes 对象
        hours: 自定义小时列表 (默认 range(24))
    """
    if hours is None:
        hours = np.arange(24)
    ax.set_xticks(hours)
    ax.set_xticklabels([f'{h}:00' for h in hours], rotation=45, ha='right')
    ax.set_xlabel('时间 (h)')


def add_value_labels(ax, bars, fmt: str = '{:.0f}',
                     offset: float = 0.5, **text_kw):
    """
    为柱状图在柱顶添加数值标签。

    Args:
        ax:       matplotlib.axes.Axes 对象
        bars:     ax.bar() / ax.barh() 返回的 BarContainer
        fmt:      格式化字符串
        offset:   标签相对柱顶的偏移
        **text_kw: 透传给 ax.text() 的参数
    """
    default_kw = {'ha': 'center', 'va': 'bottom', 'fontsize': 9}
    default_kw.update(text_kw)

    for bar in bars:
        height = bar.get_height()
        if height >= 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                fmt.format(height),
                **default_kw,
            )


def add_legend_outside(ax, ncol: int = 1, **kwargs):
    """
    将图例放置在图框外侧右上方。

    Args:
        ax:    matplotlib.axes.Axes 对象
        ncol:  图例列数
        **kwargs: 透传给 ax.legend()
    """
    default_kw = {
        'loc':           'upper left',
        'bbox_to_anchor': (1.01, 1.0),
        'framealpha':    0.85,
        'ncol':          ncol,
    }
    default_kw.update(kwargs)
    ax.legend(**default_kw)


# ╔══════════════════════════════════════════════════════════════╗
# ║  5. 常用图表模板                                            ║
# ╚══════════════════════════════════════════════════════════════╝

def plot_power_stacked_bar(figsize: tuple = (14, 7),
                           title: str = '功率平衡堆叠柱状图') -> tuple:
    """
    快速创建正/负半轴堆叠柱状图画布 — 适用于 Q1/Q3 源-荷双侧功率平衡图。

    返回预先配置好 x=0 水平线和坐标轴的 (fig, ax)。

    Returns:
        tuple: (fig, ax)
    """
    fig, ax = create_figure(figsize=figsize)
    ax.axhline(y=0, color='black', linewidth=1.2, zorder=0)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('小时')
    ax.set_ylabel('功率 (MW)')
    ax.set_xticks(np.arange(24))
    ax.grid(axis='y', alpha=0.3)
    return fig, ax


def plot_comparison_line(figsize: tuple = (12, 6),
                         title: str = '成本分布对比',
                         xlabel: str = None,
                         ylabel: str = None) -> tuple:
    """
    快速创建双曲线对比图画布 — 适用于 Q3 vs Q2 成本分布等。

    Returns:
        tuple: (fig, ax)
    """
    fig, ax = create_figure(figsize=figsize)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    return fig, ax


def plot_boxplot(data: list, labels: list,
                 title: str = '指标分布箱线图',
                 ylabel: str = None,
                 figsize: tuple = (12, 6),
                 colors: list = None) -> tuple:
    """
    箱线图快捷函数 — 适用于 Q2 多场景指标分布。

    Args:
        data:    每组数据 (list of array-like)
        labels:  每组标签 (如 ['36吨/天', '45吨/天', ...])
        title:   图表标题
        ylabel:  Y 轴标签
        figsize: 图窗尺寸
        colors:  各组箱体填充色 (长度需与 data 一致)

    Returns:
        tuple: (fig, ax)
    """
    fig, ax = create_figure(figsize=figsize)
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)

    if colors and len(colors) == len(data):
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis='y', alpha=0.3)
    return fig, ax


def plot_heatmap(matrix: np.ndarray, x_labels: list, y_labels: list,
                 title: str = '敏感性热力图',
                 xlabel: str = None, ylabel: str = None,
                 cmap: str = 'YlGn',
                 annotate: bool = True, fmt: str = '.1f',
                 figsize: tuple = (10, 8)) -> tuple:
    """
    热力图快捷函数 — 适用于 Q4 装机容量扫描等二维敏感性分析。

    Args:
        matrix:    (n_rows, n_cols) 数值矩阵
        x_labels:  列标签
        y_labels:  行标签
        title:    图表标题
        xlabel:   X 轴标签
        ylabel:   Y 轴标签
        cmap:     colormap 名称
        annotate: 是否标注数值
        fmt:      数值格式化字符串
        figsize:  图窗尺寸

    Returns:
        tuple: (fig, ax)
    """
    fig, ax = create_figure(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, aspect='auto', origin='lower')

    if x_labels:
        ax.set_xticks(np.arange(len(x_labels)))
        ax.set_xticklabels(x_labels, rotation=45, ha='right')
    if y_labels:
        ax.set_yticks(np.arange(len(y_labels)))
        ax.set_yticklabels(y_labels)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    if annotate:
        mean_val = np.nanmean(matrix)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                text_color = 'white' if val > mean_val else 'black'
                ax.text(j, i, f'{val:{fmt}}',
                        ha='center', va='center',
                        fontsize=9, color=text_color)

    fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    return fig, ax


def plot_sankey_style_bar(labels: list, values: list,
                          colors: list = None,
                          title: str = '能量流桑基图',
                          xlabel: str = '能量 (MWh)',
                          figsize: tuple = (14, 8)) -> tuple:
    """
    桑基风格水平柱状图 — 适用于 Q1 能源流可视化。

    正值向右延伸 (源侧)，负值也可展示为向左的负荷侧。

    Args:
        labels:  各节点名称
        values:  各节点数值
        colors:  各节点颜色 (可选)
        title:   图表标题
        xlabel:  X 轴标签
        figsize: 图窗尺寸

    Returns:
        tuple: (fig, ax)
    """
    fig, ax = create_figure(figsize=figsize)
    y_pos = np.arange(len(labels))

    # 分离正负值分别绘制
    for i, (v, lb) in enumerate(zip(values, labels)):
        clr = colors[i] if colors else None
        ax.barh(y_pos[i], abs(v), color=clr, alpha=0.85,
                edgecolor='white', linewidth=0.5)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linewidth=1.0)
    ax.grid(axis='x', alpha=0.25)

    # 数值标注
    max_val = max(abs(v) for v in values) if values else 1
    for i, v in enumerate(values):
        if v >= 0:
            ax.text(v + max_val * 0.015, y_pos[i], f'{v:.0f}',
                    va='center', fontsize=10)
        else:
            ax.text(v - max_val * 0.015, y_pos[i], f'{v:.0f}',
                    va='center', ha='right', fontsize=10)

    return fig, ax


# ╔══════════════════════════════════════════════════════════════╗
# ║  6. 论文排版辅助                                            ║
# ╚══════════════════════════════════════════════════════════════╝

def apply_paper_style(fig):
    """
    应用学术论文图表风格 — 去除顶部/右侧边框、加粗轴线。

    应在数据绘制完成后、save_figure 前调用。

    Args:
        fig: matplotlib.figure.Figure 对象
    """
    for ax in fig.axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        ax.tick_params(width=1.2)


# ╔══════════════════════════════════════════════════════════════╗
# ║  7. 模块公开接口                                            ║
# ╚══════════════════════════════════════════════════════════════╝

__all__ = [
    # 字体
    'setup_chinese_font',
    'get_current_font',
    # 配色
    'COLOR_SCHEMES',
    'get_color_scheme',
    # 图窗工厂
    'create_figure',
    'save_figure',
    # 装饰
    'style_axis',
    'add_value_labels',
    'add_legend_outside',
    'set_hour_xticks',
    # 图表模板
    'plot_power_stacked_bar',
    'plot_comparison_line',
    'plot_boxplot',
    'plot_heatmap',
    'plot_sankey_style_bar',
    # 论文
    'apply_paper_style',
]

if __name__ == '__main__':
    print(f"绘图工具模块已加载")
    print(f"  当前中文字体: {_CURRENT_FONT}")
    print(f"  可用配色方案: {list(COLOR_SCHEMES.keys())}")
    print(f"  公开函数数量: {len(__all__)}")
    print(f"\n用法示例:")
    print(f"  from visualization.plotting import create_figure, save_figure, get_color_scheme")
    print(f"  fig, ax = create_figure(figsize=(14, 6))")
    print(f"  colors = get_color_scheme(1)")
    print(f"  ax.plot(x, y, color=colors['primary'])")
    print(f"  save_figure(fig, 'my_chart.png')")
