"""
visualization - 统一可视化工具包

提供中文字体支持、统一图表样式、常用绘图辅助函数。
"""

from .plotting import (
    # 字体
    setup_chinese_font,
    get_current_font,
    # 配色
    COLOR_SCHEMES,
    get_color_scheme,
    # 图窗工厂
    create_figure,
    save_figure,
    # 装饰
    style_axis,
    add_value_labels,
    add_legend_outside,
    set_hour_xticks,
    # 图表模板
    plot_power_stacked_bar,
    plot_comparison_line,
    plot_boxplot,
    plot_heatmap,
    plot_sankey_style_bar,
    # 论文
    apply_paper_style,
)

__all__ = [
    'setup_chinese_font',
    'get_current_font',
    'COLOR_SCHEMES',
    'get_color_scheme',
    'create_figure',
    'save_figure',
    'style_axis',
    'add_value_labels',
    'add_legend_outside',
    'set_hour_xticks',
    'plot_power_stacked_bar',
    'plot_comparison_line',
    'plot_boxplot',
    'plot_heatmap',
    'plot_sankey_style_bar',
    'apply_paper_style',
]
