"""
中文字体配置模块 — Chinese Font Setup
=====================================
为所有可视化脚本提供统一的中文字体入口。
内部委托给 plotting.py 的 setup_chinese_font()。

用法:
    from visualization.chinese_font import setup_chinese_font
    setup_chinese_font()
"""

from visualization.plotting import setup_chinese_font as _setup

def setup_chinese_font():
    """配置 matplotlib 中文字体 (委托给 plotting.setup_chinese_font)。"""
    return _setup()

__all__ = ['setup_chinese_font']
