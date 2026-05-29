"""
sensitivity/utils.py — OAT (One-at-a-Time) 参数灵敏度分析工具函数

提供扰动点生成、单参数OAT扫描、结果收集与保存等基础框架。
供 Q1-Q4 所有灵敏度分析脚本统一调用。

依赖: numpy, csv, json, 以及 sensitivity.config.
      通过 sys.path 可访问根目录的 config.py 和 utils.py。
"""

import os
import csv
import json
import numpy as np


# =============================================================================
# 扰动点生成
# =============================================================================

def generate_oat_points(param_config: dict) -> list:
    """为单个参数生成 OAT 扰动点列表。

    在基线值 ±(PERTURBATION_RANGE × baseline) 范围内,
    均匀生成 PERTURBATION_POINTS 个点。

    Args:
        param_config: 参数字典, 至少包含 {"baseline": float}.
                      可从 sensitivity.config.PARAMETERS[i] 获取。

    Returns:
        list of float, 包含 PERTURBATION_POINTS 个扰动值 (含 baseline).

    Example:
        >>> from sensitivity.config import PARAMETERS, PERTURBATION_POINTS
        >>> pts = generate_oat_points(PARAMETERS[0])
        >>> len(pts) == PERTURBATION_POINTS
        True
    """
    from sensitivity.config import PERTURBATION_RANGE, PERTURBATION_POINTS

    baseline = float(param_config["baseline"])
    delta = baseline * PERTURBATION_RANGE
    return np.linspace(baseline - delta, baseline + delta, PERTURBATION_POINTS).tolist()


# =============================================================================
# 相对变化
# =============================================================================

def relative_change(baseline_val: float, perturbed_val: float) -> float:
    """计算扰动值相对于基线值的百分比变化。

    Args:
        baseline_val:  基线值
        perturbed_val: 扰动后值

    Returns:
        float: (perturbed - baseline) / baseline × 100 (百分比)
               若 baseline_val == 0, 返回 0.0 避免除零。

    Example:
        >>> relative_change(100, 120)
        20.0
        >>> relative_change(100, 80)
        -20.0
    """
    if baseline_val == 0:
        return 0.0
    return (perturbed_val - baseline_val) / baseline_val * 100.0


# =============================================================================
# 修改/恢复参数值 (操纵根 config 模块)
# =============================================================================

def _patch_root_config(param_name: str, new_value: float):
    """临时修改根目录 config.py 模块中的参数值。

    Args:
        param_name: 根 config 中的常量名 (e.g. "WIND_LCOE")
        new_value:  新数值
    """
    import config as root_config
    setattr(root_config, param_name, new_value)


def _get_root_config_value(param_name: str) -> float:
    """读取根目录 config.py 模块中的参数值。"""
    import config as root_config
    return getattr(root_config, param_name)


_FORCE_REIMPORT_PARAMS = {
    "nh3_load_min": "NH3_LOAD_MIN",
    "wind_lcoe": "WIND_LCOE",
    "solar_lcoe": "SOLAR_LCOE",
    "storage_investment": "STORAGE_INVESTMENT",
    "alk_h2_per_mw": "ALK_H2_PER_MW",
    "pem_h2_per_mw": "PEM_H2_PER_MW",
    "alkel_om": "ALKEL_OM",
    "pemel_om": "PEMEL_OM",
    "tou_price_scale": None,       # 特殊处理: 缩放分时电价
    "green_penalty_self_use": None,  # 惩罚参数不存在于根config, 直接传值
    "green_penalty_green_rate": None,
    "green_penalty_feed_rate": None,
    "scale_factor": None,           # 产能规模 → PRODUCTION_LEVELS 变换
}


def apply_perturbation(param_name: str, perturbed_value: float):
    """将扰动后的参数值应用到根 config 模块 (或返回给调用者)。

    处理三类参数:
      A) 直接映射到根 config 常量 (e.g. WIND_LCOE)
      B) 惩罚参数 — 不修改根 config, 返回 (name, value) 供调用者传入 run_func
      C) 特殊参数 — 分时电价系数、产能规模系数需额外变换

    Args:
        param_name:    参数标识名 (来自 PARAMETERS[i]["name"])
        perturbed_value: 扰动后的值

    Returns:
        dict: {"param_name": str, "perturbed_value": float,
               "root_attr": str or None, "original_value": float or None}
              - root_attr: 根 config 中被修改的属性名 (None 表示未修改)
              - original_value: 修改前的值 (None 表示未修改)
    """
    import config as root_config

    mapping = _FORCE_REIMPORT_PARAMS
    root_attr = mapping.get(param_name)

    result = {
        "param_name": param_name,
        "perturbed_value": perturbed_value,
        "root_attr": None,
        "original_value": None,
    }

    if root_attr is not None:
        # A 类: 直接映射
        original = getattr(root_config, root_attr)
        setattr(root_config, root_attr, perturbed_value)
        result["root_attr"] = root_attr
        result["original_value"] = original

    # B/C 类: 返回扰动值, 由调用者处理
    return result


def restore_perturbation(patch_info: dict):
    """恢复被扰动修改的根 config 参数。

    Args:
        patch_info: apply_perturbation() 返回的字典
    """
    import config as root_config

    root_attr = patch_info.get("root_attr")
    original = patch_info.get("original_value")
    if root_attr is not None and original is not None:
        setattr(root_config, root_attr, original)


# =============================================================================
# OAT 单参数扫描
# =============================================================================

def oat_sweep(
    param: dict,
    run_func,
    baseline_result: dict,
    verbose: bool = True,
) -> list:
    """对单个参数执行 OAT 扫描。

    对参数的每个扰动点:
      1. 应用扰动 (修改根 config 或传值)
      2. 调用 run_func(perturbed_value=..., param_name=...) 或 run_func(**kwargs)
      3. 记录每个指标的变化
      4. 恢复基线

    Args:
        param:            参数字典 (来自 PARAMETERS[i])
        run_func:         单次运行函数.
                          签名: run_func(perturbed_param, perturbed_value) -> dict
                          返回: {metric_name: value, ...}
        baseline_result:  基线运行结果 dict, 用于计算相对变化
        verbose:          是否打印进度

    Returns:
        list of dict, 每个元素:
          {
            "param_name": str,
            "param_label": str,
            "perturbed_value": float,
            "perturbation_pct": float,  # 相对基线变化 %
            "metrics": {metric: value, ...},
            "relative_changes": {metric: pct_change, ...},
          }
    """
    from sensitivity.config import PERTURBATION_POINTS

    param_name = param["name"]
    param_label = param["label"]
    points = generate_oat_points(param)

    results = []
    for i, pv in enumerate(points):
        pct = relative_change(param["baseline"], pv)
        if verbose:
            print(f"  [{param_label}] point {i+1}/{PERTURBATION_POINTS}: "
                  f"val={pv:.4f} (Δ{pct:+.1f}%)")

        # 应用扰动 → 运行 → 恢复
        patch_info = apply_perturbation(param_name, pv)
        try:
            metrics = run_func(param_name, pv)
        finally:
            restore_perturbation(patch_info)

        # 计算各指标相对变化
        rel_changes = {}
        for metric, val in metrics.items():
            bl_val = baseline_result.get(metric, None)
            if bl_val is not None and bl_val != 0 and not np.isinf(bl_val):
                rel_changes[metric] = relative_change(bl_val, val)
            else:
                rel_changes[metric] = None

        results.append({
            "param_name": param_name,
            "param_label": param_label,
            "perturbed_value": pv,
            "perturbation_pct": pct,
            "metrics": metrics,
            "relative_changes": rel_changes,
        })

    return results


# =============================================================================
# 结果保存
# =============================================================================

def save_oat_results(results: list, filepath: str):
    """将 OAT 扫描结果保存为 CSV 文件。

    每行一个扰动点, 列: param_name, param_label, perturbed_value,
    perturbation_pct, 以及各指标的值和相对变化。

    Args:
        results:   oat_sweep() 返回的列表 (可拼接多个参数的扫描结果)
        filepath:  输出 CSV 路径 (.csv)
    """
    if not results:
        print("[WARN] save_oat_results: empty results, skipping.")
        return

    # 收集所有出现的指标名
    all_metrics = set()
    for r in results:
        all_metrics.update(r["metrics"].keys())

    sorted_metrics = sorted(all_metrics)

    # 构建表头
    header = ["param_name", "param_label", "perturbed_value", "perturbation_pct"]
    for m in sorted_metrics:
        header.append(f"metric_{m}")
        header.append(f"rel_change_{m}")

    rows = []
    for r in results:
        row = [
            r["param_name"],
            r["param_label"],
            r["perturbed_value"],
            r["perturbation_pct"],
        ]
        for m in sorted_metrics:
            row.append(r["metrics"].get(m, None))
            row.append(r["relative_changes"].get(m, None))
        rows.append(row)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"[OK] OAT results saved to: {filepath}  ({len(rows)} rows × {len(header)} cols)")


def save_oat_results_json(results: list, filepath: str):
    """将 OAT 扫描结果保存为 JSON 文件 (结构化, 含元数据)。

    Args:
        results:   oat_sweep() 返回的列表
        filepath:  输出 JSON 路径 (.json)
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"[OK] OAT results (JSON) saved to: {filepath}")


# =============================================================================
# 便捷: 汇总所有参数扫描
# =============================================================================

def run_all_oat(
    parameters: list,
    run_func,
    baseline_result: dict,
    typical_scenarios: "list | None" = None,
    verbose: bool = True,
) -> list:
    """对所有参数依次执行 OAT 扫描, 返回合并结果列表。

    Args:
        parameters:         PARAMETERS 列表
        run_func:           单次运行函数 run_func(param_name, perturbed_value) -> dict
        baseline_result:    基线结果 dict
        typical_scenarios:  可选, 场景列表 (用于日志)
        verbose:            是否打印进度

    Returns:
        list of dict, 所有参数扫描结果的拼接
    """
    all_results = []
    n_params = len(parameters)

    for i, param in enumerate(parameters):
        if verbose:
            print(f"\n[OAT {i+1}/{n_params}] Parameter: {param['label']} "
                  f"(baseline={param['baseline']}{param.get('unit','')})")
        param_results = oat_sweep(param, run_func, baseline_result, verbose=verbose)
        all_results.extend(param_results)

    return all_results
