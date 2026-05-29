# -*- coding: utf-8 -*-
"""
q2_sensitivity.py — OAT参数灵敏度分析 for Q2 (discrete on/off MILP, 72t/d)

Reuses q2_milp.py's build_and_solve_milp() function.
Perturbs ≥6 parameters one-at-a-time across 8 typical wind+solar scenarios.
Fixed production level = 72 tons/day.

Output: sensitivity/data/q2_oat_results.csv
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import numpy as np

import config as root_config
import utils
from q2.q2_milp import build_and_solve_milp, EL_OM_BLENDED as _q2_el_om_blended

from sensitivity.config import TYPICAL_SCENARIOS, PERTURBATION_POINTS, generate_oat_points

FIXED_OUTPUT = 72  # ton/day

# =============================================================================
# Q2-specific parameter definitions
# =============================================================================
# Baselines taken from root config (not sensitivity/config.py's PARAMETERS,
# which has a stale green_penalty_feed_rate=0.4 vs root's 1.5).

Q2_PARAMS = [
    {
        "name": "green_penalty_self_use",
        "label": u"\u03bb\u2081 \u81ea\u7528\u7387\u60e9\u7f5a",  # λ₁ 自用率惩罚
        "baseline": root_config.GREEN_PENALTY_SELF_USE,              # 0.5
        "unit": u"\u00a5/kWh",
        "range": [0.2, 1.0],
    },
    {
        "name": "green_penalty_green_rate",
        "label": u"\u03bb\u2082 \u7eff\u7535\u7387\u60e9\u7f5a",  # λ₂ 绿电率惩罚
        "baseline": root_config.GREEN_PENALTY_GREEN_RATE,            # 0.3
        "unit": u"\u00a5/kWh",
        "range": [0.1, 0.6],
    },
    {
        "name": "green_penalty_feed_rate",
        "label": u"\u03bb\u2083 \u4e0a\u7f51\u7387\u60e9\u7f5a",  # λ₃ 上网率惩罚
        "baseline": root_config.GREEN_PENALTY_FEED_RATE,             # 1.5
        "unit": u"\u00a5/kWh",
        "range": [0.5, 2.5],
    },
    {
        "name": "tou_price_scale",
        "label": u"\u5206\u65f6\u7535\u4ef7\u7f29\u653e",          # 分时电价缩放
        "baseline": 1.0,
        "unit": "scale",
        "range": [0.7, 1.3],
    },
    {
        "name": "wind_lcoe",
        "label": u"\u98ce\u7535LCOE",                                # 风电LCOE
        "baseline": root_config.WIND_LCOE,                           # 0.15
        "unit": u"\u00a5/kWh",
        "range": [0.075, 0.225],
    },
    {
        "name": "solar_lcoe",
        "label": u"\u5149\u4f0fLCOE",                                # 光伏LCOE
        "baseline": root_config.SOLAR_LCOE,                          # 0.12
        "unit": u"\u00a5/kWh",
        "range": [0.06, 0.18],
    },
    {
        "name": "el_om_blended",
        "label": u"\u7535\u89e3\u8fd0\u7ef4(\u6df7\u5408)",         # 电解运维(混合)
        "baseline": (root_config.ALKEL_OM + root_config.PEMEL_OM) / 2,  # 0.125
        "unit": u"\u00a5/kWh",
        "range": [0.06, 0.19],
    },
]

# =============================================================================
# Parameter override / restore helpers
# =============================================================================

# Capture originals at module load time (before any patching)
_ORIG_GET_PRICE = root_config.get_price
_ORIG_CONFIG_GP_SELF = root_config.GREEN_PENALTY_SELF_USE
_ORIG_CONFIG_GP_GREEN = root_config.GREEN_PENALTY_GREEN_RATE
_ORIG_CONFIG_GP_FEED = root_config.GREEN_PENALTY_FEED_RATE
_ORIG_CONFIG_WIND_LCOE = root_config.WIND_LCOE
_ORIG_CONFIG_SOLAR_LCOE = root_config.SOLAR_LCOE
_ORIG_UTILS_WIND_LCOE = utils.WIND_LCOE
_ORIG_UTILS_SOLAR_LCOE = utils.SOLAR_LCOE
_ORIG_Q2_EL_OM = _q2_el_om_blended


def _apply_q2_override(param_name, perturbed_value):
    """Apply Q2 parameter override across all relevant modules.

    Returns a dict with undo information for _restore_q2_override().
    """
    undo = {"param_name": param_name}

    if param_name == "green_penalty_self_use":
        undo["orig"] = root_config.GREEN_PENALTY_SELF_USE
        root_config.GREEN_PENALTY_SELF_USE = perturbed_value

    elif param_name == "green_penalty_green_rate":
        undo["orig"] = root_config.GREEN_PENALTY_GREEN_RATE
        root_config.GREEN_PENALTY_GREEN_RATE = perturbed_value

    elif param_name == "green_penalty_feed_rate":
        undo["orig"] = root_config.GREEN_PENALTY_FEED_RATE
        root_config.GREEN_PENALTY_FEED_RATE = perturbed_value

    elif param_name == "tou_price_scale":
        # Replace get_price with a scaled wrapper
        undo["orig_func"] = root_config.get_price
        scale = perturbed_value
        root_config.get_price = lambda h, s=scale: _ORIG_GET_PRICE(h) * s

    elif param_name == "wind_lcoe":
        undo["cfg"] = root_config.WIND_LCOE
        undo["utl"] = utils.WIND_LCOE
        root_config.WIND_LCOE = perturbed_value
        utils.WIND_LCOE = perturbed_value

    elif param_name == "solar_lcoe":
        undo["cfg"] = root_config.SOLAR_LCOE
        undo["utl"] = utils.SOLAR_LCOE
        root_config.SOLAR_LCOE = perturbed_value
        utils.SOLAR_LCOE = perturbed_value

    elif param_name == "el_om_blended":
        import q2.q2_milp as q2m
        undo["orig"] = q2m.EL_OM_BLENDED
        q2m.EL_OM_BLENDED = perturbed_value

    return undo


def _restore_q2_override(undo):
    """Restore all module attributes modified by _apply_q2_override."""
    pn = undo["param_name"]

    if pn == "green_penalty_self_use":
        root_config.GREEN_PENALTY_SELF_USE = undo["orig"]
    elif pn == "green_penalty_green_rate":
        root_config.GREEN_PENALTY_GREEN_RATE = undo["orig"]
    elif pn == "green_penalty_feed_rate":
        root_config.GREEN_PENALTY_FEED_RATE = undo["orig"]
    elif pn == "tou_price_scale":
        root_config.get_price = undo["orig_func"]
    elif pn == "wind_lcoe":
        root_config.WIND_LCOE = undo["cfg"]
        utils.WIND_LCOE = undo["utl"]
    elif pn == "solar_lcoe":
        root_config.SOLAR_LCOE = undo["cfg"]
        utils.SOLAR_LCOE = undo["utl"]
    elif pn == "el_om_blended":
        import q2.q2_milp as q2m
        q2m.EL_OM_BLENDED = undo["orig"]


# =============================================================================
# Metric extraction
# =============================================================================

def _extract_metrics(result):
    """Extract key metrics from a MILP result dict into a flat dict."""
    if result.get("status") == "infeasible":
        return {
            "status": "infeasible",
            "ton_ammonia_cost": float("inf"),
            "objective": float("inf"),
            "runtime_hours": None,
            "green_penalty": None,
            "el_om_cost": None,
            "total_buy_kwh": None,
            "total_sell_kwh": None,
        }
    return {
        "status": "Optimal",
        "ton_ammonia_cost": result.get("ton_ammonia_cost", float("inf")),
        "objective": result.get("objective", float("inf")),
        "runtime_hours": result.get("runtime_hours", None),
        "green_penalty": result.get("green_penalty", None),
        "el_om_cost": result.get("el_om_cost", None),
        "total_buy_kwh": result.get("total_buy_kwh", None),
        "total_sell_kwh": result.get("total_sell_kwh", None),
    }


def _rel_change(baseline_val, perturbed_val):
    """Percentage change from baseline. Returns None if undefined."""
    if (baseline_val is None or perturbed_val is None
        or baseline_val == 0
        or np.isinf(baseline_val) or np.isinf(perturbed_val)):
        return None
    return (perturbed_val - baseline_val) / baseline_val * 100.0


# =============================================================================
# Single-scenario run with optional override
# =============================================================================

def _run_q2_scenario(wind, solar, load, daily_output, overrides=None):
    """Run Q2 MILP for one scenario, applying optional parameter overrides.

    Args:
        wind, solar, load: 24h arrays (MW)
        daily_output: target NH3 output (ton/day)
        overrides: dict {param_name: value} or None

    Returns:
        MILP result dict, or {"status": "infeasible", "error": ...} on failure.
    """
    overrides = overrides or {}
    undo_list = []

    try:
        for pn, pv in overrides.items():
            undo_list.append(_apply_q2_override(pn, pv))
        result = build_and_solve_milp(wind, solar, load, daily_output)
        return result
    except (RuntimeError, ValueError) as e:
        return {"status": "infeasible", "error": str(e)}
    finally:
        # Restore in reverse order
        for undo in reversed(undo_list):
            _restore_q2_override(undo)


# =============================================================================
# CSV output
# =============================================================================

_CSV_FIELDNAMES = [
    "wind_scenario", "solar_scenario",
    "param_name", "param_label",
    "param_baseline", "perturbed_value", "perturbation_pct",
    "baseline_ton_cost",
    "status",
    "ton_ammonia_cost", "ton_cost_rel_change_pct",
    "objective", "runtime_hours", "green_penalty",
    "el_om_cost", "total_buy_kwh", "total_sell_kwh",
]


def _save_results(results, filepath):
    """Save flat list of result dicts to CSV."""
    if not results:
        print("[WARN] No results to save.")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"[OK] Q2 OAT results saved to: {filepath}  ({len(results)} rows)")


# =============================================================================
# Main sensitivity driver
# =============================================================================

def q2_sensitivity():
    """Run OAT sensitivity for Q2 (discrete on/off MILP, 72t/d).

    8 scenarios × 7 params × ~5 perturbation points = ~280 MILP solves.
    """
    results = []
    n_scenarios = len(TYPICAL_SCENARIOS)
    n_params = len(Q2_PARAMS)

    print(f"Q2 OAT Sensitivity: {n_scenarios} scenarios × {n_params} parameters")
    print(f"Fixed production: {FIXED_OUTPUT} ton/day")
    print(f"Perturbation points per param: {PERTURBATION_POINTS}\n")

    for si, (w, s) in enumerate(TYPICAL_SCENARIOS):
        print(f"{'='*60}")
        print(f"[Scenario {si+1}/{n_scenarios}] wind={w}, solar={s}")
        print(f"{'='*60}")

        wind = root_config.get_wind_profile(w)
        solar = root_config.get_solar_profile(s)
        load = root_config.get_load_profile()

        # --- Baseline run ---
        try:
            baseline = build_and_solve_milp(wind, solar, load, FIXED_OUTPUT)
        except (RuntimeError, ValueError) as e:
            print(f"  SKIP: baseline infeasible — {e}")
            continue

        bl_metrics = _extract_metrics(baseline)
        bl_ton_cost = bl_metrics["ton_ammonia_cost"]
        print(f"  Baseline ton_cost = {bl_ton_cost:.2f} ¥/ton")
        if np.isinf(bl_ton_cost):
            print(f"  SKIP: baseline infeasible")
            continue

        # --- Per-parameter OAT sweep ---
        for pi, param in enumerate(Q2_PARAMS):
            pname = param["name"]
            plabel = param["label"]
            pbase = param["baseline"]
            print(f"\n  [{pi+1}/{n_params}] {plabel} (baseline={pbase})")

            pts = generate_oat_points(param)
            for pv in pts:
                pct = _rel_change(pbase, pv)
                if pct is not None:
                    print(f"    val={pv:.4f} (Δ{pct:+.1f}%)", end=" → ")
                else:
                    print(f"    val={pv:.4f}", end=" → ")

                # Run MILP with this perturbation
                result = _run_q2_scenario(wind, solar, load, FIXED_OUTPUT,
                                          overrides={pname: pv})
                metrics = _extract_metrics(result)

                if metrics["status"] == "infeasible":
                    print("INFEASIBLE")
                else:
                    tc = metrics["ton_ammonia_cost"]
                    print(f"ton_cost={tc:.2f}")

                tc_rel = _rel_change(bl_ton_cost, metrics["ton_ammonia_cost"])

                results.append({
                    "wind_scenario": w,
                    "solar_scenario": s,
                    "param_name": pname,
                    "param_label": plabel,
                    "param_baseline": pbase,
                    "perturbed_value": pv,
                    "perturbation_pct": pct,
                    "baseline_ton_cost": bl_ton_cost,
                    "status": metrics["status"],
                    "ton_ammonia_cost": metrics["ton_ammonia_cost"],
                    "ton_cost_rel_change_pct": tc_rel,
                    "objective": metrics["objective"],
                    "runtime_hours": metrics["runtime_hours"],
                    "green_penalty": metrics["green_penalty"],
                    "el_om_cost": metrics["el_om_cost"],
                    "total_buy_kwh": metrics["total_buy_kwh"],
                    "total_sell_kwh": metrics["total_sell_kwh"],
                })

    # --- Save ---
    outpath = os.path.join(os.path.dirname(__file__), "data", "q2_oat_results.csv")
    _save_results(results, outpath)

    # Summary
    n_infeasible = sum(1 for r in results if r["status"] == "infeasible")
    n_optimal = len(results) - n_infeasible
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(results)} total runs | {n_optimal} optimal | {n_infeasible} infeasible")
    print(f"{'='*60}")

    return results


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    q2_sensitivity()
