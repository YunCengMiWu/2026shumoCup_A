# -*- coding: utf-8 -*-
"""Q4 OAT 参数灵敏度分析 — 离网运行 + 储能优化

Part A: 无储能离网下，日产量对风光容量系数 α 的敏感性
Part B: 储能配置 (容量/功率/日折旧) 对储能投资、充放电效率、SOC 边界的敏感性

场景: W1S2 (风电索引1, 光伏索引2 — 低资源场景，储能敏感性显著)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import csv

import config as root_config
from sensitivity.config import DATA_DIR

from utils import generate_all_scenarios, calc_total_depreciation, calc_ton_ammonia_cost

# Import Q4 module-level constants and functions
from q4 import q4_offgrid as _q4mod
from q4.q4_offgrid import (
    P_TOTAL, P_MIN, NH3_RATE_72,
    EL_RATIO, NH3_RATIO, EL_OM_BLENDED, NH3_OM_RATE,
)

# =============================================================================
# Q4-Specific Storage OAT Parameters
#
# Ranges chosen to cover physically/technically plausible extremes:
#   - Investment:  500 → 1500 ¥/kWh (task range)
#   - Efficiency:  0.80 → 0.95        (reasonable tech bounds)
#   - SOC bounds:  0.05→0.20 / 0.80→0.95
# =============================================================================

Q4_STORAGE_PARAMS = [
    {
        "name": "storage_investment",
        "label": "储能投资成本",
        "baseline": 1000,
        "unit": "¥/kWh",
        "oat_points": [round(x, 2) for x in np.arange(500, 1550, 50)],
        "patch_type": "investment",
    },
    {
        "name": "charge_efficiency",
        "label": "充电效率 $\\eta_c$",
        "baseline": 0.90,
        "unit": "ratio",
        "oat_points": [round(x, 4) for x in np.arange(0.80, 0.975, 0.025)],
        "patch_type": "charge_eff",
    },
    {
        "name": "discharge_efficiency",
        "label": "放电效率 $\\eta_d$",
        "baseline": 0.90,
        "unit": "ratio",
        "oat_points": [round(x, 4) for x in np.arange(0.80, 0.975, 0.025)],
        "patch_type": "discharge_eff",
    },
    {
        "name": "self_discharge_rate",
        "label": "自放电率 $\\sigma$",
        "baseline": 0.002,
        "unit": "per-hour",
        "oat_points": [round(x, 4) for x in np.arange(0.0, 0.055, 0.005)],
        "patch_type": "self_discharge",
    },
    {
        "name": "soc_min",
        "label": "SOC$_{min}$",
        "baseline": 0.10,
        "unit": "ratio",
        "oat_points": [round(x, 4) for x in np.arange(0.05, 0.225, 0.025)],
        "patch_type": "soc_bound",  # uses parameterized wrapper
    },
    {
        "name": "soc_max",
        "label": "SOC$_{max}$",
        "baseline": 0.90,
        "unit": "ratio",
        "oat_points": [round(x, 4) for x in np.arange(0.80, 0.975, 0.025)],
        "patch_type": "soc_bound",
    },
]


# =============================================================================
# Parameterized storage-LP wrapper (for SOC_MIN / SOC_MAX sensitivity)
#
# _build_storage_lp() in q4_offgrid.py hardcodes soc_min=0.10, soc_max=0.90
# as LOCAL variables (lines 250-251).  We cannot vary them without modifying
# the source file.  This wrapper replicates the exact logic with parameterized
# SOC bounds, reading all other constants from the live q4_offgrid module
# (so that patched investment/efficiency values are respected).
# =============================================================================

def _build_storage_lp_soc(wind_arr, solar_arr, load_arr, prod_target,
                           soc_min=0.10, soc_max=0.90,
                           fixed_capacity=None, fixed_power=None,
                           label="", maximize_production=False):
    """Identical to q4_offgrid._build_storage_lp() except SOC bounds are arguments."""
    from pulp import (LpProblem, LpVariable, LpMinimize, LpMaximize,
                       lpSum, value, PULP_CBC_CMD)

    # Read live module-level values (possibly patched) from q4_offgrid
    p_total = _q4mod.P_TOTAL
    p_min   = _q4mod.P_MIN
    nh3_r   = _q4mod.NH3_RATE_72
    eta_c   = _q4mod.ETA_C
    eta_d   = _q4mod.ETA_D
    s_disch = _q4mod.SELF_DISCH
    s_inv   = _q4mod.STORAGE_INVESTMENT
    s_life  = _q4mod.STORAGE_LIFETIME

    if maximize_production:
        prob = LpProblem(f"Q4_MaxProd_{label}", LpMaximize)
    else:
        prob = LpProblem(f"Q4_Storage_{label}", LpMinimize)

    N = 24
    # Decision variables
    p    = [LpVariable(f"p_{t}",    lowBound=0) for t in range(N)]
    Pc   = [LpVariable(f"Pc_{t}",   lowBound=0) for t in range(N)]
    Pd   = [LpVariable(f"Pd_{t}",   lowBound=0) for t in range(N)]
    curt = [LpVariable(f"curt_{t}", lowBound=0) for t in range(N)]
    z    = [LpVariable(f"z_{t}",    cat="Binary") for t in range(N)]

    if fixed_capacity is not None:
        capacity = fixed_capacity
    else:
        capacity = LpVariable("capacity", lowBound=0)

    if fixed_power is not None:
        spower = fixed_power
    else:
        spower = LpVariable("spower", lowBound=0)

    soc = [LpVariable(f"soc_{t}", lowBound=0) for t in range(N + 1)]

    # --- Semi-continuous: p[t] ∈ {0} ∪ [p_min, p_total] ---
    for t in range(N):
        prob += p[t] >= p_min * z[t],   f"Pmin_{t}"
        prob += p[t] <= p_total * z[t], f"Pmax_{t}"

    # --- SOC init & terminal ---
    if fixed_capacity is not None:
        prob += soc[0]  == 0.5 * capacity, "SOC_Init"
        prob += soc[N]  == soc[0],         "SOC_Term"
    else:
        prob += soc[0]  == 0.5 * capacity, "SOC_Init"
        prob += soc[N]  == soc[0],         "SOC_Term"

    # --- SOC bounds (PARAMETERIZED) ---
    for t in range(N):
        prob += soc[t] >= soc_min * capacity, f"SOC_lo_{t}"
        prob += soc[t] <= soc_max * capacity, f"SOC_up_{t}"

    # --- SOC dynamics: soc[t+1] = soc[t]·SELF_DISCH + η_c·Pc[t] - Pd[t]/η_d ---
    inv_eta_d = 1.0 / eta_d
    for t in range(N):
        prob += (soc[t + 1] == soc[t] * s_disch + eta_c * Pc[t] - inv_eta_d * Pd[t],
                 f"SOC_dyn_{t}")

    # --- Charge / discharge ≤ rated power ---
    for t in range(N):
        prob += Pc[t] <= spower, f"Pc_bound_{t}"
        prob += Pd[t] <= spower, f"Pd_bound_{t}"

    # --- Power balance (with curtailment) ---
    for t in range(N):
        prob += (wind_arr[t] + solar_arr[t] + Pd[t]
                 == Pc[t] + p[t] + load_arr[t] + curt[t], f"Bal_{t}")

    # --- Production target ---
    if maximize_production:
        prob += lpSum(p) * nh3_r / p_total
    else:
        prob += lpSum(p) * nh3_r / p_total >= prod_target, "ProdTarget"
        # C-rate ≤ 4 (4C discharge limit)
        if fixed_capacity is None or fixed_power is None:
            prob += spower <= capacity * 4.0, "PowerCapRatio"
        # Objective: minimize daily depreciation + tiny power penalty
        daily_dep = capacity * s_inv * 1000 / s_life / 365
        prob += daily_dep + 0.01 * spower, "MinStorageDep"

    # --- Solve ---
    solver = PULP_CBC_CMD(msg=False, timeLimit=600, gapRel=0.01)
    prob.solve(solver)

    # --- Extract results ---
    p_vals    = [value(p[t])    for t in range(N)]
    pc_vals   = [value(Pc[t])   for t in range(N)]
    pd_vals   = [value(Pd[t])   for t in range(N)]
    curt_vals = [value(curt[t]) for t in range(N)]
    z_vals    = [int(round(value(z[t]))) for t in range(N)]
    soc_vals  = [value(soc[t])  for t in range(N + 1)]

    actual_prod = sum(p_vals) * nh3_r / p_total
    cap_val = fixed_capacity if fixed_capacity is not None else value(capacity)
    pow_val = fixed_power if fixed_power is not None else value(spower)

    result = {
        'status':               'Optimal' if prob.status == 1 else f'Status_{prob.status}',
        'storage_capacity_mwh': cap_val,
        'storage_power_mw':     pow_val,
        'p_schedule':           p_vals,
        'charge_schedule':      pc_vals,
        'discharge_schedule':   pd_vals,
        'curtailment_schedule': curt_vals,
        'z_schedule':           z_vals,
        'soc_schedule':         soc_vals,
        'daily_production':     actual_prod,
        'objective':            value(prob.objective),
        'runtime_hours':        sum(z_vals),
    }
    return result, prob


# =============================================================================
# Parameter patching helpers
#
# When varying a storage parameter, we may need to update values in multiple
# modules: root_config (config.py), _q4mod (q4_offgrid.py), and sometimes
# utils.  Each patch_type maps to a specific patching/restore strategy.
# =============================================================================

def _patch(param_name, new_val):
    """Apply parameter perturbation. Returns a restore dict."""
    r = {}

    if param_name == "storage_investment":
        import utils as _utils
        r["_r0"] = root_config.STORAGE_INVESTMENT;  root_config.STORAGE_INVESTMENT = new_val
        r["_q4"] = _q4mod.STORAGE_INVESTMENT;        _q4mod.STORAGE_INVESTMENT = new_val
        r["_ut"] = _utils.STORAGE_INVESTMENT;         _utils.STORAGE_INVESTMENT = new_val

    elif param_name == "charge_efficiency":
        r["_r0"] = root_config.STORAGE_CHARGE_EFF;   root_config.STORAGE_CHARGE_EFF = new_val
        r["_q4"] = _q4mod.ETA_C;                     _q4mod.ETA_C = new_val

    elif param_name == "discharge_efficiency":
        r["_r0"] = root_config.STORAGE_DISCHARGE_EFF; root_config.STORAGE_DISCHARGE_EFF = new_val
        r["_q4"] = _q4mod.ETA_D;                     _q4mod.ETA_D = new_val

    elif param_name == "self_discharge_rate":
        r["_r0"] = root_config.STORAGE_SELF_DISCHARGE; root_config.STORAGE_SELF_DISCHARGE = new_val
        r["_q4"] = _q4mod.SELF_DISCH;                  _q4mod.SELF_DISCH = 1.0 - new_val

    # soc_min / soc_max handled by wrapper — no patching needed
    return r


def _restore(patch_dict):
    """Reverse parameter perturbation."""
    for key, val in patch_dict.items():
        if key == "_r0":
            root_config.STORAGE_INVESTMENT = val
        elif key == "_q4":
            _q4mod.STORAGE_INVESTMENT = val
        elif key == "_ut":
            import utils as _utils
            _utils.STORAGE_INVESTMENT = val
        else:
            # Generic: key tells which attr to restore
            if key == "_r0_c":
                root_config.STORAGE_CHARGE_EFF = val
            elif key == "_q4_c":
                _q4mod.ETA_C = val
            elif key == "_r0_d":
                root_config.STORAGE_DISCHARGE_EFF = val
            elif key == "_q4_d":
                _q4mod.ETA_D = val
            elif key == "_r0_s":
                root_config.STORAGE_SELF_DISCHARGE = val
            elif key == "_q4_s":
                _q4mod.SELF_DISCH = val


# =============================================================================
# Part A: No-storage off-grid — capacity coefficient α sweep
# =============================================================================

def part_a_offgrid_sensitivity():
    """Vary capacity coefficient α ∈ [0.2, 2.0] (step ~0.05).
    For each α, find the MINIMUM daily NH3 production across all 24 scenarios.
    """
    print("=" * 60)
    print("Part A: No-storage off-grid capacity sensitivity")
    print("=" * 60)

    all_scenarios = generate_all_scenarios()  # 24 (wind, solar) tuples
    results = []

    for alpha_try in np.arange(0.2, 2.05, 0.05):
        alpha = round(alpha_try, 2)
        P_scaled    = alpha * P_TOTAL
        P_min_scaled = 0.1 * P_scaled
        NH3_scaled   = alpha * NH3_RATE_72

        min_daily = float('inf')
        worst_ws  = (0, 0)

        for w, s in all_scenarios:
            wind  = root_config.get_wind_profile(w)
            solar = root_config.get_solar_profile(s)
            load  = root_config.get_load_profile()

            p_arr = np.zeros(24)
            for t in range(24):
                net = wind[t] + solar[t] - load[t]
                p_avail = max(net, 0.0)
                if p_avail >= P_min_scaled:
                    p_arr[t] = min(p_avail, P_scaled)

            if P_scaled > 0:
                daily_p = float(sum(p_arr)) * NH3_scaled / P_scaled
            else:
                daily_p = 0.0

            if daily_p < min_daily:
                min_daily = daily_p
                worst_ws  = (w, s)

        results.append({
            'section':               'part_a',
            'alpha':                 alpha,
            'P_scaled_mw':           round(P_scaled, 2),
            'min_daily_production_tpd': round(min_daily, 4),
            'worst_scenario':        f'W{worst_ws[0]}S{worst_ws[1]}',
        })

        print(f"  α={alpha:.2f}  P={P_scaled:5.1f}MW  "
              f"min_prod={min_daily:6.2f} t/d  ({results[-1]['worst_scenario']})")

    print(f"  → {len(results)} α-points evaluated")
    return results


# =============================================================================
# Part B: Storage optimisation sensitivity (W1S2 only)
# =============================================================================

def part_b_storage_sensitivity():
    """For each storage parameter, sweep its range and record optimal storage
    capacity, power, daily depreciation, and achievable production.

    Uses W1S2 (low-resource scenario where storage significantly affects production).
    """
    print("\n" + "=" * 60)
    print("Part B: Storage optimization sensitivity  (W1S2 only)")
    print("=" * 60)

    wind  = root_config.get_wind_profile(1)   # W1
    solar = root_config.get_solar_profile(2)   # S2
    load  = root_config.get_load_profile()

    PROD_TARGET = 12.5  # t/d — W1S2 low-resource scenario max achievable

    all_results = []

    for param in Q4_STORAGE_PARAMS:
        pname   = param["name"]
        plabel  = param["label"]
        baseline = param["baseline"]
        ptype   = param["patch_type"]

        print(f"\n  [{plabel}]  baseline={baseline}  ({len(param['oat_points'])} points)")

        for val in param["oat_points"]:
            val = round(val, 6)
            restore_info = {}
            result = None

            try:
                if ptype == "soc_bound":
                    # --- Use parameterized wrapper for SOC bounds ---
                    soc_min = val if pname == "soc_min" else 0.10
                    soc_max = val if pname == "soc_max" else 0.90

                    result, _ = _build_storage_lp_soc(
                        wind, solar, load, PROD_TARGET,
                        soc_min=soc_min, soc_max=soc_max,
                        label=f"sens_{pname}_{val:.4f}",
                    )
                else:
                    # --- Patch root config + q4_offgrid, call original LP ---
                    restore_info = _patch(pname, val)
                    result, _ = _q4mod._build_storage_lp(
                        wind, solar, load, PROD_TARGET,
                        label=f"sens_{pname}_{val:.4f}",
                    )

                status = result['status']
                base_pct = (val - baseline) / baseline * 100.0 if baseline != 0 else 0.0

                has_result = (result['daily_production'] is not None
                              and result['daily_production'] > 0)
                all_results.append({
                    'section':               'part_b',
                    'param_name':            pname,
                    'param_label':           plabel,
                    'perturbed_value':       val,
                    'perturbation_pct':      round(base_pct, 2),
                    'storage_capacity_mwh':  round(result['storage_capacity_mwh'], 4) if has_result else None,
                    'storage_power_mw':      round(result['storage_power_mw'], 4)   if has_result else None,
                    'daily_depreciation_yuan': round(result['objective'], 2)        if has_result else None,
                    'daily_production_tpd':  round(result['daily_production'], 4)   if has_result else None,
                    'status':                status,
                })

                if has_result:
                    print(f"    val={val:.4f}  Δ={base_pct:+.1f}%  "
                          f"cap={result['storage_capacity_mwh']:.2f}MWh  "
                          f"pow={result['storage_power_mw']:.2f}MW  "
                          f"dep={result['objective']:.2f}¥/d  [{status}]")
                else:
                    print(f"    val={val:.4f}  Δ={base_pct:+.1f}%  {status}")

            except Exception as e:
                base_pct = (val - baseline) / baseline * 100.0 if baseline != 0 else 0.0
                all_results.append({
                    'section':               'part_b',
                    'param_name':            pname,
                    'param_label':           plabel,
                    'perturbed_value':       val,
                    'perturbation_pct':      round(base_pct, 2),
                    'storage_capacity_mwh':  None,
                    'storage_power_mw':      None,
                    'daily_depreciation_yuan': None,
                    'daily_production_tpd':  None,
                    'status':                f'Error: {e}',
                })
                print(f"    val={val:.4f}  ERROR: {e}")

            finally:
                if restore_info and ptype != "soc_bound":
                    _restore_from_dict(restore_info, pname)

    print(f"\n  → {len(all_results)} parameter-points evaluated")
    return all_results


def _restore_from_dict(rdict, pname):
    """Restore parameters that were patched."""
    import utils as _utils

    if pname == "storage_investment":
        if "_r0" in rdict: root_config.STORAGE_INVESTMENT = rdict["_r0"]
        if "_q4" in rdict: _q4mod.STORAGE_INVESTMENT       = rdict["_q4"]
        if "_ut" in rdict: _utils.STORAGE_INVESTMENT        = rdict["_ut"]
    elif pname == "charge_efficiency":
        if "_r0" in rdict: root_config.STORAGE_CHARGE_EFF = rdict["_r0"]
        if "_q4" in rdict: _q4mod.ETA_C                   = rdict["_q4"]
    elif pname == "discharge_efficiency":
        if "_r0" in rdict: root_config.STORAGE_DISCHARGE_EFF = rdict["_r0"]
        if "_q4" in rdict: _q4mod.ETA_D                      = rdict["_q4"]
    elif pname == "self_discharge_rate":
        if "_r0" in rdict: root_config.STORAGE_SELF_DISCHARGE = rdict["_r0"]
        if "_q4" in rdict: _q4mod.SELF_DISCH                  = rdict["_q4"]


# =============================================================================
# Main: run both parts, save unified CSV
# =============================================================================

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # --- Part A ---
    part_a = part_a_offgrid_sensitivity()

    # --- Part B ---
    part_b = part_b_storage_sensitivity()

    # --- Save unified CSV ---
    csv_path = os.path.join(DATA_DIR, "q4_oat_results.csv")

    # Common fieldnames (union of Part A and Part B columns)
    fieldnames = [
        'section',
        # Part A fields
        'alpha', 'P_scaled_mw', 'min_daily_production_tpd', 'worst_scenario',
        # Part B fields
        'param_name', 'param_label', 'perturbed_value', 'perturbation_pct',
        'storage_capacity_mwh', 'storage_power_mw',
        'daily_depreciation_yuan', 'daily_production_tpd', 'status',
    ]

    all_rows = part_a + part_b

    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{'=' * 60}")
    print(f"[OK] Unified Q4 OAT results saved → {csv_path}")
    print(f"     Part A: {len(part_a)} rows  |  Part B: {len(part_b)} rows")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
