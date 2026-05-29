"""Quick import verification for Wave 1 modules"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Test 1: constants
from utils.constants import BIG_M, WIND_CAPACITY, PV_CAPACITY, MIN_LOAD_RATIO, get_price
assert BIG_M == 1000, f"BIG_M={BIG_M}"
assert WIND_CAPACITY == 40
assert PV_CAPACITY == 64
assert MIN_LOAD_RATIO == 0.10
assert get_price(10) == 0.8024, f"Peak price wrong: {get_price(10)}"
assert get_price(8) == 0.6074, f"Flat price wrong: {get_price(8)}"
assert get_price(3) == 0.3424, f"Valley price wrong: {get_price(3)}"
print("PASS: constants.py")

# Test 2: data_loader
from utils.data_loader import get_all_scenarios, get_typical_day, get_base_load, get_scenario_by_id
scenarios = get_all_scenarios()
assert len(scenarios) == 24, f"Expected 24 scenarios, got {len(scenarios)}"
assert scenarios[0]['id'] == 1
assert scenarios[-1]['id'] == 24
wind_pu, pv_pu = get_typical_day()
base_pu = get_base_load()
assert len(wind_pu) == 24
assert len(pv_pu) == 24
assert len(base_pu) == 24
s = get_scenario_by_id(12)
assert 1 <= s['wind_scene'] <= 6
assert 1 <= s['pv_scene'] <= 4
print("PASS: data_loader.py")

# Test 3: indicators
from utils.indicators import compute_green_indicators, compute_compliance, compute_ton_cost
eta_self, eta_green, eta_grid = compute_green_indicators(100.0, 80.0, 10.0, 20.0)
assert 0 <= eta_self <= 1, f"eta_self={eta_self}"
assert 0 <= eta_green <= 1, f"eta_green={eta_green}"
assert 0 <= eta_grid <= 1, f"eta_grid={eta_grid}"
print(f"  eta_self={eta_self:.4f}, eta_green={eta_green:.4f}, eta_grid={eta_grid:.4f}")
comp = compute_compliance(eta_self, eta_green, eta_grid)
print(f"  all_compliant={comp['all_compliant']}")
print("PASS: indicators.py")

# Test 4: plotting (font + color schemes)
from visualization.plotting import get_current_font, get_color_scheme, COLOR_SCHEMES
font = get_current_font()
print(f"  Font: {font}")
colors_q1 = get_color_scheme(1)
assert 'primary' in colors_q1
assert len(COLOR_SCHEMES) >= 5
print("PASS: plotting.py")

# Test 5: run_q1
from models.run_q1 import run_q1
result, indicators = run_q1()
print(f"  E_RE={indicators['E_RE']:.1f}, E_total={indicators['E_total']:.1f}")
print(f"  eta_self={indicators['eta_self']:.4f}, eta_green={indicators['eta_green']:.4f}, eta_grid={indicators['eta_grid']:.4f}")
print(f"  daily_nh3={indicators['daily_nh3_tons']:.2f} tons, ton_cost={indicators['ton_cost']:.2f} yuan/ton")
print("PASS: run_q1.py")

print("\n=== ALL 6 WAVE 1 MODULES PASS ===")
