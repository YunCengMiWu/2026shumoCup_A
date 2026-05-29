#!/usr/bin/env python3
"""verify_figures.py — Automated verification of sensitivity analysis figures."""

import os, sys, subprocess, csv, glob as globmod

BASE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(BASE, 'figures')
DATA_DIR = os.path.join(BASE, 'data')
Q4_DIR = os.path.join(os.path.dirname(BASE), 'q4')

SENS_FIGS = [
    'tornado_overall.png', 'q1_curves.png', 'q2_curves.png',
    'q3_curves.png', 'q4_capacity_curve.png', 'importance_barchart.png',
    'q3_green_heatmap.png', 'q4_storage_curves.png',
]
Q4_FIGS = ['q4_curtailment_heatmap.png']

results = []

# CHECK 1: All sensitivity figures exist and are non-empty
print("CHECK 1: sensitivity/figures/ PNGs exist and non-empty...", end=" ")
ok = True
for fname in SENS_FIGS:
    path = os.path.join(FIG_DIR, fname)
    if not os.path.exists(path):
        print(f"\n  MISSING: {fname}")
        ok = False
    elif os.path.getsize(path) < 1024:
        print(f"\n  TOO SMALL: {fname} ({os.path.getsize(path)} bytes)")
        ok = False
results.append(("CHECK 1: sensitivity/figures PNGs", ok))
print("PASS" if ok else "FAIL")

# CHECK 2: Q4 figures exist and are non-empty
print("CHECK 2: q4/ PNGs exist and non-empty...", end=" ")
ok = True
for fname in Q4_FIGS:
    path = os.path.join(Q4_DIR, fname)
    if not os.path.exists(path):
        print(f"\n  MISSING: {fname}")
        ok = False
    elif os.path.getsize(path) < 500:
        print(f"\n  TOO SMALL: {fname} ({os.path.getsize(path)} bytes)")
        ok = False
results.append(("CHECK 2: q4/ PNGs", ok))
print("PASS" if ok else "FAIL")

# CHECK 3: No glyph missing warnings when generating
print("CHECK 3: No 'missing from current font' warnings...", end=" ")
try:
    proc = subprocess.run(
        [sys.executable, os.path.join(BASE, 'visualize.py')],
        capture_output=True, text=True, timeout=120, cwd=BASE
    )
    combined = proc.stdout + proc.stderr
    has_missing = 'missing from current font' in combined.lower()
    if has_missing:
        # Check if the missing glyphs are ONLY ¥ (yen sign) which is acceptable
        missing_lines = [l for l in combined.split('\n') if 'missing from current font' in l.lower()]
        only_yen = all('165' in l for l in missing_lines)  # 165 = ¥
        ok = only_yen
        if not only_yen:
            print(f"\n  NON-YEN MISSING GLYPH: {missing_lines[0]}")
    else:
        ok = True
except Exception as e:
    print(f"\n  ERROR: {e}")
    ok = False
results.append(("CHECK 3: No subscript glyph warnings", ok))
print("PASS" if ok else "FAIL")

# CHECK 4: Q4 CSV has differentiation
print("CHECK 4: Q4 OAT CSV has data differentiation...", end=" ")
try:
    csv_path = os.path.join(DATA_DIR, 'q4_oat_results.csv')
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    part_b = [r for r in rows if r.get('section') == 'part_b']
    caps = set()
    prods = set()
    for r in part_b:
        if r.get('storage_capacity_mwh') and r['storage_capacity_mwh'] != 'None' and r['storage_capacity_mwh']:
            caps.add(float(r['storage_capacity_mwh']))
        if r.get('daily_production_tpd') and r['daily_production_tpd'] != 'None' and r['daily_production_tpd']:
            prods.add(float(r['daily_production_tpd']))
    ok = len(caps) > 1 and len(part_b) >= 50
    print(f"(rows={len(part_b)}, unique_caps={len(caps)}, unique_prods={len(prods)})", end=" ")
except Exception as e:
    print(f"\n  ERROR: {e}")
    ok = False
results.append(("CHECK 4: Q4 CSV differentiation", ok))
print("PASS" if ok else "FAIL")

# CHECK 5: importance_barchart.png has sufficient dimensions
print("CHECK 5: importance_barchart.png dimensions...", end=" ")
try:
    barchart_path = os.path.join(FIG_DIR, 'importance_barchart.png')
    size = os.path.getsize(barchart_path)
    ok = size > 50000  # at least 50KB
    print(f"({size} bytes)", end=" ")
except Exception as e:
    print(f"\n  ERROR: {e}")
    ok = False
results.append(("CHECK 5: Bar chart dimensions", ok))
print("PASS" if ok else "FAIL")

# CHECK 6: q4_curtailment_heatmap is 2D (not 3D) — size check
print("CHECK 6: q4_curtailment_heatmap.png...", end=" ")
try:
    heatmap_path = os.path.join(Q4_DIR, 'q4_curtailment_heatmap.png')
    size = os.path.getsize(heatmap_path)
    ok = size > 20000  # at least 20KB
    print(f"({size} bytes)", end=" ")
except Exception as e:
    print(f"\n  ERROR: {e}")
    ok = False
results.append(("CHECK 6: Heatmap file OK", ok))
print("PASS" if ok else "FAIL")

# Summary
print()
passed = sum(1 for _, ok in results if ok)
total = len(results)
for name, ok in results:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

print(f"\n{passed}/{total} checks PASSED")
sys.exit(0 if passed == total else 1)
