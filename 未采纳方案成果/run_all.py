#!/usr/bin/env python3
"""
run_all.py ── 一键端到端编排 Q1→Q5 (11 步骤)
==============================================
用法: python run_all.py
特性: 中文进度输出 / 容错不中断 / 最终汇总表
"""

import sys
import os
import time
import subprocess
import traceback

import pulp

# ── Path setup ────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ── Step definitions ──────────────────────────────────────────────────────────
# Format: (序号, 步骤名称, 类型, 代码或脚本路径)
#   类型 "exec"           → compile+exec 行内代码
#   类型 "subprocess:路径" → 子进程运行可视化脚本
STEPS = [
    ( 1, "Q1 典型日指标计算",                                 "exec", "from models.run_q1 import run_q1; run_q1()"),
    ( 2, "Q1 可视化",                                          "subprocess:visualization/charts_q1.py", None),
    ( 3, "Q2 MILP离散调度(24场景×5产量)",                     "exec", "from models.q2_milp import run_q2; run_q2()"),
    ( 4, "Q2 可视化",                                          "subprocess:visualization/charts_q2.py", None),
    ( 5, "Q3 LP连续调度(24场景)",                              "exec", "from models.q3_lp import run_all; run_all()"),
    ( 6, "Q3 聚类分析(6特征K-means)",                          "exec", "from models.q3_clustering import main; main()"),
    ( 7, "Q3 可视化",                                          "subprocess:visualization/charts_q3.py", None),
    ( 8, "Q4 储能优化(24场景离网)",                            "exec", "from models.run_q4 import main as q4_main; q4_main()"),
    ( 9, "Q4 可视化",                                          "subprocess:visualization/charts_q4.py", None),
    (10, "Q5 参数扫描(alpha∈[1,4])",                           "exec", "from models.run_q5 import run_q5; run_q5()"),
    (11, "Q5 可视化",                                          "subprocess:visualization/charts_q5.py", None),
]


def run_exec_step(code: str) -> bool:
    """Execute inline Python code. Return True on success."""
    compiled = compile(code, "<run_all>", "exec")
    exec(compiled, {"__name__": "__main__"})
    return True


def run_subprocess_step(relpath: str) -> bool:
    """Run a sibling script as subprocess. Return True if returncode == 0."""
    script = os.path.join(PROJECT_ROOT, relpath)
    if not os.path.isfile(script):
        print(f"    错误: 脚本不存在 → {script}")
        return False

    proc = subprocess.run(
        [sys.executable, script],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,   # 1 hour max per step
    )
    # Print stdout (trimmed) for visibility
    out = proc.stdout.strip()
    if out:
        lines = out.splitlines()
        for line in lines[-5:]:
            print(f"    {line}")

    if proc.returncode != 0:
        err = proc.stderr.strip()
        if err:
            print(f"    stderr: {err[:500]}")
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t_start = time.time()
    results = []  # list of (step_num, name, status, elapsed)

    print("=" * 72)
    print("  电工杯 A 题 — 一键端到端运行 (Q1→Q5)")
    print(f"  项目根目录: {PROJECT_ROOT}")
    print("=" * 72)

    for num, name, stype, code in STEPS:
        print(f"\n[{num:2d}/11] {name}...", end="", flush=True)

        t0 = time.time()
        status = "完成"
        try:
            if stype == "exec":
                run_exec_step(code)
            elif stype.startswith("subprocess:"):
                relpath = stype.split(":", 1)[1]
                ok = run_subprocess_step(relpath)
                if not ok:
                    status = "失败"
            else:
                status = f"未知类型: {stype}"
        except (ValueError, RuntimeError, pulp.PulpSolverError) as e:
            traceback.print_exc()
            status = "失败"

        elapsed = time.time() - t0
        print(f"  → {status} ({elapsed:.1f}s)")
        results.append((num, name, status, elapsed))

    # ── Final summary ──────────────────────────────────────────────────────
    total_elapsed = time.time() - t_start
    print("\n" + "=" * 72)
    print("  运行汇总")
    print("=" * 72)
    print(f"{'步骤':>4}  {'名称':<40} {'状态':>6} {'耗时':>8}")
    print("-" * 66)
    n_pass = 0
    n_fail = 0
    for num, name, status, elapsed in results:
        tag = "[OK]" if status == "完成" else "[FAIL]"
        print(f" {tag} {num:2d}  {name:<40} {status:>6} {elapsed:>7.1f}s")
        if status == "完成":
            n_pass += 1
        else:
            n_fail += 1
    print("-" * 66)
    print(f"  通过: {n_pass}/{len(STEPS)}  失败: {n_fail}/{len(STEPS)}  总耗时: {total_elapsed:.1f}s")
    print("=" * 72)

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
