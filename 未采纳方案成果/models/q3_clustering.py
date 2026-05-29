"""Q3 聚类分析: 对24场景LP结果做K-means聚类,识别典型运行模态
==================================================================
从 results/q3_results.csv 读取24行数据, 提取6维特征向量,
StandardScaler标准化后 K-means (k in {2,3,4}), 用silhouette score选最优k,
输出 labels CSV 和聚类报告 TXT。

6维特征: 吨氨成本(元/吨), eta_self, eta_green, eta_grid, 日购电量(MWh), 日售电量(MWh)

Run standalone:  python models/q3_clustering.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── 路径常量 ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "q3_results.csv")
LABELS_PATH = os.path.join(RESULTS_DIR, "q3_cluster_labels.csv")
REPORT_PATH = os.path.join(RESULTS_DIR, "q3_cluster_report.txt")

# ── 6维特征列名 (严格按题目要求) ──────────────────────────────────────────────
FEATURE_COLS = [
    "吨氨成本(元/吨)",
    "新能源自发自用电量占比(%)",
    "总用电量绿电比例(%)",
    "新能源上网电量比例(%)",
    "日购电量(MWh)",
    "日售电量(MWh)",
]

# ── 候选k值 ───────────────────────────────────────────────────────────────────
K_CANDIDATES = [2, 3, 4]
RANDOM_SEED = 42


def load_data(csv_path: str) -> pd.DataFrame:
    """读取 q3_results.csv, 返回含场景编号和6维特征的 DataFrame。"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    required = ["场景编号"] + FEATURE_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"缺少列: {missing}\n现有列: {list(df.columns)}")
    # Filter out summary rows (non-numeric scenario numbers like "年度汇总")
    df = df[pd.to_numeric(df["场景编号"], errors="coerce").notna()].copy()
    print(f"[OK] 读取 {len(df)} 行, {df.shape[1]} 列")
    return df[required].copy()


def select_best_k(X_scaled: np.ndarray, k_list: list, seed: int) -> tuple:
    """对每个 k 执行 K-means, 返回 (best_k, best_labels, best_score, scores_dict)。"""
    best_k = None
    best_score = -1.0
    best_labels = None
    scores = {}

    for k in k_list:
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        scores[k] = sil
        print(f"  k={k}, silhouette_score={sil:.4f}")
        if sil > best_score:
            best_score = sil
            best_k = k
            best_labels = labels

    return best_k, best_labels, best_score, scores


def name_clusters(df: pd.DataFrame, labels: np.ndarray, best_k: int) -> dict:
    """为每簇计算6个特征均值, 据此生成中文模态名称。

    命名规则: 对每个特征, 簇均值与全局均值对比, 偏离 >=10% 纳入命名标签,
    最后取前3个标签拼接。

    返回 {cluster_id: modal_name}。
    """
    global_means = df[FEATURE_COLS].mean()

    def _tag(val: float, gv: float, higher_is: str, lower_is: str) -> str:
        ratio = val / gv if gv != 0 else 1.0
        if ratio >= 1.10:
            return higher_is
        elif ratio <= 0.90:
            return lower_is
        return ""

    names = {}
    for cid in range(best_k):
        cm = df.loc[labels == cid, FEATURE_COLS].mean()
        tags = []

        t = _tag(cm["吨氨成本(元/吨)"], global_means["吨氨成本(元/吨)"], "高成本", "低成本")
        if t:
            tags.append(t)

        t = _tag(cm["新能源自发自用电量占比(%)"], global_means["新能源自发自用电量占比(%)"], "高自用", "低自用")
        if t:
            tags.append(t)

        t = _tag(cm["总用电量绿电比例(%)"], global_means["总用电量绿电比例(%)"], "高绿电", "低绿电")
        if t:
            tags.append(t)

        t = _tag(cm["新能源上网电量比例(%)"], global_means["新能源上网电量比例(%)"], "高电网", "低电网")
        if t:
            tags.append(t)

        t = _tag(cm["日购电量(MWh)"], global_means["日购电量(MWh)"], "高购电", "低购电")
        if t:
            tags.append(t)

        t = _tag(cm["日售电量(MWh)"], global_means["日售电量(MWh)"], "高售电", "低售电")
        if t:
            tags.append(t)

        if not tags:
            names[cid] = "均衡模式"
        else:
            # 去重保留顺序
            seen = set()
            unique = [t for t in tags if not (t in seen or seen.add(t))]
            names[cid] = "-".join(unique[:3]) + "型"

    return names


def generate_report(
    filepath: str,
    best_k: int,
    best_score: float,
    all_scores: dict,
    df: pd.DataFrame,
    labels: np.ndarray,
    cluster_names: dict,
):
    """生成聚类报告 TXT 文件。"""
    lines = []
    lines.append("=" * 72)
    lines.append("Q3 场景聚类分析报告")
    lines.append("=" * 72)
    lines.append(f"数据来源: results/q3_results.csv ({len(df)} 行)")
    lines.append(f"特征维度: {len(FEATURE_COLS)} — {', '.join(FEATURE_COLS)}")
    lines.append("")

    # ── Silhouette Score 汇总 ──
    lines.append("-" * 72)
    lines.append("Silhouette Score 汇总")
    lines.append("-" * 72)
    for k in sorted(all_scores):
        marker = "  <-- 最优" if k == best_k else ""
        lines.append(f"  k={k}: {all_scores[k]:.6f}{marker}")
    lines.append("")

    # ── 最优k ──
    lines.append("-" * 72)
    lines.append(f"最优 k = {best_k}, Silhouette Score = {best_score:.6f}")
    lines.append("-" * 72)
    lines.append("")

    # ── 每簇统计 ──
    global_means = df[FEATURE_COLS].mean()
    for cid in range(best_k):
        mask = labels == cid
        n = mask.sum()
        cmean = df.loc[mask, FEATURE_COLS].mean()
        scenes = df.loc[mask, "场景编号"].tolist()

        lines.append(f"簇 {cid}: {cluster_names[cid]}")
        lines.append(f"  场景数: {n}")
        lines.append(f"  场景编号: {scenes}")
        lines.append(f"  模态命名推理 (簇均值 vs 全局均值):")
        for f in FEATURE_COLS:
            gv = global_means[f]
            cv = cmean[f]
            ratio = cv / gv if gv != 0 else 1.0
            if ratio >= 1.05:
                direction = "高于"
            elif ratio <= 0.95:
                direction = "低于"
            else:
                direction = "接近"
            lines.append(
                f"    {f:20s}: 簇均值={cv:.4f}, 全局均值={gv:.4f}, "
                f"{direction}全局 ({ratio:.3f})"
            )
        lines.append("")

    # ── 特征均值对比表 ──
    lines.append("-" * 72)
    lines.append("特征簇均值对比 (行=簇, 列=特征)")
    lines.append("-" * 72)
    header = f"{'簇/模态':>32s}"
    for f in FEATURE_COLS:
        header += f" | {f:>14s}"
    lines.append(header)
    lines.append("-" * len(header))

    g_row = f"{'全局均值':>32s}"
    for f in FEATURE_COLS:
        g_row += f" | {global_means[f]:14.4f}"
    lines.append(g_row)

    for cid in range(best_k):
        mask = labels == cid
        cmean = df.loc[mask, FEATURE_COLS].mean()
        c_row = f"{f'簇{cid}: {cluster_names[cid]}':>32s}"
        for f in FEATURE_COLS:
            c_row += f" | {cmean[f]:14.4f}"
        lines.append(c_row)
    lines.append("")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] 报告已保存: {filepath}")


def save_labels(filepath: str, df: pd.DataFrame, labels: np.ndarray, cluster_names: dict):
    """保存聚类标签 CSV (utf-8-sig, 列: 场景编号, 聚类标签, 模态名称)。"""
    out = df[["场景编号"]].copy()
    out["聚类标签"] = labels
    out["模态名称"] = out["聚类标签"].map(cluster_names)
    out.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"[OK] 标签已保存: {filepath}")


def main():
    print("=" * 60)
    print("Q3 聚类分析 — K-means (6维特征, StandardScaler)")
    print("=" * 60)

    # ── 1. 加载数据 ──
    df = load_data(CSV_PATH)
    print(f"  特征列: {FEATURE_COLS}")

    # ── 2. 提取特征 & StandardScaler 标准化 ──
    X = df[FEATURE_COLS].values.astype(np.float64)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[OK] StandardScaler 标准化完成, shape={X_scaled.shape}")

    # ── 3. K-means 聚类, silhouette 选最优 k ──
    print(f"\nK-means 聚类 (k in {K_CANDIDATES}):")
    best_k, best_labels, best_score, all_scores = select_best_k(
        X_scaled, K_CANDIDATES, RANDOM_SEED
    )
    print(f"\n-> 最优 k = {best_k}, Silhouette Score = {best_score:.6f}")

    # ── 4. 模态命名 ──
    cluster_names = name_clusters(df, best_labels, best_k)
    for cid, name in sorted(cluster_names.items()):
        print(f"  簇 {cid}: {name}")

    # ── 5. 输出文件 ──
    save_labels(LABELS_PATH, df, best_labels, cluster_names)
    generate_report(
        REPORT_PATH, best_k, best_score, all_scores,
        df, best_labels, cluster_names,
    )

    print("\n" + "=" * 60)
    print("聚类分析完成!")
    print(f"  标签文件: {LABELS_PATH}")
    print(f"  报告文件: {REPORT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
