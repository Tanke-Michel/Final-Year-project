"""
Non-parametric comparison of SCORE results across quantization arms.

SCORE is a 3-point Likert scale, so the data are ordinal and non-normal.
Following Elangovan et al.:

    Kruskal-Wallis        omnibus test for a difference in medians across arms
    Dunn's post hoc       pairwise, with Bonferroni correction
    Fleiss' Kappa         inter-rater agreement, if you have >1 grader

Dunn's test is implemented here rather than pulled from a package: it is
thirty lines, it removes a dependency, and you can explain it at the defence.

    python src/stats.py results/score_results.jsonl
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats

SCORE_DOMAINS = ["safety", "clinical_accuracy", "objectivity", "reproducibility", "ease"]


def kruskal_wallis(groups: dict[str, list[float]]) -> tuple[float, float, int]:
    names = list(groups)
    h, p = stats.kruskal(*[groups[n] for n in names])
    return h, p, len(names) - 1


def dunn_test(groups: dict[str, list[float]]) -> list[dict]:
    """Pairwise post hoc after Kruskal-Wallis, Bonferroni corrected.

    z = (Ri - Rj) / sqrt( ((N(N+1)/12) - T) * (1/ni + 1/nj) )

    where Ri, Rj are mean ranks and T is the tie correction. Ties matter here:
    a 3-point scale produces a great many of them, so omitting the correction
    inflates significance.
    """
    names = list(groups)
    all_vals = np.concatenate([np.asarray(groups[n], dtype=float) for n in names])
    ranks = stats.rankdata(all_vals)

    n_total = len(all_vals)
    idx, mean_ranks, sizes = 0, {}, {}
    for name in names:
        k = len(groups[name])
        mean_ranks[name] = ranks[idx: idx + k].mean()
        sizes[name] = k
        idx += k

    # Tie correction
    _, counts = np.unique(all_vals, return_counts=True)
    ties = counts[counts > 1]
    tie_term = float(np.sum(ties ** 3 - ties)) / (12.0 * (n_total - 1)) if len(ties) else 0.0
    sigma_base = (n_total * (n_total + 1)) / 12.0 - tie_term

    pairs = list(combinations(names, 2))
    m = len(pairs)
    out = []
    for a, b in pairs:
        se = np.sqrt(sigma_base * (1.0 / sizes[a] + 1.0 / sizes[b]))
        z = (mean_ranks[a] - mean_ranks[b]) / se if se > 0 else 0.0
        p_raw = 2.0 * (1.0 - stats.norm.cdf(abs(z)))
        out.append({
            "a": a, "b": b,
            "mean_rank_a": round(mean_ranks[a], 2),
            "mean_rank_b": round(mean_ranks[b], 2),
            "z": round(float(z), 3),
            "p_raw": float(p_raw),
            "p_bonferroni": float(min(1.0, p_raw * m)),
            "significant_005": bool(min(1.0, p_raw * m) < 0.05),
        })
    return out


def fleiss_kappa(matrix: np.ndarray) -> float:
    """matrix[item, category] = number of raters choosing that category."""
    n_items, _ = matrix.shape
    n_raters = matrix[0].sum()
    p_j = matrix.sum(axis=0) / (n_items * n_raters)
    p_i = (np.sum(matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar, pe_bar = p_i.mean(), float(np.sum(p_j ** 2))
    return (p_bar - pe_bar) / (1 - pe_bar) if pe_bar < 1 else 1.0


def load(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def report(rows: list[dict]) -> None:
    by_arm: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    print("=" * 78)
    print("SCORE BY ARM")
    print("=" * 78)
    header = f"{'arm':<18}{'n':>5}{'median':>9}{'IQR':>14}{'safe+acc %':>13}"
    print(header)
    print("-" * 78)

    totals: dict[str, list[float]] = {}
    for arm, items in sorted(by_arm.items()):
        tot = [sum(i["scores"][d] for d in SCORE_DOMAINS) for i in items]
        totals[arm] = tot
        q1, q3 = np.percentile(tot, [25, 75])
        # Med-Pal's headline metric: proportion scoring 3 on BOTH safety and
        # clinical accuracy. This is the number that gives you comparability.
        good = sum(
            1 for i in items
            if i["scores"]["safety"] == 3 and i["scores"]["clinical_accuracy"] == 3
        )
        pct = 100.0 * good / len(items)
        print(f"{arm:<18}{len(items):>5}{np.median(tot):>9.1f}{f'{q1:.1f} - {q3:.1f}':>14}{pct:>12.1f}%")

    if len(totals) < 2:
        print("\nOnly one arm present — no comparison to run.")
        return

    print()
    print("=" * 78)
    print("KRUSKAL-WALLIS")
    print("=" * 78)
    h, p, df = kruskal_wallis(totals)
    print(f"  H = {h:.3f}   df = {df}   p = {p:.6f}")
    print(f"  {'Significant difference between arms.' if p < 0.05 else 'No significant difference between arms.'}")

    if p >= 0.05:
        print("\n  Omnibus test not significant — post hoc comparisons are not")
        print("  interpretable. Report this honestly rather than cherry-picking pairs.")
        return

    print()
    print("=" * 78)
    print("DUNN'S POST HOC (Bonferroni)")
    print("=" * 78)
    print(f"{'comparison':<40}{'z':>9}{'p_adj':>12}{'sig':>7}")
    print("-" * 78)
    for d in dunn_test(totals):
        label = f"{d['a']} vs {d['b']}"
        print(f"{label:<40}{d['z']:>9.3f}{d['p_bonferroni']:>12.5f}{'  *' if d['significant_005'] else '   ':>7}")


def _self_test() -> None:
    """Synthetic data with a known separation, to prove the maths runs."""
    rng = np.random.default_rng(42)
    rows = []
    for arm, weights in [
        ("fp16",   [0.10, 0.25, 0.65]),
        ("lora",   [0.12, 0.28, 0.60]),
        ("qlora4", [0.20, 0.35, 0.45]),
        ("qat4",   [0.15, 0.30, 0.55]),
        ("ptq4",   [0.45, 0.35, 0.20]),
    ]:
        for i in range(60):
            rows.append({
                "arm": arm,
                "item_id": f"qa-{i:04d}",
                "scores": {d: int(rng.choice([1, 2, 3], p=weights)) for d in SCORE_DOMAINS},
            })
    report(rows)

    print()
    print("=" * 78)
    print("FLEISS' KAPPA (self-test)")
    print("=" * 78)
    m = np.array([[3, 0, 0], [0, 3, 0], [1, 2, 0], [0, 1, 2], [2, 1, 0]])
    print(f"  kappa = {fleiss_kappa(m):.3f}")
    print("  For reference, Med-Pal's 8-member expert panel reached 0.111 —")
    print("  only slight agreement. Cite that when defending your own method.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        report(load(Path(sys.argv[1])))
    else:
        print("No results file given — running self-test on synthetic data.\n")
        _self_test()
