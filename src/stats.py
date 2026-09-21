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


def wilcoxon_paired(rows: list[dict]) -> list[dict]:
    """Guard on vs off is a PAIRED factor: the same evaluation items are scored
    twice. Throwing both into one Kruskal-Wallis treats them as independent
    arms, which inflates the comparison count and destroys power under
    Bonferroni. A paired signed-rank test on matched item ids is both correct
    and far more sensitive."""
    paired: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        base = r.get("base_arm") or r["arm"].replace("+guard", "")
        cond = "on" if r["arm"].endswith("+guard") else "off"
        paired[base][r["item_id"]][cond] = sum(r["scores"][d] for d in SCORE_DOMAINS)

    out = []
    for base, items in sorted(paired.items()):
        both = [(v["off"], v["on"]) for v in items.values() if "off" in v and "on" in v]
        if len(both) < 6:
            out.append({"arm": base, "n": len(both), "note": "too few pairs to test"})
            continue
        off = np.array([b[0] for b in both], dtype=float)
        on = np.array([b[1] for b in both], dtype=float)
        if np.all(off == on):
            out.append({"arm": base, "n": len(both), "delta": 0.0, "p": 1.0, "note": "identical"})
            continue
        try:
            stat, p = stats.wilcoxon(on, off)
        except ValueError as exc:
            out.append({"arm": base, "n": len(both), "note": str(exc)})
            continue
        out.append({
            "arm": base, "n": len(both),
            "median_off": float(np.median(off)), "median_on": float(np.median(on)),
            "delta": float(np.median(on - off)), "p": float(p),
        })
    return out


def load(path: Path) -> list[dict]:
    """Load graded rows, dropping failures and de-duplicating resumed passes.

    score.py writes append-only and records failures explicitly rather than
    dropping them, so this file can contain a failed record followed by a later
    successful one for the same row. Taking the last record per key resolves
    that. Failed records carry no "scores" key, so they must be filtered before
    anything downstream touches them.
    """
    raw = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    latest: dict[str, dict] = {}
    for r in raw:
        latest[f"{r['arm']}::{r['item_id']}"] = r

    rows = [r for r in latest.values() if r.get("status", "ok") == "ok" and "scores" in r]
    dropped = len(latest) - len(rows)
    if dropped:
        print(f"  NOTE: {dropped} ungraded row(s) excluded. Rerun src/score.py to")
        print(f"        retry them before reporting — unequal n across arms from")
        print(f"        non-random missingness biases the comparison.\n")
    return rows


def report(rows: list[dict], guard_filter: str | None = None) -> None:
    """guard_filter: 'off' | 'on' | None. Compare quantization arms WITHIN one
    guard condition. Mixing conditions makes the omnibus test uninterpretable."""
    if guard_filter is not None:
        want_guard = guard_filter == "on"
        rows = [r for r in rows if r["arm"].endswith("+guard") == want_guard]
        if not rows:
            print(f"No rows with guard={guard_filter}.")
            return

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

    if not np.isfinite(p):
        # scipy returns NaN when every observation is identical: there is no
        # variance to partition. Printing "p = nan" looks like a bug; it is a
        # real and interpretable outcome, and at sub-billion-parameter scale a
        # degenerate model scoring identically everywhere is entirely possible.
        print("  p = undefined (no variance)")
        print()
        print("  Every arm produced identical total scores, so there is nothing")
        print("  for the test to distinguish. Check the raw generations before")
        print("  concluding anything: this usually means the model is emitting")
        print("  the same output regardless of input, or the judge is failing")
        print("  and defaulting to a constant score.")
        return

    print(f"  H = {h:.3f}   df = {df}   p = {p:.6f}")
    print(f"  {'Significant difference between arms.' if p < 0.05 else 'No significant difference between arms.'}")

    if p >= 0.05:
        print("\n  Omnibus test not significant — post hoc comparisons are not")
        print("  interpretable. Report this honestly rather than cherry-picking pairs.")
        return

    n_arms = len(totals)
    n_comp = n_arms * (n_arms - 1) // 2
    min_n = min(len(v) for v in totals.values())
    if n_comp > 15 or min_n < 30:
        print()
        print("  ** POWER WARNING **")
        print(f"     {n_arms} arms -> {n_comp} pairwise comparisons; smallest group n = {min_n}.")
        print("     Bonferroni multiplies every p-value by the comparison count, so with")
        print("     many arms or few items almost nothing reaches significance. Reduce the")
        print("     arms compared at once, or enlarge the evaluation set. Med-Pal used 231")
        print("     validation items; aim for at least 120.")

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


def full_report(rows: list[dict]) -> None:
    has_guard = any(r["arm"].endswith("+guard") for r in rows)

    if not has_guard:
        report(rows)
        return

    print("#" * 78)
    print("# QUANTIZATION ARMS — GUARD OFF (raw model behaviour)")
    print("#" * 78)
    report(rows, guard_filter="off")

    print()
    print("#" * 78)
    print("# QUANTIZATION ARMS — GUARD ON (deployed system behaviour)")
    print("#" * 78)
    report(rows, guard_filter="on")

    print()
    print("#" * 78)
    print("# GUARD EFFECT — Wilcoxon signed-rank, paired on item id")
    print("#" * 78)
    print(f"{'arm':<16}{'n':>5}{'med off':>10}{'med on':>10}{'delta':>9}{'p':>12}{'sig':>6}")
    print("-" * 78)
    for w in wilcoxon_paired(rows):
        if "p" not in w:
            print(f"{w['arm']:<16}{w['n']:>5}   {w.get('note','')}")
            continue
        sig = "  *" if w["p"] < 0.05 else "   "
        print(f"{w['arm']:<16}{w['n']:>5}{w['median_off']:>10.1f}{w['median_on']:>10.1f}"
              f"{w['delta']:>9.1f}{w['p']:>12.5f}{sig:>6}")
    print()
    print("  The delta column is what the safety layer contributes on top of the")
    print("  model. Report it separately from the quantization comparison — it")
    print("  answers a different question and a jury will ask which component")
    print("  is doing the work.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        full_report(load(Path(sys.argv[1])))
    else:
        print("No results file given — running self-test on synthetic data.\n")
        _self_test()
