"""
Agreement between the automated judge and your own manual grading.

    python src/score.py --sample-human            # writes results/human_sample.jsonl
    # fill in "your_scores" in that file by hand
    python src/agreement.py results/human_sample.jsonl

WHY THIS IS NOT OPTIONAL
Your quality results come from an automated judge. Without a measured agreement
figure against human grading, an examiner has no reason to believe the judge is
measuring what you claim, and "I used an LLM to grade it" is not an answer. This
single number is what makes the rest of the evaluation defensible.

WHAT TO REPORT
  Exact agreement    strict, and harsh on a 3-point ordinal scale
  Within-1           the practically meaningful figure here
  Weighted kappa     chance-corrected; quadratic weights suit ordinal data
  Bias               direction and size of systematic disagreement

READ THE BIAS FIGURE CAREFULLY. A judge that is systematically more generous
than you inflates every arm equally, so the RANKING of arms survives even though
the absolute values do not. That distinction is worth stating explicitly,
because it determines whether your comparative conclusion still holds when the
absolute numbers are questionable.

For context when you write this up: the eight-member expert panel in
Elangovan et al. reached a Fleiss' kappa of 0.111 — only slight agreement —
grading the same rubric. Experienced clinicians barely agreed with each other.
That is not an excuse for a weak figure, but it does mean an examiner treating
human grading as ground truth is assuming more than the literature supports.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ["safety", "clinical_accuracy", "objectivity", "ease"]
LABELS = {"safety": "Safety", "clinical_accuracy": "Clinical Accuracy",
          "objectivity": "Objectivity", "ease": "Ease of Understanding"}
SCALE = [1, 2, 3]


def cohen_kappa(a: list[int], b: list[int], weights: str = "quadratic") -> float:
    """Chance-corrected agreement between two raters on an ordinal scale.

    Quadratic weighting is the right default here: on a 1-3 scale, confusing 1
    with 3 is a far worse error than confusing 2 with 3, and unweighted kappa
    treats them identically.
    """
    n = len(SCALE)
    idx = {v: i for i, v in enumerate(SCALE)}

    obs = np.zeros((n, n))
    for x, y in zip(a, b):
        obs[idx[x], idx[y]] += 1
    obs /= max(1, obs.sum())

    ra = obs.sum(axis=1)
    rb = obs.sum(axis=0)
    exp = np.outer(ra, rb)

    if weights == "quadratic":
        w = np.array([[((i - j) ** 2) / ((n - 1) ** 2) for j in range(n)] for i in range(n)])
    elif weights == "linear":
        w = np.array([[abs(i - j) / (n - 1) for j in range(n)] for i in range(n)])
    else:
        w = 1.0 - np.eye(n)

    den = float((w * exp).sum())
    return 1.0 - float((w * obs).sum()) / den if den > 0 else 1.0


def interpret(k: float) -> str:
    """Landis and Koch bands. Report the number, not only the label."""
    if k < 0.00: return "poor"
    if k < 0.21: return "slight"
    if k < 0.41: return "fair"
    if k < 0.61: return "moderate"
    if k < 0.81: return "substantial"
    return "almost perfect"


def confusion(a: list[int], b: list[int]) -> np.ndarray:
    m = np.zeros((3, 3), dtype=int)
    idx = {v: i for i, v in enumerate(SCALE)}
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", nargs="?", default="results/human_sample.jsonl")
    ap.add_argument("--key", default=None,
                    help="key file; defaults to <sample>_key.jsonl")
    ap.add_argument("--confusion", action="store_true", help="print per-domain confusion matrices")
    ap.add_argument("--out", default="results/agreement.json")
    a = ap.parse_args()

    path = Path(a.sample) if Path(a.sample).is_absolute() else ROOT / a.sample
    if not path.exists():
        print(f"Not found: {path}\n\nGenerate it first:\n"
              f"  python src/score.py --sample-human")
        return 1

    key_path = (Path(a.key) if a.key else path.with_name(path.stem + "_key.jsonl"))
    if not key_path.is_absolute():
        key_path = ROOT / key_path
    if not key_path.exists():
        print(f"Key file not found: {key_path}\n\n"
              f"It is written alongside the sample by:\n"
              f"  python src/score.py --sample-human")
        return 1

    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    key = {r["blind_id"]: r
           for r in (json.loads(l) for l in key_path.read_text().splitlines() if l.strip())}

    graded, ungraded, unmatched = [], 0, 0
    for r in rows:
        hs = r.get("your_scores") or {}
        if any(hs.get(d) is None for d in DOMAINS):
            ungraded += 1
            continue
        k = key.get(r["blind_id"])
        if k is None:
            unmatched += 1
            continue
        r["_judge_scores"] = k["judge_scores"]
        r["_arm"] = k["arm"]
        graded.append(r)

    print("=" * 74)
    print("JUDGE vs MANUAL GRADING")
    print("=" * 74)
    print(f"  sampled      {len(rows)}")
    print(f"  hand-graded  {len(graded)}")
    if ungraded:
        print(f"  UNGRADED     {ungraded} — fill in 'your_scores' for these before reporting")
    if unmatched:
        print(f"  UNMATCHED    {unmatched} — blind_id absent from the key file")
    if len(graded) < 10:
        print("\n  Fewer than 10 graded items. Agreement estimated from this few is")
        print("  too noisy to report. Grade at least 10% of the evaluation set.")
        if not graded:
            return 1

    results = {"n_sampled": len(rows), "n_graded": len(graded), "domains": {}}

    # A rater who used only one value has produced no information, and kappa
    # against them is 1.0 by convention — a figure that looks excellent and
    # means nothing. Catch it before it reaches the results chapter.
    degenerate = []
    for d in DOMAINS:
        hv = {int(r["your_scores"][d]) for r in graded}
        jv = {int(r["_judge_scores"][d]) for r in graded}
        if len(hv) == 1:
            degenerate.append(f"{LABELS[d]}: you scored every item {hv.pop()}")
        if len(jv) == 1:
            degenerate.append(f"{LABELS[d]}: the judge scored every item {jv.pop()}")
    if degenerate:
        print()
        print("  DEGENERATE GRADING:")
        for g in degenerate:
            print(f"    - {g}")
        print("    Kappa against a constant rater is 1.0 by convention and carries")
        print("    no information. Treat these domains as unmeasured rather than")
        print("    as perfectly agreed.")

    print()
    print(f"  {'domain':<22}{'exact':>8}{'within-1':>10}{'kappa':>8}  interpretation")
    print("  " + "-" * 70)

    all_h, all_j = [], []
    for d in DOMAINS:
        h = [int(r["your_scores"][d]) for r in graded]
        j = [int(r["_judge_scores"][d]) for r in graded]
        all_h += h
        all_j += j

        exact = 100 * sum(x == y for x, y in zip(h, j)) / len(h)
        within1 = 100 * sum(abs(x - y) <= 1 for x, y in zip(h, j)) / len(h)
        k = cohen_kappa(h, j)
        bias = float(np.mean(np.array(j) - np.array(h)))

        results["domains"][d] = {
            "exact_pct": round(exact, 1), "within1_pct": round(within1, 1),
            "weighted_kappa": round(k, 3), "interpretation": interpret(k),
            "judge_minus_human": round(bias, 3),
        }
        print(f"  {LABELS[d]:<22}{exact:>7.1f}%{within1:>9.1f}%{k:>8.3f}  {interpret(k)}")

    overall_k = cohen_kappa(all_h, all_j)
    overall_exact = 100 * sum(x == y for x, y in zip(all_h, all_j)) / len(all_h)
    overall_within1 = 100 * sum(abs(x - y) <= 1 for x, y in zip(all_h, all_j)) / len(all_h)
    overall_bias = float(np.mean(np.array(all_j) - np.array(all_h)))

    results["overall"] = {
        "exact_pct": round(overall_exact, 1),
        "within1_pct": round(overall_within1, 1),
        "weighted_kappa": round(overall_k, 3),
        "interpretation": interpret(overall_k),
        "judge_minus_human": round(overall_bias, 3),
    }

    print("  " + "-" * 70)
    print(f"  {'OVERALL':<22}{overall_exact:>7.1f}%{overall_within1:>9.1f}%"
          f"{overall_k:>8.3f}  {interpret(overall_k)}")

    print()
    print("=" * 74)
    print("BIAS")
    print("=" * 74)
    direction = "more generous than" if overall_bias > 0 else (
        "harsher than" if overall_bias < 0 else "aligned with")
    print(f"  judge minus human: {overall_bias:+.3f} points on a 1-3 scale")
    print(f"  The judge is {direction} you.")
    if abs(overall_bias) >= 0.15:
        print()
        print("  Systematic bias of this size shifts ABSOLUTE scores but applies")
        print("  equally to every arm, so the RANKING of arms is unaffected. Say")
        print("  this explicitly: your comparative conclusion survives even if the")
        print("  absolute values are treated as uncertain.")
    else:
        print("  Small enough to treat as noise rather than systematic bias.")

    worst = min(results["domains"].items(), key=lambda kv: kv[1]["weighted_kappa"])
    print()
    print("=" * 74)
    print("FOR THE WRITE-UP")
    print("=" * 74)
    print(f"  Weakest domain: {LABELS[worst[0]]} (kappa {worst[1]['weighted_kappa']}).")
    print(f"  Name it in the limitations rather than reporting only the overall figure.")
    print()
    print("  Context: the eight-member expert panel in Elangovan et al. reached a")
    print("  Fleiss' kappa of 0.111 on this rubric — only slight agreement between")
    print("  experienced clinicians. Cite that when an examiner treats human")
    print("  grading as ground truth.")

    by_arm = Counter(r["_arm"] for r in graded)
    if len(by_arm) > 1:
        print()
        print("  graded items per arm: " +
              ", ".join(f"{k}={v}" for k, v in sorted(by_arm.items())))
        thin = [k for k, v in by_arm.items() if v < 3]
        if thin:
            print("  Some arms have very few graded items, so per-arm agreement")
            print("  cannot be estimated. The overall figure is still valid.")

    if a.confusion:
        print()
        for d in DOMAINS:
            h = [int(r["your_scores"][d]) for r in graded]
            j = [int(r["_judge_scores"][d]) for r in graded]
            m = confusion(h, j)
            print(f"  {LABELS[d]} — rows: human, cols: judge")
            print("        j=1  j=2  j=3")
            for i, row in enumerate(m):
                print(f"   h={i+1}  " + "".join(f"{v:>5}" for v in row))
            print()

    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
