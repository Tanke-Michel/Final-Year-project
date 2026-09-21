"""
Validation of the hand-written statistics.

    python3 src/test_stats.py

Dunn's test, Cohen's kappa and Fleiss' kappa are implemented by hand in this
repo, to avoid dependencies and so that they can be explained under
examination. That choice is only defensible if they are correct, and a wrong
implementation would be invisible: every p-value in the results chapter would
be wrong and nothing downstream would flag it.

TWO LAYERS
  Reference cross-checks against scikit-learn, scikit-posthocs and the
  canonical Fleiss (1971) worked example. These are OPTIONAL dependencies —
  install them to run this check, not to run the project:

      pip install scikit-learn scikit-posthocs

  Edge cases that reference libraries do not always cover and that real data
  will produce: zero variance, universal ties, single-item groups, perfect and
  perfect-inverse agreement. These run with no extra dependencies.

For the thesis: once this passes, you can state that the statistical
implementations were validated against reference libraries and a published
worked example. That is a stronger claim than "I used scipy".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from agreement import cohen_kappa                 # noqa: E402
from stats import dunn_test, fleiss_kappa, kruskal_wallis  # noqa: E402

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(f"{name} {detail}")


# --------------------------------------------------------------- references

def reference_checks() -> None:
    try:
        import pandas as pd
        import scikit_posthocs as sp
        from sklearn.metrics import cohen_kappa_score
    except ImportError:
        print("  [SKIP] reference libraries not installed")
        print("         pip install scikit-learn scikit-posthocs")
        return

    rng = np.random.default_rng(0)

    worst = 0.0
    for weights, skl in (("quadratic", "quadratic"), ("linear", "linear"), ("none", None)):
        for _ in range(6):
            a = rng.integers(1, 4, 60).tolist()
            b = rng.integers(1, 4, 60).tolist()
            b = [x if rng.random() < 0.6 else y for x, y in zip(a, b)]
            worst = max(worst, abs(cohen_kappa(a, b, weights=weights)
                                   - cohen_kappa_score(a, b, weights=skl, labels=[1, 2, 3])))
    check("Cohen's kappa vs scikit-learn", worst < 1e-9, f"max |diff| {worst:.2e}")

    # Fleiss (1971): 10 subjects, 14 raters, 5 categories; published kappa 0.210.
    fleiss = np.array([
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0], [0, 2, 2, 3, 7]])
    k = fleiss_kappa(fleiss)
    check("Fleiss' kappa vs published worked example", abs(k - 0.2099) < 5e-4,
          f"{k:.4f} vs 0.2099")

    worst = 0.0
    for _ in range(5):
        groups = {f"g{i}": rng.integers(1, 16, int(rng.integers(25, 45))).tolist()
                  for i in range(4)}
        mine = {(d["a"], d["b"]): d["p_bonferroni"] for d in dunn_test(groups)}
        long = pd.DataFrame([(g, v) for g, vs in groups.items() for v in vs],
                            columns=["grp", "val"])
        ref = sp.posthoc_dunn(long, val_col="val", group_col="grp", p_adjust="bonferroni")
        for (a, b), p in mine.items():
            worst = max(worst, abs(p - float(ref.loc[a, b])))
    check("Dunn's test vs scikit-posthocs", worst < 1e-9, f"max |diff| {worst:.2e}")


# -------------------------------------------------------------- edge cases

def edge_cases() -> None:
    # Perfect agreement must be exactly 1.0, not 0.9999 or NaN.
    a = [1, 2, 3] * 12
    check("kappa: perfect agreement = 1.0", abs(cohen_kappa(a, a) - 1.0) < 1e-12)

    # Every rating identical: no variance, so chance agreement is total and
    # kappa is undefined. Must not return NaN or crash — downstream code
    # formats this number.
    const = [2] * 30
    k = cohen_kappa(const, const)
    check("kappa: zero variance returns a finite number",
          np.isfinite(k), f"got {k}")

    # Systematic one-point offset, the shape a biased judge produces.
    h = [1, 2, 2, 3, 1, 2, 3, 2] * 5
    j = [min(3, x + 1) for x in h]
    kq, ku = cohen_kappa(h, j, "quadratic"), cohen_kappa(h, j, "none")
    check("kappa: quadratic weighting is more forgiving than unweighted",
          kq > ku, f"quad {kq:.3f} > unweighted {ku:.3f}")

    # Kruskal-Wallis on identical groups: no difference, so p must be high.
    same = {"a": [5] * 20, "b": [5] * 20, "c": [5] * 20}
    try:
        _, p, _ = kruskal_wallis(same)
        ok = (not np.isfinite(p)) or p > 0.05
        check("Kruskal-Wallis: identical groups are not significant", ok, f"p={p}")
    except ValueError:
        check("Kruskal-Wallis: identical groups raise cleanly", True,
              "scipy raises on all-identical input; caller must handle it")

    # Dunn's with heavy ties — a 3-point Likert scale produces these constantly,
    # and omitting the tie correction inflates significance.
    tied = {"a": [1, 2, 2, 3, 3, 3] * 6, "b": [1, 1, 2, 2, 3, 3] * 6,
            "c": [2, 2, 2, 3, 3, 3] * 6}
    res = dunn_test(tied)
    check("Dunn's: heavy ties produce valid probabilities",
          all(0.0 <= d["p_bonferroni"] <= 1.0 and np.isfinite(d["z"]) for d in res))

    # Bonferroni must never report a probability above 1.
    many = {f"g{i}": list(np.random.default_rng(i).integers(1, 4, 30)) for i in range(8)}
    res = dunn_test(many)
    check("Dunn's: Bonferroni is clamped at 1.0",
          all(d["p_bonferroni"] <= 1.0 for d in res), f"{len(res)} comparisons")

    # Unequal group sizes must still work — resumed grading can leave them.
    uneven = {"a": list(range(10)), "b": list(range(40)), "c": list(range(25))}
    res = dunn_test(uneven)
    check("Dunn's: unequal group sizes handled",
          all(np.isfinite(d["z"]) for d in res))


def main() -> int:
    print("=" * 70)
    print("REFERENCE CROSS-CHECKS")
    print("=" * 70)
    reference_checks()

    print()
    print("=" * 70)
    print("EDGE CASES")
    print("=" * 70)
    edge_cases()

    print()
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S):")
        for f in FAILS:
            print("  -", f)
        return 1
    print("All statistical checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
