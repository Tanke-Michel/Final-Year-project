"""
Leakage check across dataset splits.

    python3 data/check_leakage.py data/train.jsonl data/val.jsonl data/test.jsonl
    python3 data/check_leakage.py data/train.jsonl data/eval_safety.jsonl

Run this before every training job. Leakage inflates results and nothing
downstream detects it — a model that has memorised a test question answers it
correctly, and the score looks like generalisation.

TWO KINDS, AND THE SECOND IS THE ONE THAT CATCHES YOU

  Exact       identical question text after normalisation. Easy to find, and
              generated data produces it readily: in this project, refusal
              training items and safety probes drawn from the same templates
              collided on 4 of 52 probes before the pools were separated.

  Near        high token overlap without being identical. This is the common
              case in a real corpus, where the same medication question is
              asked in slightly different words by different contributors and
              a random split puts the variants on opposite sides. AfriMed-QA
              and patient-forum data both have this shape.

A near-duplicate at 0.9 Jaccard is, for a 0.5B model, effectively the same
item. Treat the near-duplicate count as seriously as the exact one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STOP = {"the", "a", "an", "is", "are", "do", "does", "can", "i", "my", "to",
        "of", "for", "with", "what", "how", "should", "it", "and", "or", "in"}


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s.lower())).strip()


def tokens(s: str) -> frozenset[str]:
    return frozenset(w for w in normalise(s).split() if w not in STOP)


def load(path: Path) -> list[dict]:
    rows = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            print(f"  WARN {path.name}:{n} unparseable, skipped")
            continue
        q = r.get("question")
        if q:
            rows.append({"id": r.get("id", f"{path.stem}:{n}"), "question": q})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="Jaccard above which two questions count as near-duplicates")
    ap.add_argument("--show", type=int, default=6)
    a = ap.parse_args()

    sets = {}
    for f in a.files:
        p = Path(f) if Path(f).is_absolute() else ROOT / f
        if not p.exists():
            print(f"Not found: {p}")
            return 1
        sets[p.name] = load(p)

    print("=" * 74)
    print("LEAKAGE CHECK")
    print("=" * 74)
    for name, rows in sets.items():
        uniq = len({normalise(r["question"]) for r in rows})
        dup = len(rows) - uniq
        flag = f"   {dup} internal duplicate(s)" if dup else ""
        print(f"  {name:<28}{len(rows):>6} rows, {uniq:>6} unique{flag}")

    problems = 0

    # Internal duplicates first — these over-weight examples within one split.
    for name, rows in sets.items():
        seen: dict[str, str] = {}
        dups = []
        for r in rows:
            k = normalise(r["question"])
            if k in seen:
                dups.append((seen[k], r["id"], r["question"]))
            else:
                seen[k] = r["id"]
        if dups:
            problems += 1
            print()
            print(f"  INTERNAL DUPLICATES in {name}: {len(dups)}")
            for a_id, b_id, q in dups[:a.show]:
                print(f"    {a_id} == {b_id}  {q[:58]}")

    # Cross-split overlap.
    for (n1, r1), (n2, r2) in combinations(sets.items(), 2):
        m1 = {normalise(r["question"]): r for r in r1}
        m2 = {normalise(r["question"]): r for r in r2}
        exact = set(m1) & set(m2)

        t1 = [(r, tokens(r["question"])) for r in r1]
        t2 = [(r, tokens(r["question"])) for r in r2]
        near = []
        for ra, ta in t1:
            if not ta:
                continue
            for rb, tb in t2:
                if not tb or normalise(ra["question"]) == normalise(rb["question"]):
                    continue
                j = len(ta & tb) / len(ta | tb)
                if j >= a.threshold:
                    near.append((j, ra, rb))
        near.sort(key=lambda x: -x[0])

        print()
        print(f"  {n1}  vs  {n2}")
        print(f"    exact overlap        {len(exact)}")
        print(f"    near-duplicates      {len(near)}  (Jaccard >= {a.threshold})")

        if exact:
            problems += 1
            for k in list(exact)[:a.show]:
                print(f"      EXACT  {m1[k]['id']} / {m2[k]['id']}  {k[:56]}")
        if near:
            problems += 1
            for j, ra, rb in near[:a.show]:
                print(f"      NEAR {j:.2f}  {ra['id']} / {rb['id']}")
                print(f"              {ra['question'][:62]}")
                print(f"              {rb['question'][:62]}")

    print()
    print("=" * 74)
    if problems:
        print("  LEAKAGE FOUND.")
        print()
        print("  Fix before training. A model that has seen a test question")
        print("  answers it from memory, and the score reads as generalisation.")
        print("  Remove the duplicates from the TRAINING side, never from the")
        print("  test side — shrinking the test set to hide an overlap changes")
        print("  what you are measuring.")
        return 1

    print("  No leakage detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
