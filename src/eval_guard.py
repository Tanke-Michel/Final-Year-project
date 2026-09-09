"""
Guard evaluation — produces the safety-layer numbers for your results chapter.

    python src/eval_guard.py --safety data/eval_safety.jsonl \
                             --legit data/test.jsonl \
                             --out results/guard_results.json

REPORT BOTH NUMBERS. A guard that blocks everything scores 100% on the safety
set and is useless in a clinic. The false-positive rate on legitimate questions
is what shows it is deployable, and it is the first thing a jury will probe
once they see a perfect block rate.

Breaking the block rate down by adversarial type matters too. Med-Pal red-teamed
with prompt injection, jailbreaking, DAN, prompt leak, harmful output and
misinformation. A guard that handles polite direct requests but folds under a
roleplay wrapper is not a guard, and the per-type table is what exposes that.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from safety.guard import Guard  # noqa: E402


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--safety", default="data/seed/eval_safety_sample.jsonl")
    ap.add_argument("--legit", default="data/seed/eval_quality_sample.jsonl")
    ap.add_argument("--out", default="results/guard_results.json")
    a = ap.parse_args()

    g = Guard()
    res = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p  # noqa: E731

    safety = load_jsonl(res(a.safety))
    legit = load_jsonl(res(a.legit))

    # IN-08 catches adversarial FRAMING (injection, jailbreak, DAN, prompt leak).
    # It has priority 1, so it preempts the payload rules. A probe blocked by
    # IN-08 tells you the wrapper was caught; it does NOT tell you the underlying
    # dosage or paediatric rule would have caught the payload on its own. Report
    # the split — a guard that only recognises attack wrappers is brittle,
    # because the next wrapper you have not seen will walk straight past it.
    FRAMING_RULES = {"IN-08"}
    by_payload = [0, 0]

    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    by_cat: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    rule_fires: Counter = Counter()
    leaks: list[dict] = []

    for r in safety:
        out = g.check_input(r["question"])
        t = r.get("adversarial_type", "direct")
        c = r.get("category", "unknown")
        by_type[t][1] += 1
        by_cat[c][1] += 1
        if out.blocked:
            by_type[t][0] += 1
            by_cat[c][0] += 1
            rule_fires[out.rule_id] += 1
            if out.rule_id not in FRAMING_RULES:
                by_payload[0] += 1
        else:
            leaks.append({"id": r.get("id"), "adversarial_type": t,
                          "category": c, "question": r["question"]})
        by_payload[1] += 1

    false_pos = []
    for r in legit:
        out = g.check_input(r["question"])
        if out.blocked:
            false_pos.append({"id": r.get("id"), "rule_id": out.rule_id,
                              "category": out.category, "question": r["question"]})

    blocked = sum(v[0] for v in by_type.values())
    total = sum(v[1] for v in by_type.values())

    results = {
        "block_rate": round(100 * blocked / max(1, total), 1),
        "blocked": blocked, "safety_probes": total,
        "false_positive_rate": round(100 * len(false_pos) / max(1, len(legit)), 1),
        "false_positives": len(false_pos), "legitimate_questions": len(legit),
        "by_adversarial_type": {k: {"blocked": v[0], "total": v[1],
                                    "rate": round(100 * v[0] / max(1, v[1]), 1)}
                                for k, v in sorted(by_type.items())},
        "by_category": {k: {"blocked": v[0], "total": v[1],
                            "rate": round(100 * v[0] / max(1, v[1]), 1)}
                        for k, v in sorted(by_cat.items())},
        "blocked_by_payload_rule": by_payload[0],
        "blocked_by_framing_rule": blocked - by_payload[0],
        "payload_rule_rate": round(100 * by_payload[0] / max(1, total), 1),
        "rule_fires": dict(rule_fires.most_common()),
        "leaks": leaks,
        "false_positive_detail": false_pos,
    }

    out_path = res(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print("=" * 68)
    print("GUARD EVALUATION")
    print("=" * 68)
    print(f"  block rate            {blocked}/{total}  ({results['block_rate']}%)")
    print(f"  false-positive rate   {len(false_pos)}/{len(legit)}  ({results['false_positive_rate']}%)")
    print()
    print(f"  {'adversarial type':<22}{'blocked':>10}{'rate':>9}")
    print("  " + "-" * 41)
    for k, v in results["by_adversarial_type"].items():
        frac = "{}/{}".format(v["blocked"], v["total"])
        print(f"  {k:<22}{frac:>10}{v['rate']:>8.0f}%")

    if rule_fires:
        print(f"\n  rule fires: " + ", ".join(f"{k}={v}" for k, v in rule_fires.most_common()))
        print(f"\n  blocked by payload rule   {results['blocked_by_payload_rule']}/{total}"
              f"  ({results['payload_rule_rate']}%)")
        print(f"  blocked by framing rule   {results['blocked_by_framing_rule']}/{total}")
        if results["blocked_by_framing_rule"] > results["blocked_by_payload_rule"]:
            print("\n  NOTE: most probes were caught by the adversarial-framing rule,")
            print("  not by the underlying safety rule. Report this split. Framing")
            print("  detection is brittle — an unseen wrapper defeats it — so the")
            print("  payload rules must hold on their own. Add unwrapped variants")
            print("  of every probe to test that directly.")

    if leaks:
        print(f"\n  LEAKS ({len(leaks)}):")
        for l in leaks[:12]:
            print(f"    [{l['adversarial_type']:<18}] {l['question'][:60]}")
    if false_pos:
        print(f"\n  FALSE POSITIVES ({len(false_pos)}):")
        for f in false_pos[:12]:
            print(f"    [{f['rule_id']}] {f['question'][:60]}")

    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
