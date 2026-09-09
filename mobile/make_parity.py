"""Regenerate the Dart parity corpus from the Python guard.

Run this after ANY change to rules.json or guard.py, then re-run the Dart test:

    python mobile/make_parity.py
    dart test mobile/test/guard_test.dart

The Python guard is the reference implementation because it is the one that
produces the safety numbers in your results chapter. The Dart port must follow
it, never the other way round.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from safety.guard import Guard, normalise  # noqa: E402

CASES = ROOT / "mobile" / "test" / "guard_parity_cases.json"

def main() -> int:
    existing = json.loads(CASES.read_text(encoding="utf-8"))
    g = Guard()

    inputs = [c["input"] for c in existing["input_cases"]]
    outputs = [c["input"] for c in existing["output_cases"]]
    norm_inputs = [c["input"] for c in existing["normalisation"]]

    out = {
        "_comment": existing["_comment"],
        "normalisation": [{"input": t, "expected": normalise(t)} for t in norm_inputs],
        "input_cases": [],
        "output_cases": [],
    }
    for t in inputs:
        r = g.check_input(t)
        out["input_cases"].append({"input": t, "blocked": r.blocked,
                                   "rule_id": r.rule_id, "category": r.category})
    for t in outputs:
        r = g.check_output(t)
        out["output_cases"].append({"input": t, "blocked": r.blocked,
                                    "rule_id": r.rule_id, "category": r.category})

    CASES.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    nb = sum(1 for c in out["input_cases"] if c["blocked"])
    print(f"Regenerated {CASES}")
    print(f"  normalisation {len(out['normalisation'])}")
    print(f"  input         {len(out['input_cases'])} ({nb} blocked)")
    print(f"  output        {len(out['output_cases'])}")
    print("\nNow run: dart test mobile/test/guard_test.dart")
    return 0

if __name__ == "__main__":
    sys.exit(main())
