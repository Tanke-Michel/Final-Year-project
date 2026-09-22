"""
Run the whole experimental sweep unattended.

    python3 src/run_pipeline.py --status          # what is done, what is not
    python3 src/run_pipeline.py --dry-run         # print the plan, run nothing
    python3 src/run_pipeline.py --mock            # rehearse end-to-end, no GPU
    python3 src/run_pipeline.py                   # the real sweep

WHY THIS EXISTS
Days 7 to 10 in the brief are eleven training runs plus quantization,
generation, grading, statistics and figures. Driven by hand that is four days
of babysitting, and one mistyped flag in run seven invalidates the comparison.
Driven from here it is one command, and the arms are guaranteed to share the
configuration that makes the comparison controlled.

RESUME IS THE POINT. Every stage writes a marker on success. Rerunning skips
what is already done, so a crash at hour six costs you that stage and not the
night. Training a 0.5B model is cheap; retraining ten of them because the
eleventh failed is not.

    python3 src/run_pipeline.py --force train:qwen05:qat4    # redo one stage
    python3 src/run_pipeline.py --only train                 # one phase only
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "runs" / "pipeline_state.json"
LOGS = ROOT / "runs" / "logs"

# Arm allocation. The full method sweep runs on the primary model; the
# secondary models get the arms needed to test whether the ranking is
# model-dependent. Running every arm on every model is eleven-plus runs of
# debugging for a comparison the brief does not require.
ARM_PLAN = {
    "primary":   ["fp16", "lora", "qlora4", "qat8", "qat4", "qat_mixed"],
    "secondary": ["lora", "qlora4", "qat4"],
    "candidate": ["lora", "qat4"],
    "floor_probe": ["qat4"],
}


@dataclass
class Stage:
    key: str
    desc: str
    cmd: list[str]
    produces: Path | None = None
    optional: bool = False
    tags: list[str] = field(default_factory=list)


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except json.JSONDecodeError:
            pass
    return {"done": {}, "failed": {}}


def save_state(st: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2))


def build_plan(cfg_models: dict, mock: bool, eval_set: str,
               with_export: bool = False) -> list[Stage]:
    stages: list[Stage] = []
    arms_by_model: list[tuple[str, list[str]]] = []

    # Datasets the later stages depend on. Cheap, and their absence otherwise
    # shows up as a confusing failure three phases later.
    if not (ROOT / "data" / "eval_safety.jsonl").exists():
        stages.append(Stage(
            key="prepare:safety-set",
            desc="generate the adversarial safety evaluation set",
            cmd=[sys.executable, "data/build_dataset.py", "safety",
                 "--out", "data/eval_safety.jsonl"],
            produces=ROOT / "data" / "eval_safety.jsonl", tags=["prepare"]))
    if not (ROOT / "data" / "refusals.jsonl").exists():
        stages.append(Stage(
            key="prepare:refusals",
            desc="generate refusal training items",
            cmd=[sys.executable, "data/build_dataset.py", "refusals",
                 "--n", "400", "--out", "data/refusals.jsonl"],
            produces=ROOT / "data" / "refusals.jsonl", tags=["prepare"]))

    for m in cfg_models["candidates"]:
        if not m.get("params_total") or m["hf_id"] in (None, "TBD"):
            continue
        arms = ARM_PLAN.get(m.get("role", ""), [])
        if arms:
            arms_by_model.append((m["key"], arms))

    # --- train ---
    for key, arms in arms_by_model:
        for arm in arms:
            if arm == "ptq":
                continue
            run_dir = ROOT / "runs" / f"{key}_{arm}"
            stages.append(Stage(
                key=f"train:{key}:{arm}",
                desc=f"fine-tune {key} [{arm}]",
                cmd=[sys.executable, "src/train.py", "--model", key, "--method", arm,
                     "--data", "data/train.jsonl"] + (["--dry-run"] if mock else []),
                produces=None if mock else run_dir / "final",
                tags=["train"],
            ))

    # --- evaluation plan: what gets scored, in which form ---
    #
    # Every arm is scored in the form the phone would run it (src/quant_sim.py):
    # qat8 -> q8_0, qat_mixed -> mixed, everything else -> q4_0. The fp16 arm is
    # scored twice:
    #   fp16@none  unquantized — the upper bound
    #   fp16@q4_0  the same model rounded to Q4_0 with no training awareness —
    #              the post-training quantization baseline. It is llama.cpp's own
    #              round-to-nearest, i.e. the PTQ that this runtime actually
    #              supports. GPTQ and AWQ produce formats llama.cpp does not load
    #              as Q4_0, so they are not a like-for-like baseline here.
    # The headline comparison is then qat4@q4_0 against fp16@q4_0 and lora@q4_0:
    # same deployed format, with and without quantization-aware training.
    sys.path.insert(0, str(ROOT / "src"))
    from deploy_specs import deploy_spec_for_arm

    evals: list[tuple[str, str, str]] = []          # (model key, arm, deploy spec)
    for key, arms in arms_by_model:
        for arm in arms:
            if arm == "fp16":
                evals.append((key, arm, "none"))
            evals.append((key, arm, deploy_spec_for_arm(arm)))

    # --- GGUF export: optional, and not part of the GPU sweep ---
    # Quality is scored from exactly simulated deployment, so the sweep no longer
    # needs llama.cpp. Export is only for the configurations that go on the
    # phone for the Day 13 benchmark: run src/quantize.py for those, on a CPU
    # session or your own machine. --with-export adds it here.
    if with_export:
        for key, arms in arms_by_model:
            for arm in arms:
                stages.append(Stage(
                    key=f"quantize:{key}:{arm}",
                    desc=f"export {key} [{arm}] as {deploy_spec_for_arm(arm)} GGUF",
                    cmd=[sys.executable, "src/quantize.py", "--run", f"runs/{key}_{arm}"]
                        + (["--verify-only"] if mock else []),
                    optional=mock, tags=["quantize"],
                ))

    # --- generate ---
    for n, (key, arm, spec) in enumerate(evals):
        label = f"{key}-{arm}@{spec}"
        cmd = [sys.executable, "src/generate.py", "--eval", eval_set,
               "--out", "results/generations.jsonl", "--guard", "both"]
        if mock:
            cmd += ["--mock", "--arms", label]
        else:
            cmd += ["--run", f"runs/{key}_{arm}", "--arm", label, "--deploy", spec]
        if n > 0:
            cmd.append("--append")
        stages.append(Stage(
            key=f"generate:{label}",
            desc=f"score {key} [{arm}] as deployed ({spec})",
            cmd=cmd, tags=["generate"],
        ))

    # --- grade, verify, analyse ---
    stages += [
        Stage("score", "grade every generation (resumable)",
              [sys.executable, "src/score.py", "--generations", "results/generations.jsonl",
               "--out", "results/score_results.jsonl", "--workers", "4"]
              + (["--mock-judge"] if mock else []),
              produces=ROOT / "results" / "score_results.jsonl", tags=["analyse"]),
        Stage("score-verify", "completeness check on the grading",
              [sys.executable, "src/score.py", "--verify",
               "--out", "results/score_results.jsonl"], tags=["analyse"]),
        Stage("eval-guard", "safety layer block rate and false positives",
              [sys.executable, "src/eval_guard.py",
               "--safety", "data/eval_safety.jsonl",
               "--legit", eval_set,
               "--out", "results/guard_results.json"],
              optional=True, tags=["analyse"]),
        Stage("stats", "Kruskal-Wallis, Dunn's, paired guard test",
              [sys.executable, "src/stats.py", "results/score_results.jsonl"],
              tags=["analyse"]),
        Stage("figures", "figures and tables",
              [sys.executable, "src/figures.py",
               "--scores", "results/score_results.jsonl",
               "--out", "results/figures"] + (["--mock"] if mock else []),
              tags=["analyse"]),
        Stage("human-sample", "blinded subset for manual grading",
              [sys.executable, "src/score.py", "--sample-human",
               "--generations", "results/generations.jsonl",
               "--out", "results/score_results.jsonl"],
              optional=True, tags=["analyse"]),
    ]
    return stages


def run_stage(s: Stage, st: dict, mock: bool) -> bool:
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{s.key.replace(':', '_')}.log"
    t0 = time.time()
    print(f"  → {s.desc}")

    try:
        r = subprocess.run(s.cmd, cwd=ROOT, capture_output=True, text=True, timeout=None)
    except KeyboardInterrupt:
        print("\n  interrupted — state saved, rerun to resume")
        save_state(st)
        raise

    log.write_text(f"$ {' '.join(s.cmd)}\n\n{r.stdout}\n--- stderr ---\n{r.stderr}")
    dt = time.time() - t0

    if r.returncode == 0:
        if s.produces and not s.produces.exists():
            print(f"    FAILED: exit 0 but {s.produces} was not created  ({log.name})")
            st["failed"][s.key] = "missing output"
            return False
        st["done"][s.key] = {"at": time.time(), "seconds": round(dt, 1)}
        st["failed"].pop(s.key, None)
        print(f"    ok  {dt:.0f}s")
        return True

    tail = (r.stderr.strip().splitlines() or r.stdout.strip().splitlines() or ["(no output)"])[-1]
    print(f"    {'SKIP' if s.optional else 'FAILED'} ({r.returncode})  {tail[:88]}")
    print(f"    log: {log}")
    st["failed"][s.key] = tail[:200]
    return s.optional


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mock", action="store_true", help="rehearse with no GPU and no API key")
    ap.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    ap.add_argument("--status", action="store_true", help="show progress and exit")
    ap.add_argument("--only", help="run one phase: train | quantize | generate | analyse")
    ap.add_argument("--force", help="rerun one stage by key, e.g. train:qwen05:qat4")
    ap.add_argument("--eval", default="data/test.jsonl")
    ap.add_argument("--budget-hours", type=float, default=None,
                    help="stop starting new stages after this long, and exit cleanly. "
                         "On Kaggle, a session killed by the time limit may not save "
                         "its outputs; stopping early guarantees the commit completes.")
    ap.add_argument("--with-export", action="store_true",
                    help="also export GGUF files (needs llama.cpp built; not on Kaggle GPU)")
    ap.add_argument("--stop-on-fail", action="store_true",
                    help="halt on the first failure instead of continuing")
    a = ap.parse_args()

    models = yaml.safe_load((ROOT / "configs" / "models.yaml").read_text())
    eval_set = a.eval if (ROOT / a.eval).exists() else "data/seed/eval_quality_sample.jsonl"
    if eval_set != a.eval:
        print(f"NOTE: {a.eval} not found, using {eval_set}\n")

    stages = build_plan(models, a.mock, eval_set, a.with_export)
    st = load_state()

    if a.force:
        st["done"].pop(a.force, None)
        st["failed"].pop(a.force, None)
        save_state(st)
        print(f"cleared state for {a.force}\n")

    if a.only:
        stages = [s for s in stages if a.only in s.tags]

    if a.status or a.dry_run:
        print("=" * 74)
        print("PIPELINE PLAN" if a.dry_run else "PIPELINE STATUS")
        print("=" * 74)
        by_tag: dict[str, list[Stage]] = {}
        for s in stages:
            by_tag.setdefault(s.tags[0], []).append(s)
        for tag, group in by_tag.items():
            ndone = sum(1 for s in group if s.key in st["done"])
            print(f"\n  {tag.upper()}  ({ndone}/{len(group)} complete)")
            for s in group:
                if s.key in st["done"]:
                    mark, extra = "done", f"  {st['done'][s.key]['seconds']}s"
                elif s.key in st["failed"]:
                    mark, extra = "FAIL", f"  {st['failed'][s.key][:44]}"
                else:
                    mark, extra = "    ", ""
                print(f"    [{mark}] {s.key:<30}{extra}")
        total, done = len(stages), sum(1 for s in stages if s.key in st["done"])
        print(f"\n  {done}/{total} stages complete")
        if a.dry_run:
            print("\n  Nothing was run. Drop --dry-run to execute.")
        return 0

    todo = [s for s in stages if s.key not in st["done"]]
    print("=" * 74)
    print(f"RUNNING {len(todo)} of {len(stages)} stages" + ("  [MOCK]" if a.mock else ""))
    print("=" * 74)
    if not todo:
        print("  everything is already done — use --force <key> to redo a stage")
        return 0

    t0 = time.time()
    failures = []
    budget_hit = False
    for s in todo:
        if a.budget_hours and (time.time() - t0) / 3600 > a.budget_hours:
            budget_hit = True
            print(f"\n  time budget of {a.budget_hours} h reached — stopping cleanly so the")
            print("  session's outputs are saved. Commit again to continue from here.")
            break
        if not run_stage(s, st, a.mock):
            failures.append(s.key)
            save_state(st)
            if a.stop_on_fail:
                print("\n  stopping on first failure (--stop-on-fail)")
                break
            continue
        save_state(st)

    print()
    print("=" * 74)
    print(f"  {len(todo) - len(failures)}/{len(todo)} stages succeeded in {(time.time()-t0)/60:.1f} min")
    if failures:
        print(f"  FAILED: {', '.join(failures)}")
        print(f"  Logs in {LOGS}. Fix, then rerun — completed stages are skipped.")
    else:
        print("  All stages complete.")
        if a.mock:
            print("\n  MOCK RUN — results are synthetic. Do not report them.")
        else:
            print("\n  Next: grade the blinded sample by hand, then")
            print("    python3 src/agreement.py")
    return 1 if failures and not a.mock else 0


if __name__ == "__main__":
    raise SystemExit(main())
