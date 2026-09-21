"""
Day 13 on-device benchmark.

    # 1. build in PROFILE mode and install
    #    cd mobile && flutter run --profile
    # 2. open the Benchmark screen, set runs and cooldown, press Run
    # 3. collect:
    python src/benchmark.py --gguf runs/qwen05_qat4/gguf/model-q4_0.gguf --collect

Reads the OHAI_BENCH lines the app writes to logcat. The contract is defined in
mobile/lib/services/bench_logger.dart: one line per run, the marker followed by
JSON. If you change field names on one side, change them on the other — this
script validates the field set and names what is missing rather than silently
reporting zeros.

TWO RULES
  1. PROFILE MODE. Debug builds run unoptimised Dart; the numbers mean nothing.
  2. MEDIAN OVER REPEATS, WITH THE RANGE. Low-end handsets throttle thermally
     within seconds, so a single measurement is noise. A spread above roughly
     25 per cent indicates throttling rather than variance, and should be
     reported as such rather than averaged away.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKER = "OHAI_BENCH"
REQUIRED = {"run", "ttft_ms", "gen_ms", "tokens", "tok_s"}


def adb(*args: str, serial: str | None = None) -> str:
    cmd = ["adb"] + (["-s", serial] if serial else []) + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def device_info(serial: str | None) -> dict:
    def prop(k: str) -> str:
        try:
            return adb("shell", "getprop", k, serial=serial).strip()
        except Exception:
            return "unknown"

    mem = "unknown"
    try:
        for line in adb("shell", "cat", "/proc/meminfo", serial=serial).splitlines():
            if line.startswith("MemTotal"):
                mem = f"{int(line.split()[1]) / 1024:.0f} MB"
                break
    except Exception:
        pass

    return {
        "model": prop("ro.product.model"), "device": prop("ro.product.device"),
        "android": prop("ro.build.version.release"), "abi": prop("ro.product.cpu.abi"),
        "soc": prop("ro.board.platform"), "total_ram": mem,
    }


def peak_rss_mb(package: str, serial: str | None) -> float | None:
    """Total PSS from dumpsys. Poll DURING generation if you want a true peak —
    read afterwards it understates, because the peak passes quickly."""
    try:
        out = adb("shell", "dumpsys", "meminfo", package, serial=serial)
        for line in out.splitlines():
            if "TOTAL PSS" in line.upper() or line.strip().startswith("TOTAL"):
                nums = [t for t in line.split() if t.isdigit()]
                if nums:
                    return int(nums[0]) / 1024
    except Exception:
        return None
    return None


def parse_records(text: str) -> tuple[list[dict], list[str]]:
    """Extract OHAI_BENCH rows. Returns (records, problems)."""
    records, problems = [], []
    for line in text.splitlines():
        if MARKER not in line:
            continue
        payload = line.split(MARKER, 1)[1].strip()
        # logcat prefixes each line; the JSON object is the remainder.
        m = re.search(r"\{.*\}", payload)
        if not m:
            problems.append(f"unparseable: {payload[:70]}")
            continue
        try:
            rec = json.loads(m.group(0))
        except json.JSONDecodeError as e:
            problems.append(f"bad JSON ({e}): {payload[:70]}")
            continue
        missing = REQUIRED - rec.keys()
        if missing:
            problems.append(f"run {rec.get('run', '?')} missing {sorted(missing)}")
            continue
        records.append(rec)
    return records, problems


def summarise(name: str, vals: list[float], unit: str) -> dict:
    if not vals:
        return {"metric": name, "unit": unit, "n": 0}
    med = statistics.median(vals)
    spread = 100 * (max(vals) - min(vals)) / max(1e-9, med)
    return {
        "metric": name, "unit": unit, "n": len(vals),
        "median": round(med, 2), "min": round(min(vals), 2),
        "max": round(max(vals), 2), "spread_pct": round(spread, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", help="model file, for the size figure")
    ap.add_argument("--package", default="com.example.ohai")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--collect", action="store_true", help="read records from adb logcat")
    ap.add_argument("--from-file", help="read records from a saved logcat dump instead")
    ap.add_argument("--clear", action="store_true", help="clear logcat before collecting")
    ap.add_argument("--out", default="results/device_benchmark.json")
    a = ap.parse_args()

    print("=" * 70)
    print("DEVICE")
    print("=" * 70)
    info = device_info(a.serial) if not a.from_file else {"model": "from file"}
    for k, v in info.items():
        print(f"  {k:<12}{v}")
    if info.get("model") == "unknown":
        print("\n  No device detected. Connect a handset with USB debugging enabled.")
        print("  Emulator figures reflect your workstation, not a 3 GB phone —")
        print("  if no device is available, state that as a limitation rather")
        print("  than presenting emulator numbers as device numbers.")

    size_mb = None
    if a.gguf:
        g = Path(a.gguf) if Path(a.gguf).is_absolute() else ROOT / a.gguf
        if g.exists():
            size_mb = g.stat().st_size / 1e6
            print(f"\n  model file  {size_mb:.1f} MB  ({g.name})")
        else:
            print(f"\n  model file  NOT FOUND: {g}")

    if a.clear:
        try:
            adb("logcat", "-c", serial=a.serial)
            print("\n  logcat cleared — run the benchmark in the app, then rerun with --collect")
            return 0
        except Exception as e:
            print(f"  could not clear logcat: {e}")
            return 1

    text = ""
    if a.from_file:
        text = Path(a.from_file).read_text(encoding="utf-8", errors="replace")
    elif a.collect:
        try:
            text = adb("logcat", "-d", serial=a.serial)
        except Exception as e:
            print(f"\n  adb logcat failed: {e}")
            return 1
    else:
        print("\n  Pass --collect (or --from-file) to read the benchmark records.")
        return 0

    records, problems = parse_records(text)

    print()
    print("=" * 70)
    print(f"RECORDS  ({len(records)} parsed)")
    print("=" * 70)
    if problems:
        for p in problems[:10]:
            print(f"  PROBLEM  {p}")
        print("  Field names must match mobile/lib/services/bench_logger.dart.")
    if not records:
        print("  No benchmark records found in logcat.")
        print("  Check that: the app ran in PROFILE mode, the Benchmark screen")
        print("  was actually run, and logcat has not rotated past the records.")
        return 1

    for r in sorted(records, key=lambda x: x["run"]):
        cold = f"  cold {r['cold_start_ms']}ms" if "cold_start_ms" in r else ""
        print(f"  run {r['run']:<3} ttft {r['ttft_ms']:>6}ms   "
              f"{r['tok_s']:>6.1f} tok/s   gen {r['gen_ms']:>6}ms{cold}")

    rss = peak_rss_mb(a.package, a.serial) if not a.from_file else None

    # The first run carries the cold-start penalty: weights are read from
    # storage and caches are empty, so its time-to-first-token is several times
    # the steady-state figure. Pooling it with the rest inflates the spread and
    # produces a false throttling signal — which would become a wrong claim
    # about the device in the report. Cold start is a separate measurement and
    # is reported separately.
    cold = [r for r in records if "cold_start_ms" in r]
    warm = [r for r in records if "cold_start_ms" not in r]
    if not warm:
        warm = records
        print("\n  NOTE: every record is a cold start. Steady-state figures below")
        print("  include warm-up cost. Run the benchmark without restarting the")
        print("  app between iterations to get steady-state numbers.")

    metrics = [
        summarise("cold_start", [r["cold_start_ms"] for r in cold], "ms"),
        summarise("ttft_cold", [r["ttft_ms"] for r in cold], "ms"),
        summarise("ttft_steady", [r["ttft_ms"] for r in warm], "ms"),
        summarise("generation_speed", [r["tok_s"] for r in warm], "tok/s"),
        summarise("peak_rss", [r["peak_rss_mb"] for r in records if "peak_rss_mb" in r]
                  or ([rss] if rss else []), "MB"),
    ]

    print()
    print("=" * 70)
    print("SUMMARY  (median over runs, with range)")
    print("=" * 70)
    print(f"  cold runs {len(cold)}, steady-state runs {len(warm)}\n")
    throttling = False
    # Spread is only a throttling signal for steady-state metrics. A single
    # cold run trivially has zero spread, and cold runs are not comparable to
    # warm ones.
    STEADY = {"ttft_steady", "generation_speed", "peak_rss"}
    for m in metrics:
        if not m["n"]:
            print(f"  {m['metric']:<22}no data")
            continue
        flag = ""
        if m["metric"] in STEADY and m["n"] >= 3 and m["spread_pct"] > 25:
            flag = "  <- spread >25%, likely thermal throttling"
            throttling = True
        print(f"  {m['metric']:<22}{m['median']:>9} {m['unit']:<7}"
              f"(range {m['min']}-{m['max']}, spread {m['spread_pct']}%){flag}")

    if throttling:
        print()
        print("  Increase the cooldown in the app and rerun. Report the median")
        print("  with its range rather than a single figure, and say in the")
        print("  methods that repeats were spaced to allow thermal recovery.")

    results = {
        "device": info,
        "model_file": Path(a.gguf).name if a.gguf else None,
        "model_file_mb": round(size_mb, 1) if size_mb else None,
        "runs": len(records), "cold_runs": len(cold), "steady_runs": len(warm),
        "records": records, "metrics": metrics,
    }
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")
    print("Then: python src/figures.py --bench results/device_benchmark.json ...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
