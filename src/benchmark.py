"""
Day 13 on-device benchmark harness.

    python src/benchmark.py --gguf runs/qwen05_qat4/gguf/model-q4_0.gguf --repeats 5

Measures, per the brief 6.3: model file size, peak RSS, cold-start time,
time-to-first-token, generation speed.

TWO RULES
---------
1. REPEAT AND TAKE THE MEDIAN. Low-end phones throttle thermally within
   seconds. A single measurement is noise, and a jury that knows mobile
   hardware will ask. Five repeats minimum; report median and range.

2. AN EMULATOR IS NOT A DEVICE. Emulator latency reflects your laptop, not a
   3 GB phone. If you cannot source real hardware, say so prominently in the
   limitations rather than presenting emulator numbers as device numbers.
"""
from __future__ import annotations
import argparse, json, statistics, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def adb(*args: str, serial: str | None = None) -> str:
    cmd = ["adb"] + (["-s", serial] if serial else []) + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout

def device_info(serial: str | None) -> dict:
    def prop(k: str) -> str:
        try: return adb("shell", "getprop", k, serial=serial).strip()
        except Exception: return "unknown"
    mem = "unknown"
    try:
        for line in adb("shell", "cat", "/proc/meminfo", serial=serial).splitlines():
            if line.startswith("MemTotal"):
                mem = f"{int(line.split()[1]) / 1024:.0f} MB"; break
    except Exception: pass
    return {"model": prop("ro.product.model"), "device": prop("ro.product.device"),
            "android": prop("ro.build.version.release"), "abi": prop("ro.product.cpu.abi"),
            "soc": prop("ro.board.platform"), "total_ram": mem}

def peak_rss_mb(package: str, serial: str | None) -> float | None:
    """PSS from dumpsys meminfo. Poll DURING generation, not after — the peak
    passes quickly and a post-hoc reading understates it."""
    try:
        out = adb("shell", "dumpsys", "meminfo", package, serial=serial)
        for line in out.splitlines():
            if "TOTAL PSS" in line.upper() or line.strip().startswith("TOTAL"):
                nums = [t for t in line.split() if t.isdigit()]
                if nums: return int(nums[0]) / 1024
    except Exception: return None
    return None

def summarise(name: str, vals: list[float], unit: str) -> dict:
    if not vals: return {"metric": name, "n": 0}
    return {"metric": name, "unit": unit, "n": len(vals),
            "median": round(statistics.median(vals), 2),
            "min": round(min(vals), 2), "max": round(max(vals), 2),
            "spread_pct": round(100 * (max(vals) - min(vals)) / max(1e-9, statistics.median(vals)), 1)}

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--package", default="com.example.ohai")
    ap.add_argument("--serial", default=None)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/device_benchmark.json")
    ap.add_argument("--cooldown", type=float, default=20.0,
                    help="seconds between repeats; thermal recovery, do not set to 0")
    a = ap.parse_args()

    gguf = Path(a.gguf) if Path(a.gguf).is_absolute() else ROOT / a.gguf
    if not gguf.exists(): raise SystemExit(f"Not found: {gguf}")

    print("=" * 68); print("DEVICE"); print("=" * 68)
    info = device_info(a.serial)
    for k, v in info.items(): print(f"  {k:<12}{v}")
    if info["model"] == "unknown":
        print("\n  No device detected via adb. Connect a real phone with USB debugging.")
        print("  Emulator numbers are not device numbers — see the docstring.")

    size_mb = gguf.stat().st_size / 1e6
    print(f"\n  model file  {size_mb:.1f} MB  ({gguf.name})")

    print(); print("=" * 68); print(f"RUNS (n={a.repeats}, {a.cooldown}s cooldown)"); print("=" * 68)
    print("  Instrument your Flutter app to log these four values per run and")
    print("  print them as JSON to logcat, then parse here. Placeholder loop:")
    print()

    cold, ttft, tps, rss = [], [], [], []
    for i in range(a.repeats):
        print(f"  run {i+1}/{a.repeats} ... (wire to your app's logcat output)")
        # Replace with real capture:
        #   adb shell am start -n {package}/.MainActivity --es prompt "..."
        #   parse the JSON your app logs: {"cold_ms":..,"ttft_ms":..,"tok_s":..}
        r = peak_rss_mb(a.package, a.serial)
        if r: rss.append(r)
        if i < a.repeats - 1: time.sleep(a.cooldown)

    results = {
        "device": info, "model_file": gguf.name, "model_file_mb": round(size_mb, 1),
        "repeats": a.repeats,
        "metrics": [summarise("cold_start", cold, "ms"), summarise("time_to_first_token", ttft, "ms"),
                    summarise("generation_speed", tps, "tok/s"), summarise("peak_rss", rss, "MB")],
    }
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(); print("=" * 68)
    for m in results["metrics"]:
        if m["n"]: print(f"  {m['metric']:<22}{m['median']:>9} {m['unit']:<6} "
                         f"(range {m['min']}-{m['max']}, spread {m['spread_pct']}%)")
        else: print(f"  {m['metric']:<22}    no data — wire up capture")
    print("=" * 68)
    print(f"\nWrote {out}")
    print("\nIf spread exceeds ~25%, the device is throttling. Increase --cooldown")
    print("and report the median with its range rather than a single figure.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
