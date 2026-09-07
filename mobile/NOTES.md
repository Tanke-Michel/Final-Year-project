# Mobile — Day 3 spike, not Day 12

## Why this is on Day 3

The brief allocates one day (Day 12) to Android integration. That is enough
ONLY if the toolchain is already known to work. An NDK, ABI or FFI problem
discovered on Day 12 costs you the prototype entirely — and the brief lists a
working offline app as the primary deliverable.

## The spike (about one hour)

Build a throwaway Flutter app that loads any small GGUF and generates ONE
token. Not the real interface. Not the real model. Proof only that:

- [ ] Android SDK + NDK installed and on PATH
- [ ] Flutter builds and installs to the physical device
- [ ] The llama.cpp Android build produces a working shared library
- [ ] Dart FFI can call into it and return a token
- [ ] The correct ABI is bundled (arm64-v8a for essentially all current phones)

Delete it afterwards. Its only purpose is to move toolchain risk from Day 12
to Day 3.

## Engine decision

| Engine | QAT alignment | Flutter integration | Verdict |
|---|---|---|---|
| llama.cpp / GGUF | adequate if you match Q4_0 | mature, Dart FFI packages exist | **recommended** |
| ExecuTorch | excellent — pairs with torchao | essentially DIY | cleaner science, higher risk |
| MediaPipe LLM Inference | limited model support | easiest | check Qwen/SmolLM support first |
| MLC-LLM | good | own app, bridge non-trivial | fallback |

## Ship the model as an asset, not in the APK

A 400 MB APK is painful to build, transfer and reinstall a dozen times. Download
or side-load the model to app storage on first run.

## Memory

3 GB total, roughly 1.4-1.5 GB usable after OS and app. Peak RAM is weights
plus KV cache, and KV cache scales with context length — so keep the context
window short on device. Measure with real tooling; do not trust file size as a
proxy for peak RSS.

## Benchmarks for Day 13

Model file size, peak RSS, cold-start time, time-to-first-token, tokens/second.
**Repeat each measurement at least five times and report the median.** Low-end
phones throttle thermally, so a single measurement is noise.
