# Mobile

## What is already here

```
mobile/lib/guard.dart              Dart port of the safety guard
mobile/assets/rules.json           COPY of src/safety/rules.json — never edit directly
mobile/test/guard_test.dart        Parity test against the Python implementation
mobile/test/guard_parity_cases.json Golden corpus (36 input, 8 output, 26 normalisation)
mobile/make_parity.py              Regenerate the corpus after a rules change
mobile/sync_rules.sh               Copy rules + retest + regenerate, in one command
mobile/pubspec.yaml                Dependencies
```

## The parity test, and why it decides whether your results are valid

Your reported safety numbers come from the **Python** guard, running on your
laptop during Day 11. The app ships the **Dart** guard. If the two behave
differently, the system you demonstrate is not the system you measured, and the
safety section of your results does not describe the deliverable.

`guard_test.dart` replays a golden corpus generated from the Python
implementation and fails on any divergence — block decision, rule id, category,
or normalised text.

```bash
./mobile/sync_rules.sh                    # after any rules.json edit
dart test mobile/test/guard_test.dart     # must be all green before Day 12
```

Two divergences already found and closed, both of which would have been silent:

- **Unicode normalisation.** Python used NFKD; Dart has no core equivalent. The
  fold map now lives in `rules.json` so both run identical preprocessing. This
  is not academic — francophone users type "problème" and "à jeun", and neither
  matches without folding.
- **Rule ordering.** Python's sort is stable; Dart's `List.sort` is not. Rules
  sharing a priority could evaluate in a different order on device. Both now
  sort on `(priority, id)`, which is a total key.

Mention the parity test in your defence. "The on-device guard is verified
against the implementation used to produce the results" is a stronger answer
than "I ported it carefully."

## Day 3 spike, not Day 12

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
