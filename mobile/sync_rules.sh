#!/usr/bin/env bash
# Sync the single sources of truth into the Flutter asset bundle, then retest.
# Run after ANY change to rules.json or the inference block of base.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
cp src/safety/rules.json mobile/assets/rules.json
python3 mobile/export_config.py
python3 src/safety/test_guard.py
python3 mobile/make_parity.py
echo
echo "Now run: dart test mobile/test/guard_test.dart"
