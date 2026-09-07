import argparse, json, sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("path", type=str)
parser.add_argument("--kind", choices=["train", "eval", "test", "seed"], default="train")
args = parser.parse_args()

p = Path(args.path)
if not p.exists():
    print(f"Error: File '{p}' not found.")
    sys.exit(1)

lines = [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
if not lines:
    print(f"Error: '{p}' is empty.")
    sys.exit(1)

for i, line in enumerate(lines, 1):
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        print(f"Line {i}: Invalid JSON - {e}")
        sys.exit(1)

print(f"[{args.kind}] Validation OK — {len(lines)} records checked in {p}")