"""Detect mock figures before they can reach the report.

Two independent checks, because the obvious one does not work:

  1. MARKER FILE. `src/figures.py --mock` writes MOCK_DATA_DO_NOT_PUBLISH into
     the figures directory and deletes it on a real run. Cheap and reliable.

  2. DEEP SCAN. Decompresses PDF content streams and looks for the watermark
     text. This exists because grepping a PDF for "MOCK" silently finds nothing
     — matplotlib compresses text streams, so a naive grep guard passes every
     mock figure straight through. That was a real defect in this repo.

Exit 0 when clean, 1 when mock content is found.

    python3 scripts/check_mock.py results/figures
"""
import re
import sys
import zlib
from pathlib import Path

MARKER = "MOCK_DATA_DO_NOT_PUBLISH"


def pdf_has_mock(path: Path) -> bool:
    raw = path.read_bytes()
    if b"MOCK" in raw:                      # uncompressed case
        return True
    for m in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        try:
            if b"MOCK" in zlib.decompress(m.group(1)):
                return True
        except Exception:
            continue
    return False


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "results/figures")
    if not d.exists():
        print(f"  no figures directory at {d} — nothing to check")
        return 0

    problems = []

    if (d / MARKER).exists():
        problems.append(f"marker file present: {d / MARKER}")

    for f in sorted(list(d.glob("*.pdf")) + list(d.glob("*.png"))):
        try:
            if f.suffix == ".pdf" and pdf_has_mock(f):
                problems.append(f"mock watermark in {f.name}")
            elif f.suffix == ".png" and b"MOCK" in f.read_bytes():
                problems.append(f"mock watermark in {f.name}")
        except Exception as e:
            problems.append(f"could not read {f.name}: {e}")

    if problems:
        print("  MOCK CONTENT DETECTED:")
        for p in problems:
            print(f"    - {p}")
        print()
        print("  These come from synthetic data and must never appear in the")
        print("  report or the paper. Regenerate from real results:")
        print("    make clean-results   then rerun the pipeline without --mock")
        return 1

    print(f"  no mock content in {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
