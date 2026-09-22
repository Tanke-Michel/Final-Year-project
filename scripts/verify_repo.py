"""
Check that this copy of the project is complete and current.

    python3 scripts/verify_repo.py            # check against MANIFEST.json
    python3 scripts/verify_repo.py --write    # regenerate the manifest (maintainer)

WHY. A copy assembled file by file — downloaded individually, uploaded through
a web form, merged from different versions — fails in confusing ways far from
the cause: "No rule to make target 'test'", "can't open file test_guard.py",
checks that pass because the thing they test is absent. This names the missing
and outdated files directly, before anything else runs.

  missing   -> exit 1. The project cannot work; replace it with the full zip.
  modified  -> warning. Fine if you edited the file on purpose; if not, you
               have an older version of it.

Files you add (your dataset, results, runs) are not checked. Line endings are
normalised before hashing, so a Windows checkout does not read as modified.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "MANIFEST.json"
SKIP_DIRS = {".git", "__pycache__", "results", "runs", ".ipynb_checkpoints"}
SKIP_FILES = {"MANIFEST.json"}


def digest(path: Path) -> str:
    raw = path.read_bytes()
    try:
        raw = raw.decode("utf-8").replace("\r\n", "\n").encode("utf-8")
    except UnicodeDecodeError:
        pass                                  # binary: hash as is
    return hashlib.sha256(raw).hexdigest()


def tracked() -> list[Path]:
    out = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if p.is_file() and not (set(rel.parts) & SKIP_DIRS) and rel.name not in SKIP_FILES \
                and not rel.name.endswith((".pyc", ".aux", ".log", ".bbl", ".blg", ".out")):
            out.append(p)
    return out


def main() -> int:
    if "--write" in sys.argv:
        files = {str(p.relative_to(ROOT)): digest(p) for p in tracked()}
        MANIFEST.write_text(json.dumps({"files": files}, indent=1, sort_keys=True) + "\n")
        print(f"wrote {MANIFEST.name}: {len(files)} files")
        return 0

    if not MANIFEST.exists():
        print("MANIFEST.json is missing, so completeness cannot be checked.")
        print("Your copy predates it or is incomplete — use the full project zip.")
        return 1

    expected = json.loads(MANIFEST.read_text())["files"]
    missing = [f for f in expected if not (ROOT / f).is_file()]
    modified = [f for f in expected if (ROOT / f).is_file() and digest(ROOT / f) != expected[f]]

    print(f"  {len(expected) - len(missing) - len(modified)}/{len(expected)} project files "
          f"present and current")
    if modified:
        print(f"\n  MODIFIED ({len(modified)}) — fine if you edited these on purpose;")
        print("  otherwise you have an older version:")
        for f in modified:
            print(f"    ~ {f}")
    if missing:
        print(f"\n  MISSING ({len(missing)}):")
        for f in missing:
            print(f"    - {f}")
        print("\n  This copy is incomplete. Replace it with the full project zip —")
        print("  copying files one by one is how they go missing. See docs/KAGGLE.md.")
        return 1
    if not modified:
        print("  complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
