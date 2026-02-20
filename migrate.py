"""
migrate.py — Reorganise a flat PySpark project into the
coursework directory structure required by 7006SCN.

Safe to run multiple times:
  - Creates directories only if they don't exist.
  - Never overwrites — appends '_old' when a clash is detected.
  - Skips files that are already in their target location.

Usage:
    python migrate.py            (runs in the current directory)
    python migrate.py /some/path (runs in the given directory)
"""

import os
import shutil
import sys
from pathlib import Path


# ── Configuration ────────────────────────────────────────────

REQUIRED_DIRS = [
    "notebooks",
    "src",
    "data_raw",
    "data_parquet",
    "results",
]

# keyword  →  target path under project root
# Checked in order; first match wins.
NOTEBOOK_RULES = [
    ("ingestion",    "notebooks/01_ingestion.ipynb"),
    ("preprocess",   "notebooks/02_preprocessing.ipynb"),
    ("feature",      "notebooks/03_feature_engineering.ipynb"),
    ("model",        "notebooks/04_model_training.ipynb"),
    ("eval",         "notebooks/05_evaluation_scaling.ipynb"),
    ("scaling",      "notebooks/05_evaluation_scaling.ipynb"),
]

PY_RULES = [
    ("preprocessing", "src/preprocessing.py"),
    ("preprocess",    "src/preprocessing.py"),
    ("feature",       "src/features.py"),
    ("model",         "src/models.py"),
]


# ── Helpers ──────────────────────────────────────────────────

def safe_destination(dest: Path) -> Path:
    """
    If *dest* already exists, return a path with '_old' inserted
    before the extension (repeating if necessary).
    """
    if not dest.exists():
        return dest

    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    candidate = parent / f"{stem}_old{suffix}"

    # Handle the (unlikely) case where _old also exists
    counter = 2
    while candidate.exists():
        candidate = parent / f"{stem}_old{counter}{suffix}"
        counter += 1

    return candidate


def match_rule(filename_lower: str, rules: list) -> str | None:
    """Return the first matching target path, or None."""
    for keyword, target in rules:
        if keyword in filename_lower:
            return target
    return None


def move_file(src: Path, dest: Path) -> None:
    """Move *src* to *dest*, resolving clashes safely."""
    if src.resolve() == dest.resolve():
        print(f"  SKIP  (already in place) {src.name}")
        return

    dest = safe_destination(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    print(f"  MOVE  {src.name}  ->  {dest.relative_to(dest.parent.parent)}")


# ── Main ─────────────────────────────────────────────────────

def migrate(root: Path) -> None:
    print(f"Project root: {root.resolve()}\n")

    # 1. Create required directories
    print("Creating directories …")
    for d in REQUIRED_DIRS:
        target = root / d
        target.mkdir(parents=True, exist_ok=True)
        print(f"  {str(d):20s}  OK")
    print()

    # 2. Collect candidate files (top-level only, not inside
    #    sub-folders that are already part of the structure)
    candidates = [
        p for p in root.iterdir()
        if p.is_file() and p.name != "migrate.py"
    ]

    # 3. Move notebooks
    print("Moving notebooks …")
    moved_notebooks = 0
    for path in candidates:
        if path.suffix.lower() != ".ipynb":
            continue
        target_rel = match_rule(path.name.lower(), NOTEBOOK_RULES)
        if target_rel is None:
            print(f"  SKIP  (no rule matched) {path.name}")
            continue
        move_file(path, root / target_rel)
        moved_notebooks += 1

    if moved_notebooks == 0:
        print("  (nothing to move)")
    print()

    # 4. Move Python source files
    print("Moving Python sources …")
    moved_py = 0
    for path in candidates:
        if path.suffix.lower() != ".py":
            continue
        target_rel = match_rule(path.name.lower(), PY_RULES)
        if target_rel is None:
            print(f"  SKIP  (no rule matched) {path.name}")
            continue
        move_file(path, root / target_rel)
        moved_py += 1

    if moved_py == 0:
        print("  (nothing to move)")
    print()

    # 5. Summary
    print("-" * 50)
    print("Done.  Final structure:\n")
    for d in sorted(root.rglob("*")):
        # Only show the managed directories + their direct children
        rel = d.relative_to(root)
        parts = rel.parts
        if parts[0] in REQUIRED_DIRS and len(parts) <= 2:
            indent = "  " * (len(parts) - 1)
            marker = "[DIR]" if d.is_dir() else "[FILE]"
            print(f"  {indent}{marker} {parts[-1]}")
        elif len(parts) == 1 and d.is_dir() and parts[0] in REQUIRED_DIRS:
            print(f"  [DIR] {parts[0]}/")


if __name__ == "__main__":
    project_root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    migrate(project_root)
