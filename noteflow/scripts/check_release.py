"""Fail when a source release contains data or generated experiment artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BLOCKED_DIRECTORIES = {
    "data",
    "datasets",
    "inputs",
    "outputs",
    "artifacts",
    "cache",
    "checkpoints",
    "models",
    "results",
}
IGNORED_DEVELOPMENT_DIRECTORIES = {
    ".git",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}
BLOCKED_SUFFIXES = {
    ".csv",
    ".tsv",
    ".parquet",
    ".feather",
    ".npy",
    ".npz",
    ".pkl",
    ".pickle",
    ".pt",
    ".pth",
    ".ckpt",
}


def main() -> int:
    violations: list[str] = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if any(
            part in IGNORED_DEVELOPMENT_DIRECTORIES or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if any(part in BLOCKED_DIRECTORIES for part in relative.parts):
            violations.append(str(relative))
        elif path.is_file() and path.suffix.lower() in BLOCKED_SUFFIXES:
            violations.append(str(relative))
    if violations:
        print("release audit failed; remove these paths:", file=sys.stderr)
        print("\n".join(f"- {path}" for path in sorted(set(violations))), file=sys.stderr)
        return 1
    print("release audit passed: no project data, checkpoints, caches, or results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
