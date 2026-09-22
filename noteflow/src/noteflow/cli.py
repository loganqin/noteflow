"""Command-line entry point for data-independent NoteFlow inference."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from . import __version__
from .core import NoteFlowConfig, run_noteflow
from .io import load_input, save_result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NoteFlow on packed prefix-candidate scores."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--input", type=Path, required=True, help="input .npz")
    parser.add_argument("--output", type=Path, required=True, help="output .npz")
    parser.add_argument("--config", type=Path, help="JSON method configuration")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--top-l", type=int)
    parser.add_argument("--full-support", action="store_true")
    parser.add_argument("--max-iter", type=int)
    parser.add_argument("--tolerance", type=float)
    parser.add_argument("--damping", type=float)
    parser.add_argument("--max-dense-elements", type=int)
    return parser


def _load_config(args: argparse.Namespace) -> NoteFlowConfig:
    values: dict[str, Any] = asdict(NoteFlowConfig())
    if args.config is not None:
        loaded = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("configuration JSON must contain an object")
        unknown = set(loaded) - set(values)
        if unknown:
            raise ValueError(f"unknown configuration keys: {', '.join(sorted(unknown))}")
        values.update(loaded)

    for key in (
        "temperature",
        "strength",
        "top_l",
        "max_iter",
        "tolerance",
        "damping",
        "max_dense_elements",
    ):
        command_value = getattr(args, key)
        if command_value is not None:
            values[key] = command_value
    if args.full_support:
        values["top_l"] = None
    return NoteFlowConfig(**values)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_config(args)
    scores, counts, row_ids = load_input(args.input)
    result = run_noteflow(scores, counts, row_ids=row_ids, config=config)
    save_result(args.output, result)
    print(json.dumps(result.diagnostics(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

