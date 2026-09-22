"""Safe NumPy interchange helpers for the command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .core import NoteFlowResult


def load_input(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Load scores, counts, and optional row IDs without pickle support."""
    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as archive:
        missing = {"scores", "counts"} - set(archive.files)
        if missing:
            raise ValueError(f"input archive is missing: {', '.join(sorted(missing))}")
        scores = np.array(archive["scores"], copy=True)
        counts = np.array(archive["counts"], copy=True)
        row_ids = np.array(archive["row_ids"], copy=True) if "row_ids" in archive else None
    return scores, counts, row_ids


def save_result(path: str | Path, result: NoteFlowResult) -> None:
    """Save method outputs as a compressed, non-pickled NumPy archive."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(result.diagnostics(), sort_keys=True, separators=(",", ":"))
    np.savez_compressed(
        output_path,
        assignment=result.assignment.columns,
        prices=result.soft_capacity.prices,
        column_mass=result.soft_capacity.column_mass,
        adjusted_scores=result.ranking.adjusted_scores,
        rankings=result.ranking.rankings,
        ranking_offsets=result.ranking.offsets,
        metadata=np.asarray(metadata),
    )


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Read only the JSON diagnostics from an output archive."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return json.loads(str(archive["metadata"].item()))

