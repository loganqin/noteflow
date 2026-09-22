"""Shared validation for packed prefix candidate graphs."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_packed(
    scores: ArrayLike,
    counts: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.int64], NDArray[np.int64]]:
    """Validate and normalize a packed prefix graph.

    Returns float64 scores, integer counts, and packed row offsets. The method
    requires a nonempty graph with an injective row-covering assignment.
    """
    normalized_scores = np.asarray(scores, dtype=np.float64)
    raw_counts = np.asarray(counts)
    if raw_counts.ndim != 1 or len(raw_counts) == 0:
        raise ValueError("counts must be a nonempty one-dimensional array")
    if not np.issubdtype(raw_counts.dtype, np.number):
        raise ValueError("counts must contain positive integers")
    if not np.isfinite(raw_counts).all():
        raise ValueError("counts must contain positive integers")
    normalized_counts = raw_counts.astype(np.int64)
    if not np.array_equal(normalized_counts, raw_counts) or np.any(normalized_counts < 1):
        raise ValueError("counts must contain positive integers")
    if normalized_scores.ndim != 1:
        raise ValueError("scores must be a one-dimensional packed array")
    if len(normalized_scores) != int(normalized_counts.sum(dtype=np.int64)):
        raise ValueError("scores must contain exactly sum(counts) entries")
    if not np.isfinite(normalized_scores).all():
        raise ValueError("scores must be finite")

    ordered_counts = np.sort(normalized_counts)
    if np.any(ordered_counts < np.arange(1, len(ordered_counts) + 1)):
        raise ValueError("prefix candidate graph has no injective row-covering assignment")

    offsets = np.empty(len(normalized_counts) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(normalized_counts, dtype=np.int64, out=offsets[1:])
    return normalized_scores, normalized_counts, offsets


def normalize_row_ids(row_ids: Any, row_count: int) -> np.ndarray:
    """Validate stable public row identifiers used for deterministic ties."""
    if row_ids is None:
        return np.arange(row_count, dtype=np.int64)
    ids = np.asarray(row_ids)
    if ids.ndim != 1 or len(ids) != row_count:
        raise ValueError("row_ids must be one-dimensional and match counts")
    try:
        unique_count = len(np.unique(ids))
    except TypeError as exc:
        raise ValueError("row_ids must have a single comparable dtype") from exc
    if unique_count != row_count:
        raise ValueError("row_ids must be unique")
    return ids

