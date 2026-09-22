"""Sparse entropy-regularized deposit-price estimation.

The implementation follows the Deposit-Price Estimation stage of NoteFlow.
It reads only fixed candidate scores, prefix counts, and optional public row
identifiers. It does not read correspondence labels or controller identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .validation import normalize_row_ids, validate_packed


@dataclass(frozen=True)
class SoftCapacityResult:
    """Prices and finite-iteration diagnostics for the sparse relaxation."""

    prices: NDArray[np.float64]
    column_mass: NDArray[np.float64]
    iterations: int
    converged: bool
    capacity_violation: float
    kkt_residual: float
    support_edges: int
    full_edges: int
    temperature: float
    approximate: bool

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-compatible diagnostics without the two large arrays."""
        result = asdict(self)
        result.pop("prices")
        result.pop("column_mass")
        return result


def _top_indices(values: NDArray[np.float64], k: int) -> NDArray[np.int64]:
    """Select top scores, breaking boundary ties by the lowest column."""
    if k >= len(values):
        return np.arange(len(values), dtype=np.int64)
    boundary = np.partition(values, len(values) - k)[len(values) - k]
    better = np.flatnonzero(values > boundary)
    tied = np.flatnonzero(values == boundary)[: k - len(better)]
    return np.sort(np.concatenate((better, tied)))


def _build_support(
    scores: NDArray[np.float64],
    counts: NDArray[np.int64],
    offsets: NDArray[np.int64],
    top_l: int | None,
    row_ids: np.ndarray,
) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.int64]]:
    # Sort by candidate count, then public row ID. The r-th row receives
    # backbone column r, which is legal because prefix feasibility was checked.
    try:
        order = np.lexsort((row_ids, counts))
    except TypeError as exc:
        raise ValueError("row_ids must be comparable for deterministic tie breaking") from exc
    backbone = np.empty(len(counts), dtype=np.int64)
    backbone[order] = np.arange(len(counts), dtype=np.int64)

    columns: list[NDArray[np.int64]] = []
    centered_values: list[NDArray[np.float64]] = []
    lengths: list[int] = []
    for row_index, count in enumerate(counts):
        row = scores[offsets[row_index] : offsets[row_index + 1]]
        keep_count = int(count) if top_l is None else min(top_l, int(count))
        keep = _top_indices(row, keep_count)
        if not np.any(keep == backbone[row_index]):
            keep = np.sort(np.append(keep, backbone[row_index]))
        columns.append(keep)
        centered_values.append(row[keep] - np.max(row))
        lengths.append(len(keep))

    support_offsets = np.empty(len(lengths) + 1, dtype=np.int64)
    support_offsets[0] = 0
    np.cumsum(lengths, dtype=np.int64, out=support_offsets[1:])
    return np.concatenate(columns), np.concatenate(centered_values), support_offsets


def _column_marginals(
    log_kernel: NDArray[np.float64],
    columns: NDArray[np.int64],
    offsets: NDArray[np.int64],
    scaled_prices: NDArray[np.float64],
    column_count: int,
) -> NDArray[np.float64]:
    logits = log_kernel - scaled_prices[columns]
    row_maximum = np.maximum.reduceat(logits, offsets[:-1])
    stabilized = logits - np.repeat(row_maximum, np.diff(offsets))
    weights = np.exp(stabilized)
    row_sum = np.add.reduceat(weights, offsets[:-1])
    weights /= np.repeat(row_sum, np.diff(offsets))
    return np.bincount(columns, weights=weights, minlength=column_count)


def fit_soft_capacity(
    scores: ArrayLike,
    counts: ArrayLike,
    *,
    temperature: float = 0.3,
    top_l: int | None = 64,
    max_iter: int = 1000,
    tolerance: float = 1e-3,
    damping: float = 1.0,
    row_ids: ArrayLike | None = None,
) -> SoftCapacityResult:
    """Estimate nonnegative reusable competition prices.

    Prices are estimated on each row's top-``top_l`` edges plus a public,
    feasible prefix-matching backbone. Set ``top_l=None`` to use the full
    graph. Finite-iteration non-convergence is returned explicitly rather than
    hidden; the resulting values are still usable as approximate prices.
    """
    normalized_scores, normalized_counts, offsets = validate_packed(scores, counts)
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if top_l is not None and (
        not isinstance(top_l, (int, np.integer)) or isinstance(top_l, bool) or top_l < 1
    ):
        raise ValueError("top_l must be a positive integer or None")
    if (
        not isinstance(max_iter, (int, np.integer))
        or isinstance(max_iter, bool)
        or max_iter < 0
    ):
        raise ValueError("max_iter must be a nonnegative integer")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not np.isfinite(damping) or not 0 < damping <= 1:
        raise ValueError("damping must be in (0, 1]")

    ids = normalize_row_ids(row_ids, len(normalized_counts))
    columns, centered, support_offsets = _build_support(
        normalized_scores, normalized_counts, offsets, top_l, ids
    )
    log_kernel = centered / float(temperature)
    if not np.isfinite(log_kernel).all():
        raise ValueError("score span divided by temperature exceeds floating-point range")

    column_count = int(normalized_counts.max())
    # q = price / temperature is dimensionless.
    q = np.zeros(column_count, dtype=np.float64)
    mass = _column_marginals(log_kernel, columns, support_offsets, q, column_count)
    iteration = 0
    while True:
        capacity_violation = float(np.maximum(mass - 1.0, 0.0).max(initial=0.0))
        active = q > tolerance
        complementarity = (
            float(np.abs(mass[active] - 1.0).max(initial=0.0)) if active.any() else 0.0
        )
        residual = max(capacity_violation, complementarity)
        if residual <= tolerance or iteration >= max_iter:
            break
        safe_mass = np.maximum(mass, np.finfo(np.float64).tiny)
        new_q = np.maximum(0.0, q + np.log(safe_mass))
        new_q[mass == 0] = 0.0
        q += float(damping) * (new_q - q)
        mass = _column_marginals(log_kernel, columns, support_offsets, q, column_count)
        iteration += 1

    return SoftCapacityResult(
        prices=q * float(temperature),
        column_mass=mass,
        iterations=iteration,
        converged=residual <= tolerance,
        capacity_violation=capacity_violation,
        kkt_residual=residual,
        support_edges=len(columns),
        full_edges=len(normalized_scores),
        temperature=float(temperature),
        approximate=len(columns) != len(normalized_scores),
    )

