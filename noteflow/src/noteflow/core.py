"""End-to-end implementation of the three NoteFlow inference stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .soft_capacity import SoftCapacityResult, fit_soft_capacity
from .validation import validate_packed


@dataclass(frozen=True)
class NoteFlowConfig:
    """Method parameters; defaults reproduce the paper's locked setting."""

    temperature: float = 0.3
    strength: float = 1.0
    top_l: int | None = 64
    max_iter: int = 1000
    tolerance: float = 1e-3
    damping: float = 1.0
    max_dense_elements: int | None = 100_000_000


@dataclass(frozen=True)
class AssignmentResult:
    """Exact full-graph maximum-weight assignment."""

    columns: NDArray[np.int64]
    objective: float


@dataclass(frozen=True)
class RankingResult:
    """Packed complete rankings and adjusted scores."""

    rankings: NDArray[np.int64]
    offsets: NDArray[np.int64]
    adjusted_scores: NDArray[np.float64]

    def row(self, index: int) -> NDArray[np.int64]:
        """Return a view of one query's complete ranking."""
        if index < 0 or index >= len(self.offsets) - 1:
            raise IndexError(index)
        return self.rankings[self.offsets[index] : self.offsets[index + 1]]


@dataclass(frozen=True)
class NoteFlowResult:
    """All reusable outputs from a NoteFlow run."""

    assignment: AssignmentResult
    soft_capacity: SoftCapacityResult
    ranking: RankingResult
    config: NoteFlowConfig = field(repr=False)

    def ranking_for(self, index: int) -> NDArray[np.int64]:
        return self.ranking.row(index)

    def diagnostics(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "assignment_objective": self.assignment.objective,
            "queries": len(self.assignment.columns),
            **self.soft_capacity.diagnostics(),
        }


def _linear_sum_assignment(costs: NDArray[np.float64]) -> NDArray[np.int64]:
    """Exact rectangular Hungarian algorithm for rows <= columns.

    This small internal solver keeps the reusable method dependent only on
    NumPy. Infinite entries represent illegal edges. Stable scans prefer the
    lowest column whenever the reduced costs tie.
    """
    row_count, column_count = costs.shape
    if row_count > column_count:
        raise ValueError("a row-covering assignment requires at least as many columns as rows")

    # Potentials u/v and matching p use the conventional one-based Hungarian
    # representation. p[j] is the row currently assigned to column j.
    u = np.zeros(row_count + 1, dtype=np.float64)
    v = np.zeros(column_count + 1, dtype=np.float64)
    p = np.zeros(column_count + 1, dtype=np.int64)
    way = np.zeros(column_count + 1, dtype=np.int64)

    for row in range(1, row_count + 1):
        p[0] = row
        minimum = np.full(column_count + 1, np.inf, dtype=np.float64)
        used = np.zeros(column_count + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            active_row = p[column]
            reduced = costs[active_row - 1] - u[active_row] - v[1:]
            available = ~used[1:]
            better = available & (reduced < minimum[1:])
            minimum[1:][better] = reduced[better]
            way[1:][better] = column

            available_indices = np.flatnonzero(available) + 1
            if len(available_indices) == 0:
                raise RuntimeError("candidate graph has no row-covering assignment")
            delta = float(np.min(minimum[available_indices]))
            if not np.isfinite(delta):
                raise RuntimeError("candidate graph has no row-covering assignment")
            # flatnonzero preserves ascending columns for deterministic ties.
            tied = available_indices[minimum[available_indices] == delta]
            next_column = int(tied[0])

            used_indices = np.flatnonzero(used)
            u[p[used_indices]] += delta
            v[used_indices] -= delta
            minimum[np.flatnonzero(~used)] -= delta
            column = next_column
            if p[column] == 0:
                break

        while True:
            previous_column = int(way[column])
            p[column] = p[previous_column]
            column = previous_column
            if column == 0:
                break

    assignment = np.empty(row_count, dtype=np.int64)
    for column in range(1, column_count + 1):
        if p[column] != 0:
            assignment[p[column] - 1] = column - 1
    return assignment


def max_weight_assignment(
    scores: ArrayLike,
    counts: ArrayLike,
    *,
    max_dense_elements: int | None = 100_000_000,
) -> AssignmentResult:
    """Solve the full legal prefix graph as an exact rectangular assignment.

    The implementation intentionally does not prune scores in this stage. It
    constructs a dense row-by-column cost matrix and uses an exact rectangular
    Hungarian solver. ``max_dense_elements`` is a memory guard, not an
    approximation knob; raise it or set it to ``None`` only when the machine
    can hold the requested matrix.
    """
    normalized_scores, normalized_counts, offsets = validate_packed(scores, counts)
    row_count = len(normalized_counts)
    column_count = int(normalized_counts.max())
    dense_elements = row_count * column_count
    if max_dense_elements is not None:
        if (
            not isinstance(max_dense_elements, (int, np.integer))
            or isinstance(max_dense_elements, bool)
            or max_dense_elements < 1
        ):
            raise ValueError("max_dense_elements must be a positive integer or None")
        if dense_elements > max_dense_elements:
            gib = dense_elements * np.dtype(np.float64).itemsize / 1024**3
            raise MemoryError(
                f"exact assignment needs {dense_elements:,} float64 cells "
                f"(~{gib:.2f} GiB), above max_dense_elements={max_dense_elements:,}"
            )

    costs = np.full((row_count, column_count), np.inf, dtype=np.float64)
    for row_index, count in enumerate(normalized_counts):
        row_scores = normalized_scores[offsets[row_index] : offsets[row_index + 1]]
        costs[row_index, :count] = np.max(row_scores) - row_scores

    columns = _linear_sum_assignment(costs)
    row_indices = np.arange(row_count)
    if not np.isfinite(costs[row_indices, columns]).all():
        raise RuntimeError("assignment solver selected an illegal edge")

    selected_positions = offsets[:-1] + columns
    objective = float(normalized_scores[selected_positions].sum(dtype=np.float64))
    return AssignmentResult(columns=columns.astype(np.int64, copy=False), objective=objective)


def fixed_first_rankings(
    scores: ArrayLike,
    counts: ArrayLike,
    assignment: ArrayLike,
    prices: ArrayLike,
    *,
    strength: float = 1.0,
) -> RankingResult:
    """Rank every candidate by adjusted score while fixing assignment first."""
    normalized_scores, normalized_counts, offsets = validate_packed(scores, counts)
    chosen = np.asarray(assignment)
    normalized_prices = np.asarray(prices, dtype=np.float64)
    if chosen.ndim != 1 or len(chosen) != len(normalized_counts):
        raise ValueError("assignment must be one-dimensional and match counts")
    integer_chosen = chosen.astype(np.int64)
    if not np.array_equal(integer_chosen, chosen):
        raise ValueError("assignment must contain integer candidate columns")
    if np.any(integer_chosen < 0) or np.any(integer_chosen >= normalized_counts):
        raise ValueError("assignment contains an illegal candidate column")
    if len(np.unique(integer_chosen)) != len(integer_chosen):
        raise ValueError("assignment must not reuse a candidate column")
    if normalized_prices.ndim != 1 or len(normalized_prices) < int(normalized_counts.max()):
        raise ValueError("prices must cover the common candidate universe")
    if not np.isfinite(normalized_prices).all():
        raise ValueError("prices must be finite")
    if not np.isfinite(strength) or strength < 0:
        raise ValueError("strength must be finite and nonnegative")

    adjusted = np.empty_like(normalized_scores)
    rankings = np.empty(len(normalized_scores), dtype=np.int64)
    for row_index, count in enumerate(normalized_counts):
        start, stop = int(offsets[row_index]), int(offsets[row_index + 1])
        row_adjusted = (
            normalized_scores[start:stop] - float(strength) * normalized_prices[:count]
        )
        adjusted[start:stop] = row_adjusted
        # Stable sorting makes equal adjusted scores prefer lower columns.
        order = np.argsort(-row_adjusted, kind="stable")
        selected = integer_chosen[row_index]
        rankings[start] = selected
        rankings[start + 1 : stop] = order[order != selected]

    return RankingResult(rankings=rankings, offsets=offsets, adjusted_scores=adjusted)


def run_noteflow(
    scores: ArrayLike,
    counts: ArrayLike,
    *,
    row_ids: ArrayLike | None = None,
    config: NoteFlowConfig | None = None,
) -> NoteFlowResult:
    """Run full-graph assignment, price estimation, and fixed-first ranking."""
    selected_config = config or NoteFlowConfig()
    assignment = max_weight_assignment(
        scores,
        counts,
        max_dense_elements=selected_config.max_dense_elements,
    )
    soft_capacity = fit_soft_capacity(
        scores,
        counts,
        temperature=selected_config.temperature,
        top_l=selected_config.top_l,
        max_iter=selected_config.max_iter,
        tolerance=selected_config.tolerance,
        damping=selected_config.damping,
        row_ids=row_ids,
    )
    ranking = fixed_first_rankings(
        scores,
        counts,
        assignment.columns,
        soft_capacity.prices,
        strength=selected_config.strength,
    )
    return NoteFlowResult(
        assignment=assignment,
        soft_capacity=soft_capacity,
        ranking=ranking,
        config=selected_config,
    )
