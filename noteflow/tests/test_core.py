import itertools

import numpy as np
import pytest

from noteflow import (
    NoteFlowConfig,
    fixed_first_rankings,
    max_weight_assignment,
    run_noteflow,
)


def _brute_force_objective(rows):
    best = -np.inf
    for columns in itertools.permutations(range(max(map(len, rows))), len(rows)):
        if all(column < len(rows[row]) for row, column in enumerate(columns)):
            best = max(best, sum(rows[row][column] for row, column in enumerate(columns)))
    return best


def test_exact_assignment_matches_brute_force_on_unsorted_prefixes():
    rows = [np.array([1.0, 4.0, 0.0]), np.array([3.0]), np.array([5.0, 2.0])]
    scores = np.concatenate(rows)
    result = max_weight_assignment(scores, [3, 1, 2])
    assert len(np.unique(result.columns)) == 3
    assert result.objective == pytest.approx(_brute_force_objective(rows))


def test_exact_assignment_matches_random_small_brute_force_instances():
    rng = np.random.default_rng(20260922)
    for row_count in range(1, 5):
        for _ in range(10):
            counts = np.sort(rng.integers(1, row_count + 3, size=row_count))
            counts = np.maximum(counts, np.arange(1, row_count + 1))
            rng.shuffle(counts)
            rows = [rng.normal(size=int(count)) for count in counts]
            result = max_weight_assignment(
                np.concatenate(rows), counts, max_dense_elements=None
            )
            assert result.objective == pytest.approx(_brute_force_objective(rows))


def test_end_to_end_result_is_feasible_complete_and_fixed_first():
    rows = np.array([[1.0, 0.99, -1.0], [1.0, 0.1, -1.0]])
    result = run_noteflow(
        rows.ravel(),
        [3, 3],
        row_ids=["b", "a"],
        config=NoteFlowConfig(top_l=None, max_iter=3000, tolerance=1e-8),
    )
    assert result.soft_capacity.converged
    assert len(np.unique(result.assignment.columns)) == 2
    for row_index in range(2):
        ranking = result.ranking_for(row_index)
        assert ranking[0] == result.assignment.columns[row_index]
        np.testing.assert_array_equal(np.sort(ranking), np.arange(3))
    assert result.diagnostics()["queries"] == 2


def test_fixed_first_rejects_reused_assignment():
    with pytest.raises(ValueError, match="reuse"):
        fixed_first_rankings([1.0, 0.0, 1.0, 0.0], [2, 2], [0, 0], [0.0, 0.0])


def test_dense_memory_guard_is_explicit():
    with pytest.raises(MemoryError, match="exact assignment needs"):
        max_weight_assignment(np.zeros(9), [3, 3, 3], max_dense_elements=8)


@pytest.mark.parametrize(
    "scores,counts",
    [
        ([1.0, 2.0], [1, 1]),
        ([1.0, np.nan], [2]),
        ([1.0], [1.5]),
        ([1.0], []),
    ],
)
def test_invalid_packed_inputs_are_rejected(scores, counts):
    with pytest.raises(ValueError):
        max_weight_assignment(scores, counts)
