import numpy as np
import pytest

from noteflow import fit_soft_capacity


def test_symmetric_two_by_two_prices_match_analytic_solution():
    # Both rows have the same one-point preference for column 0. Capacity
    # requires a 50/50 soft split, so p0 - p1 must be exactly one; the
    # nonnegative minimum-price representative has p1 == 0.
    rows = np.array([[1.0, 0.0], [1.0, 0.0]])
    result = fit_soft_capacity(
        rows.ravel(),
        [2, 2],
        temperature=0.7,
        top_l=None,
        max_iter=5000,
        tolerance=1e-10,
    )
    assert result.converged
    np.testing.assert_allclose(result.prices, [1.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(result.column_mass, [1.0, 1.0], atol=1e-10)


def test_sparse_support_is_equivariant_to_query_permutation():
    rows = np.array(
        [
            [1, 0.1, 0.3, 0.2, -0.1],
            [1, 0.2, 0.1, 0.3, -0.2],
            [1, 0.3, 0.2, 0.1, -0.3],
        ],
        dtype=float,
    )
    ids = np.array(["w_b", "w_c", "w_a"])
    permutation = [2, 0, 1]
    settings = dict(temperature=0.4, top_l=2, max_iter=1000)
    first = fit_soft_capacity(rows.ravel(), [5] * 3, row_ids=ids, **settings)
    second = fit_soft_capacity(
        rows[permutation].ravel(), [5] * 3, row_ids=ids[permutation], **settings
    )
    assert first.approximate and first.support_edges < first.full_edges
    np.testing.assert_allclose(first.prices, second.prices, atol=1e-12)
    np.testing.assert_allclose(first.column_mass, second.column_mass, atol=1e-12)
    assert first.prices[4] == 0


def test_finite_iteration_nonconvergence_is_reported():
    rows = np.array([[1e4, 9999, -1e4], [1e4, 9998, -1e4]], dtype=np.float32)
    result = fit_soft_capacity(
        rows.ravel(), [3, 3], temperature=0.01, top_l=None, max_iter=20
    )
    assert not result.converged
    assert result.iterations == 20
    assert np.isfinite(result.prices).all()
    assert "prices" not in result.diagnostics()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0},
        {"top_l": 0},
        {"max_iter": -1},
        {"tolerance": 0},
        {"damping": 2},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        fit_soft_capacity([1.0, 0.0], [2], **kwargs)
