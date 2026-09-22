# NoteFlow

NoteFlow is the standalone implementation of the method described in
*NoteFlow: Ranking Deposit--Withdrawal Links in Ethereum Mixers*. It turns
fixed candidate-pair scores from any upstream scorer into:

1. a globally optimal rank-one assignment that respects single-use deposit
   capacity;
2. nonnegative deposit prices that summarize competition across withdrawals;
3. a complete candidate ranking for every withdrawal, with the feasible
   assignment fixed at rank one.

This is a **code-only release**. It contains no transaction data, labels,
model checkpoints, candidate scores, cached prices, or experiment results.
Data will be distributed separately. See [DATA.md](DATA.md) for the exact
input contract and data-separation policy.

## Method overview

For withdrawal `i`, deposit `j`, and a fixed compatibility score `s_ij`,
NoteFlow runs three stages.

### 1. Full-Graph Assignment

The method solves a maximum-weight one-to-one assignment over every legal
candidate edge. Each withdrawal receives exactly one deposit, and each deposit
is used at most once. The selected deposit is denoted by `g_i`.

This stage never applies Top-L pruning. The reference implementation uses an
exact rectangular Hungarian solver.

### 2. Deposit-Price Estimation

An entropy-regularized capacity relaxation estimates a nonnegative price
`p_j` for each deposit. By default, estimation uses each row's 64 highest
scoring edges plus a deterministic feasible backbone. The backbone depends
only on candidate counts and optional stable public row identifiers; it does
not use pairing labels or controller identities.

Sparse price estimation is an explicit approximation. If the finite iteration
budget does not reach the requested tolerance, NoteFlow returns the prices and
reports non-convergence in its diagnostics.

### 3. Fixed-First Ranking

All original candidates are scored as

```text
a_ij = s_ij - strength * p_j
```

The final ranking is

```text
[g_i] + sort_descending(C_i without g_i, by a_ij)
```

The rank-one outputs therefore form one jointly feasible assignment. Lower
ranks are per-withdrawal alternatives; they are not additional jointly
feasible matchings.

## Requirements and installation

NoteFlow requires Python 3.10 or later and NumPy 1.24 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

To install the development dependencies:

```bash
python -m pip install -e '.[dev]'
```

## Quick start

Inputs use a packed one-dimensional representation. `counts[i]` is the number
of candidates for query `i`. Its scores occupy the corresponding consecutive
segment of `scores`. Candidate columns are prefixes of one shared, stably
ordered deposit universe: `0, ..., counts[i] - 1`.

```python
import numpy as np

from noteflow import NoteFlowConfig, run_noteflow

counts = np.array([3, 4, 4])
scores = np.concatenate([
    [2.0, 1.9, 0.1],
    [2.1, 1.2, 0.4, 0.0],
    [1.8, 1.7, 1.6, 0.3],
])

result = run_noteflow(
    scores,
    counts,
    row_ids=np.array(["withdrawal-a", "withdrawal-b", "withdrawal-c"]),
    config=NoteFlowConfig(),
)

print(result.assignment.columns)  # Jointly feasible rank-one choices
print(result.ranking_for(0))      # Complete ranking for query 0
print(result.diagnostics())       # Convergence and approximation status
```

The default configuration matches the paper's locked setting:

| Parameter | Default | Meaning |
|---|---:|---|
| `temperature` | `0.3` | Concentration used during price estimation |
| `strength` | `1.0` | Price contribution to the final ranking |
| `top_l` | `64` | Highest-scoring edges retained per row for price estimation |
| `max_iter` | `1000` | Maximum number of price updates |
| `tolerance` | `0.001` | Capacity and KKT stopping tolerance |
| `damping` | `1.0` | Price-update damping |

A complete synthetic example is available in
[examples/quickstart.py](examples/quickstart.py). The example is generated in
code and is not research data.

## Input contract

The prefix representation is part of the method's problem setting, not merely
a storage optimization. The graph must admit an injective assignment covering
all queries. After sorting `counts`, the one-based `r`-th count must be at least
`r`.

The Python API accepts:

| Argument | Shape | Description |
|---|---|---|
| `scores` | `(sum(counts),)` | Finite pair scores packed row by row |
| `counts` | `(num_queries,)` | Positive integer prefix lengths |
| `row_ids` | `(num_queries,)`, optional | Unique stable public IDs used only for deterministic backbone ties |
| `config` | scalar object, optional | A `NoteFlowConfig`; paper defaults are used when omitted |

Local candidate columns are returned by the implementation. Applications that
use transaction hashes or other external identifiers should maintain their own
column-to-identifier mapping outside this repository.

## Command-line interface

The CLI reads an external NumPy `.npz` archive containing `scores`, `counts`,
and optional `row_ids` arrays:

```bash
noteflow \
  --input ../noteflow-data/scores.npz \
  --output ../noteflow-data/noteflow-output.npz \
  --config configs/paper.json
```

The output archive contains:

| Array | Description |
|---|---|
| `assignment` | Feasible rank-one candidate columns |
| `prices` | Estimated nonnegative deposit prices |
| `column_mass` | Final soft mass on each supported deposit column |
| `adjusted_scores` | Full packed scores after subtracting prices |
| `rankings` | Complete rankings in packed form |
| `ranking_offsets` | Row boundaries for `rankings` |
| `metadata` | JSON configuration and convergence diagnostics |

NumPy archives are read with `allow_pickle=False`. Keep both inputs and outputs
outside the source repository. Use `--full-support` to estimate prices on every
candidate edge; the hard assignment always uses the full graph regardless of
this option.

## Scale and memory behavior

The exact assignment stage allocates
`num_queries * max(counts)` float64 cells. The default guard rejects problems
above 100 million cells, approximately 0.75 GiB for the main cost matrix. Set
`max_dense_elements` to a larger value or `None` only after confirming that the
machine has sufficient memory.

With the default sparse support, price estimation uses at most approximately
`num_queries * (top_l + 1)` edges. Fixed-first ranking still processes and
returns every input candidate.

## What this release reproduces

This package reproduces the NoteFlow inference algorithm given frozen candidate
scores. It does not train upstream scorers or reproduce the paper's numerical
tables without the separately managed data, score arrays, and model artifacts.
The code makes non-convergence and sparse-support approximation visible instead
of silently treating prices as an exact full-graph solution.

## Testing and release audit

```bash
pytest -q
python scripts/check_release.py
python examples/quickstart.py
python -m build
```

The tests cover exact assignment against brute-force optima, analytically
checkable soft prices, sparse-support permutation equivariance, fixed-first
ranking completeness, non-convergence reporting, input validation, and a
pickle-free CLI round trip.

The release audit rejects common data, checkpoint, cache, and result paths and
file formats. The source-distribution manifest applies the same exclusions.

## Scope and responsible use

- Candidate sets must be prefixes of one stable deposit ordering and must admit
  a row-covering injective assignment.
- The implementation performs joint inference over an offline observation
  window. It is not a strictly online predictor.
- The single-use model does not directly cover splitting, many-to-many links,
  missing ground truth, or true matches outside the candidate set.
- A predicted link is a ranking hypothesis, not evidence of identity, intent,
  or illegal activity.
- Results obtained on one dataset or simulator should not be presented as
  validated performance on unrelated real-world mixer activity.

## License

NoteFlow is released under the MIT License. See [LICENSE](LICENSE). Replace the
copyright metadata before publication if your institution requires different
attribution or licensing terms.

