# NoteFlow

NoteFlow is the standalone implementation of the method described in
*NoteFlow: Ranking Deposit--Withdrawal Links in Ethereum Mixers*. It turns
fixed candidate-pair scores from any upstream scorer into:

1. a globally optimal rank-one assignment that respects single-use deposit
   capacity;
2. nonnegative deposit prices that summarize competition across withdrawals;
3. a complete candidate ranking for every withdrawal, with the feasible
   assignment fixed at rank one.


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


