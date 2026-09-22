"""Minimal synthetic example; no research data is bundled."""

import numpy as np

from noteflow import NoteFlowConfig, run_noteflow


# Three queries, each connected to a prefix of the same deposit universe.
counts = np.array([3, 4, 4])
rows = [
    np.array([2.0, 1.9, 0.1]),
    np.array([2.1, 1.2, 0.4, 0.0]),
    np.array([1.8, 1.7, 1.6, 0.3]),
]
scores = np.concatenate(rows)
row_ids = np.array(["withdrawal-a", "withdrawal-b", "withdrawal-c"])

result = run_noteflow(
    scores,
    counts,
    row_ids=row_ids,
    config=NoteFlowConfig(top_l=3, max_iter=1000, tolerance=1e-6),
)

print("feasible rank-one assignment:", result.assignment.columns.tolist())
for index in range(len(counts)):
    print(f"query {index} ranking:", result.ranking_for(index).tolist())
print("diagnostics:", result.diagnostics())
