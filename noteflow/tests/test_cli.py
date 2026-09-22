import json

import numpy as np

from noteflow.cli import main


def test_cli_round_trip_without_pickle(tmp_path):
    input_path = tmp_path / "input.npz"
    output_path = tmp_path / "output.npz"
    np.savez(
        input_path,
        scores=np.array([1.0, 0.2, 0.9, 0.8]),
        counts=np.array([2, 2]),
        row_ids=np.array(["w0", "w1"]),
    )
    assert main(
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--full-support",
            "--max-iter",
            "100",
        ]
    ) == 0
    with np.load(output_path, allow_pickle=False) as archive:
        assert set(archive.files) == {
            "assignment",
            "prices",
            "column_mass",
            "adjusted_scores",
            "rankings",
            "ranking_offsets",
            "metadata",
        }
        metadata = json.loads(str(archive["metadata"].item()))
        assert metadata["queries"] == 2

