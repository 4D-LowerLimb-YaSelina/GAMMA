import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "baselines"
    / "faust_raw"
    / "batch_decimate.py"
)
SPEC = importlib.util.spec_from_file_location("batch_decimate", MODULE_PATH)
batch_decimate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch_decimate)


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.64, "0.64 s"),
        (59.996, "1 min 0.00 s"),
        (65.43, "1 min 5.43 s"),
        (3665.43, "1 h 1 min 5.43 s"),
    ],
)
def test_format_duration(seconds, expected):
    assert batch_decimate._format_duration(seconds) == expected


def test_format_progress_includes_completed_percentage_and_precise_eta():
    assert batch_decimate._format_progress(
        completed=10,
        total=100,
        elapsed=5.9,
    ) == (
        "Progress: 10/100 (10.0%) | Elapsed: 5.90 s | "
        "ETA: 53.10 s | Average: 0.59 s/mesh"
    )
