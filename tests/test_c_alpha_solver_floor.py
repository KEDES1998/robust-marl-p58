"""Smoke test for experiments/c_alpha_solver_floor.py on a tiny config."""

import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "c_alpha_solver_floor.py"


@pytest.fixture(scope="module")
def floor_script():
    spec = importlib.util.spec_from_file_location("c_alpha_solver_floor", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TINY = {
    "name": "tiny", "family": "F1", "n": [2], "A": 2, "S": 2, "H": 2, "sigma": 0.3, "epsilon": 0.1,
    "n_instances": 2, "c_alpha": [1, 24], "K": [10, 40],
}


def test_jobs_cover_the_grid(floor_script):
    jobs = floor_script.build_jobs(TINY)
    assert len(jobs) == 2 * 2 * 2                                    # instances x c_alpha x K
    assert jobs[0]["K"] == 40                                        # most expensive first
    assert (jobs[0]["beta"], jobs[0]["w"]) == (1.0, 0.5)             # defaults of random_game


def test_larger_c_alpha_keeps_fewer_plans(floor_script):
    w1, w24 = floor_script.weight_table([1, 24], [2000], H=3)
    assert w24["effective_plans"] < w1["effective_plans"]
    assert w24["weight_first_half"] < w1["weight_first_half"]


def test_main_writes_one_row_per_run(floor_script, tmp_path):
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(TINY))
    floor_script.main([str(config), "--out", str(tmp_path / "out"), "--plot"])

    with open(tmp_path / "out" / "runs.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8
    assert set(rows[0]) == set(floor_script.FIELDS)
    assert all(float(r["gap_out"]) >= 0.0 for r in rows)
    for name in ("weights.csv", "tiny.yaml", "run_info.yaml", "floor.png"):
        assert (tmp_path / "out" / name).exists()
