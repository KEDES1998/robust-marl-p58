"""Smoke test for experiments/run_scaling.py on a tiny config."""

import csv
import importlib.util
from pathlib import Path

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "run_scaling.py"


@pytest.fixture(scope="module")
def run_scaling():
    spec = importlib.util.spec_from_file_location("run_scaling", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TINY = {
    "name": "tiny", "family": "F1", "n": [2, 3], "A": 2, "S": 2, "H": 2, "sigma": 0.3,
    "epsilon": 0.1, "n_instances": 2, "n_sampling_seeds": 2, "success_rate": 0.9, "c_alpha": 1.0,
    "methods": {"exact": {"K": 20}, "plugin": {"m": [1, 4], "K_solver": 20},
                "robust_q_ftrl": {"N": [1], "K": [8]}},
}


def test_jobs_cover_the_grid(run_scaling):
    jobs = run_scaling.build_jobs(TINY, ["exact", "plugin"])
    # per n: 2 instances x (1 exact + 2 seeds x 2 values of m)
    assert len(jobs) == 2 * 2 * (1 + 2 * 2)
    assert jobs[0]["n"] == 3                                    # most expensive first


def test_main_writes_rows_with_the_right_counters(run_scaling, tmp_path):
    config = tmp_path / "tiny.yaml"
    config.write_text(yaml.safe_dump(TINY))
    run_scaling.main([str(config), "--out", str(tmp_path / "out")])

    with open(tmp_path / "out" / "runs.csv") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    assert set(rows[0]) == set(run_scaling.FIELDS)
    for r in rows:
        n, m = int(r["n"]), r["m"]
        expected = 0 if r["method"] == "exact" else TINY["H"] * TINY["S"] * 2**n * int(m)
        assert int(r["n_samples"]) == expected
        assert float(r["gap_out"]) >= 0.0
        assert float(r["c_alpha"]) == 1.0
    assert (tmp_path / "out" / "tiny.yaml").exists()
    assert (tmp_path / "out" / "run_info.yaml").exists()
