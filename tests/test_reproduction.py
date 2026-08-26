from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "reproduce_computation_results.py"
SPEC = importlib.util.spec_from_file_location("reproduce_computation_results", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_2r2c_analytical_matches_response_curve() -> None:
    params = MODULE.generate_2r2c_params(3)
    duration_s = 2 * 3600.0

    analytical = MODULE.benchmark_analytical_2r2c(params, 2.0, duration_s, duration_s)
    proposed, _ = MODULE.benchmark_proposed_unsaved(
        "2R2C", params, MODULE.rhs_2r2c, [0.0, 0.0], 2.0, duration_s, duration_s, 60.0
    )

    assert np.isclose(analytical.reduction_w, proposed.reduction_w, rtol=2e-6)
    assert np.isclose(analytical.rebound_above_baseline_w, proposed.rebound_above_baseline_w, rtol=5e-6)


def test_saved_curve_matches_unsaved_curve(tmp_path: Path) -> None:
    params = MODULE.generate_5r4c_params(3)
    duration_s = 2 * 3600.0
    total_s = 2 * duration_s
    unsaved, curves = MODULE.benchmark_proposed_unsaved(
        "5R4C", params, MODULE.rhs_5r4c, [0.0, 0.0, 0.0, 0.0], 2.0, duration_s, duration_s, 60.0
    )
    cache = MODULE.ensure_cache(
        tmp_path, "5R4C", params, MODULE.rhs_5r4c, [0.0, 0.0, 0.0, 0.0], 60.0, total_s, True, curves
    )
    saved = MODULE.benchmark_proposed_saved("5R4C", cache, 2.0, duration_s, duration_s, 60.0, 2)

    assert np.isclose(unsaved.reduction_w, saved.reduction_w)
    assert np.isclose(unsaved.rebound_above_baseline_w, saved.rebound_above_baseline_w)


def test_published_power_values_match_full_precision_reference() -> None:
    results_dir = Path(__file__).parents[1] / "results"
    with (results_dir / "reference_run_full_precision.csv").open(newline="") as file:
        reference_rows = list(csv.DictReader(file))
    with (results_dir / "published_manuscript_table.csv").open(newline="") as file:
        published_rows = list(csv.DictReader(file))

    reference = {(row["model"], row["method"]): row for row in reference_rows}
    for row in published_rows:
        if row["method"] == "Analytical" and row["model"] == "5R4C":
            continue
        source = reference[(row["model"], row["method"])]
        assert f"{-float(source['load_reduction_w']) / 1e6:.4f}" == row["delta_p1_mw"]
        assert f"{float(source['rebound_above_baseline_w']) / 1e6:.4f}" == row["delta_p2_mw"]
