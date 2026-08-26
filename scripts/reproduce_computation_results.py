"""
Reproduce the manuscript table for flexibility and rebound evaluation.

This script uses the parameter ranges, deterministic random seeds, number of
buildings, COP, and 2-hour flexibility event reported in the manuscript. It
then computes the rebound power required to restore the baseline temperature
after another 2-hour rebound period.

Sign convention in this script:
    R = |Delta P1| > 0 is the load-reduction magnitude.
    K = Delta P2 - Delta P1 > 0 is the power increase from the
        reduced level during rebound.
    B = K - R = Delta P2 is the rebound power above the baseline.

Default scenario:
    1. Find R_i so each building reaches Delta T_in = 2 degC after 2 h.
    2. Apply rebound for 2 h and solve for K_i so Delta T_in returns to 0.
    3. Report sum(B_i), where B_i = K_i - R_i = Delta P2_i.

Run:
    uv run scripts/reproduce_computation_results.py

Quick smoke test:
    uv run scripts/reproduce_computation_results.py --buildings 20 --repeat-saved 3
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.integrate import odeint


COP = 3.0


@dataclass(frozen=True)
class Result:
    model: str
    method: str
    buildings: int
    reduction_w: float | None
    power_increase_from_reduced_w: float | None
    rebound_above_baseline_w: float | None
    time_s: float | None


def generate_2r2c_params(n: int) -> list[tuple[float, float, float, float]]:
    c_in_range = (0.13 * 1000 * 3600, 0.22 * 1000 * 3600)
    c_m_range = (0.75 * 1000 * 3600, 1.25 * 1000 * 3600)
    r_out_range = (3.15 / 1000, 5.25 / 1000)
    r_m_range = (1.88 / 1000, 3.13 / 1000)

    params = []
    for i in range(n):
        rng = np.random.RandomState(i)
        params.append(
            (
                float(rng.uniform(*c_in_range)),
                float(rng.uniform(*c_m_range)),
                float(rng.uniform(*r_out_range)),
                float(rng.uniform(*r_m_range)),
            )
        )
    return params


def generate_5r4c_params(n: int) -> list[tuple[float, float, float, float, float, float, float, float]]:
    c_w_range = (1500000, 3000000)
    c_in_range = (150000, 300000)
    c_m_range = (25000000, 40000000)
    r_win_range = (0.05, 0.08)
    r_w_o_range = (0.002, 0.004)
    r_w_range = (0.04, 0.07)
    r_w_in_range = (0.004, 0.009)
    r_in_m_range = (0.0008, 0.0012)

    params = []
    for i in range(n):
        rng = np.random.RandomState(i)
        params.append(
            (
                float(rng.uniform(*c_w_range)),
                float(rng.uniform(*c_in_range)),
                float(rng.uniform(*c_m_range)),
                float(rng.uniform(*r_win_range)),
                float(rng.uniform(*r_w_o_range)),
                float(rng.uniform(*r_w_range)),
                float(rng.uniform(*r_w_in_range)),
                float(rng.uniform(*r_in_m_range)),
            )
        )
    return params


def rhs_2r2c(y: Iterable[float], _t: float, u: float, p: tuple[float, float, float, float]) -> list[float]:
    c_in, c_m, r_out, r_m = p
    t_in, t_m = y
    d_t_in = ((0.0 - t_in) / r_out + (t_m - t_in) / r_m + COP * u) / c_in
    d_t_m = ((t_in - t_m) / r_m) / c_m
    return [d_t_in, d_t_m]


def rhs_5r4c(
    y: Iterable[float],
    _t: float,
    u: float,
    p: tuple[float, float, float, float, float, float, float, float],
) -> list[float]:
    c_w, c_in, c_m, r_win, r_w_o, r_w, r_w_in, r_in_m = p
    t_in, t_m, t_w_int, t_w_ext = y

    d_t_w_ext = ((0.0 - t_w_ext) / r_w_o + (t_w_int - t_w_ext) / r_w) / c_w
    d_t_w_int = ((t_w_ext - t_w_int) / r_w + (t_in - t_w_int) / r_w_in) / c_w
    d_t_in = ((t_m - t_in) / r_in_m + (t_w_int - t_in) / r_w_in + (0.0 - t_in) / r_win + COP * u) / c_in
    d_t_m = ((t_in - t_m) / r_in_m) / c_m

    return [d_t_in, d_t_m, d_t_w_int, d_t_w_ext]


def simulate_constant(
    rhs: Callable,
    y0: list[float],
    params: tuple,
    u: float,
    duration_s: float,
    dt_s: float,
) -> np.ndarray:
    steps = int(round(duration_s / dt_s))
    t = np.linspace(0.0, duration_s, steps + 1)
    return odeint(rhs, y0, t, args=(u, params))


def unit_step_curve(
    rhs: Callable,
    y0: list[float],
    params: tuple,
    duration_s: float,
    dt_s: float,
) -> np.ndarray:
    sol = simulate_constant(rhs, y0, params, 1.0, duration_s, dt_s)
    return sol[:, 0]


def find_reduction_numerical(
    rhs: Callable,
    y0: list[float],
    params: tuple,
    target_rise: float,
    reduction_duration_s: float,
    dt_s: float,
) -> float:
    lo = 0.0
    hi = 1000.0
    while simulate_constant(rhs, y0, params, hi, reduction_duration_s, dt_s)[-1, 0] < target_rise:
        hi *= 2.0

    for _ in range(60):
        mid = (lo + hi) / 2.0
        t_final = simulate_constant(rhs, y0, params, mid, reduction_duration_s, dt_s)[-1, 0]
        if t_final < target_rise:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def final_temp_after_rebound(
    rhs: Callable,
    y0: list[float],
    params: tuple,
    reduction_w: float,
    power_increase_w: float,
    reduction_duration_s: float,
    rebound_duration_s: float,
    dt_s: float,
) -> float:
    stage1 = simulate_constant(rhs, y0, params, reduction_w, reduction_duration_s, dt_s)
    # u is positive for reduction. During rebound, the deviation from baseline is
    # reduction_w - power_increase_w.
    stage2_u = reduction_w - power_increase_w
    stage2 = simulate_constant(rhs, stage1[-1].tolist(), params, stage2_u, rebound_duration_s, dt_s)
    return float(stage2[-1, 0])


def find_power_increase_numerical(
    rhs: Callable,
    y0: list[float],
    params: tuple,
    reduction_w: float,
    reduction_duration_s: float,
    rebound_duration_s: float,
    dt_s: float,
) -> float:
    lo = 0.0
    hi = max(2.0 * reduction_w, 1.0)
    while final_temp_after_rebound(rhs, y0, params, reduction_w, hi, reduction_duration_s, rebound_duration_s, dt_s) > 0.0:
        hi *= 2.0

    for _ in range(60):
        mid = (lo + hi) / 2.0
        t_final = final_temp_after_rebound(rhs, y0, params, reduction_w, mid, reduction_duration_s, rebound_duration_s, dt_s)
        if t_final > 0.0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def step_response_2r2c_analytical(t_s: float, p: tuple[float, float, float, float]) -> float:
    c_in, c_m, r_out, r_m = p
    a = c_in * c_m * r_m
    b = (1.0 + r_m / r_out) * c_m + c_in
    c = 1.0 / r_out
    sqrt_value = math.sqrt(b * b - 4.0 * a * c)
    r1 = (-b + sqrt_value) / (2.0 * a)
    r2 = (-b - sqrt_value) / (2.0 * a)
    numerator = (
        math.exp(r1 * t_s)
        - math.exp(r2 * t_s)
        + c_in * r_out * (r1 - r2 + r2 * math.exp(r1 * t_s) - r1 * math.exp(r2 * t_s))
    ) * COP
    denominator = (r1 - r2) * c_in
    return numerator / denominator


def rebound_from_step_values(target_rise: float, s_reduction: float, s_rebound: float, s_total: float) -> tuple[float, float, float]:
    reduction_w = target_rise / s_reduction
    power_increase_w = reduction_w * s_total / s_rebound
    rebound_above_baseline_w = power_increase_w - reduction_w
    return reduction_w, power_increase_w, rebound_above_baseline_w


def benchmark_numerical(
    model: str,
    params_list: list[tuple],
    rhs: Callable,
    y0: list[float],
    target_rise: float,
    reduction_duration_s: float,
    rebound_duration_s: float,
    dt_s: float,
) -> Result:
    start = time.perf_counter()
    reductions = []
    increases = []
    rebounds = []
    for params in params_list:
        reduction = find_reduction_numerical(rhs, y0, params, target_rise, reduction_duration_s, dt_s)
        increase = find_power_increase_numerical(rhs, y0, params, reduction, reduction_duration_s, rebound_duration_s, dt_s)
        reductions.append(reduction)
        increases.append(increase)
        rebounds.append(increase - reduction)
    elapsed = time.perf_counter() - start
    return Result(model, "Numerical", len(params_list), sum(reductions), sum(increases), sum(rebounds), elapsed)


def benchmark_analytical_2r2c(
    params_list: list[tuple[float, float, float, float]],
    target_rise: float,
    reduction_duration_s: float,
    rebound_duration_s: float,
) -> Result:
    start = time.perf_counter()
    reductions = []
    increases = []
    rebounds = []
    total_s = reduction_duration_s + rebound_duration_s
    for params in params_list:
        s_red = step_response_2r2c_analytical(reduction_duration_s, params)
        s_reb = step_response_2r2c_analytical(rebound_duration_s, params)
        s_total = step_response_2r2c_analytical(total_s, params)
        reduction, increase, rebound = rebound_from_step_values(target_rise, s_red, s_reb, s_total)
        reductions.append(reduction)
        increases.append(increase)
        rebounds.append(rebound)
    elapsed = time.perf_counter() - start
    return Result("2R2C", "Analytical", len(params_list), sum(reductions), sum(increases), sum(rebounds), elapsed)


def benchmark_proposed_unsaved(
    model: str,
    params_list: list[tuple],
    rhs: Callable,
    y0: list[float],
    target_rise: float,
    reduction_duration_s: float,
    rebound_duration_s: float,
    dt_s: float,
) -> tuple[Result, np.ndarray]:
    start = time.perf_counter()
    idx_red = int(round(reduction_duration_s / dt_s))
    idx_reb = int(round(rebound_duration_s / dt_s))
    total_s = reduction_duration_s + rebound_duration_s
    reductions = []
    increases = []
    rebounds = []
    curves = np.empty((len(params_list), int(round(total_s / dt_s)) + 1), dtype=float)
    for i, params in enumerate(params_list):
        curve = unit_step_curve(rhs, y0, params, total_s, dt_s)
        curves[i, :] = curve
        reduction, increase, rebound = rebound_from_step_values(target_rise, curve[idx_red], curve[idx_reb], curve[-1])
        reductions.append(reduction)
        increases.append(increase)
        rebounds.append(rebound)
    elapsed = time.perf_counter() - start
    result = Result(model, "Proposed (curves unsaved)", len(params_list), sum(reductions), sum(increases), sum(rebounds), elapsed)
    return result, curves


def cache_path(cache_dir: Path, model: str, n: int, dt_s: float, total_s: float) -> Path:
    dt_tag = int(round(dt_s))
    total_tag = int(round(total_s))
    return cache_dir / f"{model.lower()}_unit_step_n{n}_dt{dt_tag}_total{total_tag}_cop3.npy"


def ensure_cache(
    cache_dir: Path,
    model: str,
    params_list: list[tuple],
    rhs: Callable,
    y0: list[float],
    dt_s: float,
    total_s: float,
    force: bool,
    existing_curves: np.ndarray | None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_path(cache_dir, model, len(params_list), dt_s, total_s)
    if force or not path.exists():
        if existing_curves is None:
            curves = np.empty((len(params_list), int(round(total_s / dt_s)) + 1), dtype=float)
            for i, params in enumerate(params_list):
                curves[i, :] = unit_step_curve(rhs, y0, params, total_s, dt_s)
        else:
            curves = existing_curves
        np.save(path, curves)
    return path


def benchmark_proposed_saved(
    model: str,
    path: Path,
    target_rise: float,
    reduction_duration_s: float,
    rebound_duration_s: float,
    dt_s: float,
    repeat_saved: int,
) -> Result:
    idx_red = int(round(reduction_duration_s / dt_s))
    idx_reb = int(round(rebound_duration_s / dt_s))
    total_time = 0.0
    final_values = None
    for _ in range(max(repeat_saved, 1)):
        start = time.perf_counter()
        curves = np.load(path)
        s_red = curves[:, idx_red]
        s_reb = curves[:, idx_reb]
        s_total = curves[:, -1]
        reductions = target_rise / s_red
        increases = reductions * s_total / s_reb
        rebounds = increases - reductions
        final_values = (float(reductions.sum()), float(increases.sum()), float(rebounds.sum()), curves.shape[0])
        total_time += time.perf_counter() - start

    reduction_sum, increase_sum, rebound_sum, n = final_values
    return Result(
        model,
        "Proposed (curves saved)",
        int(n),
        reduction_sum,
        increase_sum,
        rebound_sum,
        total_time / max(repeat_saved, 1),
    )


def fmt_num(value: float | None, digits: int = 0) -> str:
    if value is None:
        return "--"
    if digits == 0:
        return f"{value:.0f}"
    return f"{value:.{digits}f}"


def print_results(results: list[Result]) -> None:
    print("\nDetailed results")
    print(
        "Model, Method, Buildings, Load reduction |R| (W), "
        "Power increase from reduced level K (W), Rebound above baseline B=K-R (W), Time (s)"
    )
    for r in results:
        print(
            f"{r.model}, {r.method}, {r.buildings}, "
            f"{fmt_num(r.reduction_w)}, {fmt_num(r.power_increase_from_reduced_w)}, "
            f"{fmt_num(r.rebound_above_baseline_w)}, {fmt_num(r.time_s, 6)}"
        )

    by_method = {r.method: r for r in results if r.model == "2R2C"}
    by_method_5 = {r.method: r for r in results if r.model == "5R4C"}
    methods = [
        "Numerical",
        "Analytical",
        "Proposed (curves unsaved)",
        "Proposed (curves saved)",
    ]

    print("\nSuggested manuscript table rows (metric: rebound power above baseline)")
    print(r"Method & \multicolumn{2}{c}{2R2C} & \multicolumn{2}{c}{5R4C} \\")
    print(r" & Rebound power (W) & Time (s) & Rebound power (W) & Time (s) \\")
    for method in methods:
        r2 = by_method.get(method)
        r5 = by_method_5.get(method)
        print(
            f"{method} & "
            f"{fmt_num(r2.rebound_above_baseline_w if r2 else None)} & {fmt_num(r2.time_s if r2 else None, 4)} & "
            f"{fmt_num(r5.rebound_above_baseline_w if r5 else None)} & {fmt_num(r5.time_s if r5 else None, 4)} \\\\"
        )


def write_csv(path: Path, results: list[Result]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "method",
                "buildings",
                "load_reduction_w",
                "power_increase_from_reduced_level_w",
                "rebound_above_baseline_w",
                "time_s",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.model,
                    r.method,
                    r.buildings,
                    r.reduction_w,
                    r.power_increase_from_reduced_w,
                    r.rebound_above_baseline_w,
                    r.time_s,
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the manuscript flexibility/rebound computation table.")
    parser.add_argument("--buildings", type=int, default=4000, help="Number of buildings per model.")
    parser.add_argument("--dt", type=float, default=60.0, help="Simulation step in seconds.")
    parser.add_argument("--duration-hours", type=float, default=2.0, help="Load-reduction duration in hours.")
    parser.add_argument("--rebound-hours", type=float, default=2.0, help="Rebound duration in hours.")
    parser.add_argument("--target-rise", type=float, default=2.0, help="Temperature rise at the end of reduction, degC.")
    parser.add_argument("--repeat-saved", type=int, default=1000, help="Repetitions used to average saved-curve timing.")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).with_name("table2_rebound_cache"),
        help="Directory for generated unit-step response curves.",
    )
    parser.add_argument("--force-cache", action="store_true", help="Regenerate saved response-curve cache.")
    parser.add_argument("--skip-numerical", action="store_true", help="Skip slow numerical bisection methods.")
    parser.add_argument("--csv-out", type=Path, default=None, help="Optional CSV output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reduction_duration_s = args.duration_hours * 3600.0
    rebound_duration_s = args.rebound_hours * 3600.0
    total_s = reduction_duration_s + rebound_duration_s

    params_2 = generate_2r2c_params(args.buildings)
    params_5 = generate_5r4c_params(args.buildings)

    results: list[Result] = []
    curves_for_cache: dict[str, np.ndarray] = {}

    if not args.skip_numerical:
        results.append(
            benchmark_numerical(
                "2R2C",
                params_2,
                rhs_2r2c,
                [0.0, 0.0],
                args.target_rise,
                reduction_duration_s,
                rebound_duration_s,
                args.dt,
            )
        )
        results.append(
            benchmark_numerical(
                "5R4C",
                params_5,
                rhs_5r4c,
                [0.0, 0.0, 0.0, 0.0],
                args.target_rise,
                reduction_duration_s,
                rebound_duration_s,
                args.dt,
            )
        )

    results.append(benchmark_analytical_2r2c(params_2, args.target_rise, reduction_duration_s, rebound_duration_s))
    results.append(Result("5R4C", "Analytical", args.buildings, None, None, None, None))

    result_2_unsaved, curves_2 = benchmark_proposed_unsaved(
        "2R2C",
        params_2,
        rhs_2r2c,
        [0.0, 0.0],
        args.target_rise,
        reduction_duration_s,
        rebound_duration_s,
        args.dt,
    )
    results.append(result_2_unsaved)
    curves_for_cache["2R2C"] = curves_2

    result_5_unsaved, curves_5 = benchmark_proposed_unsaved(
        "5R4C",
        params_5,
        rhs_5r4c,
        [0.0, 0.0, 0.0, 0.0],
        args.target_rise,
        reduction_duration_s,
        rebound_duration_s,
        args.dt,
    )
    results.append(result_5_unsaved)
    curves_for_cache["5R4C"] = curves_5

    path_2 = ensure_cache(
        args.cache_dir,
        "2R2C",
        params_2,
        rhs_2r2c,
        [0.0, 0.0],
        args.dt,
        total_s,
        args.force_cache,
        curves_for_cache["2R2C"],
    )
    path_5 = ensure_cache(
        args.cache_dir,
        "5R4C",
        params_5,
        rhs_5r4c,
        [0.0, 0.0, 0.0, 0.0],
        args.dt,
        total_s,
        args.force_cache,
        curves_for_cache["5R4C"],
    )

    results.append(
        benchmark_proposed_saved(
            "2R2C",
            path_2,
            args.target_rise,
            reduction_duration_s,
            rebound_duration_s,
            args.dt,
            args.repeat_saved,
        )
    )
    results.append(
        benchmark_proposed_saved(
            "5R4C",
            path_5,
            args.target_rise,
            reduction_duration_s,
            rebound_duration_s,
            args.dt,
            args.repeat_saved,
        )
    )

    print("\nScenario")
    print(f"Buildings per model: {args.buildings}")
    print(f"COP: {COP:g}")
    print(f"Reduction duration: {args.duration_hours:g} h")
    print(f"Rebound duration: {args.rebound_hours:g} h")
    print(f"Target temperature rise at end of reduction: {args.target_rise:g} degC")
    print("Reported rebound power is the amount above the baseline, B = K - R = Delta P2.")
    print(f"Saved-curve cache directory: {args.cache_dir}")
    print("Cache generation time is excluded from the saved-curve method timing.")

    print_results(results)

    if args.csv_out is not None:
        write_csv(args.csv_out, results)
        print(f"\nWrote CSV: {args.csv_out}")


if __name__ == "__main__":
    main()
