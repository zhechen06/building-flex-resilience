# Fast building flexibility and resilience evaluation

Reproducibility code for the manuscript **“Fast flexibility and resilience evaluation for building air-conditioning systems”** by Zhe Chen, Yongbao Chen, and Fu Xiao.

The code reproduces the manuscript calculations for clusters of 4,000 buildings represented by 2R2C and 5R4C thermal models. It compares direct numerical simulation, the 2R2C analytical solution, and the proposed response-curve method.

No external dataset is required. Building parameters are generated deterministically from the ranges and random seeds defined in the script.

## Files

- `scripts/reproduce_computation_results.py`: main calculation script.
- `results/`: published and full-precision reference results.
- `tests/`: consistency tests for the analytical and response-curve methods.
- `pyproject.toml` and `uv.lock`: Python environment and locked dependencies.

## Usage

Install [uv](https://docs.astral.sh/uv/), then run from the repository root:

```bash
uv sync --dev --locked
uv run pytest
```

Quick test:

```bash
uv run scripts/reproduce_computation_results.py \
  --buildings 20 \
  --repeat-saved 3 \
  --csv-out results/local_smoke_test.csv
```

Full manuscript calculation:

```bash
uv run scripts/reproduce_computation_results.py \
  --buildings 4000 \
  --repeat-saved 1000 \
  --csv-out results/local_full_run.csv
```

The full numerical benchmark is slow. Add `--skip-numerical` to run only the analytical and response-curve methods. Runtime values vary across computers; power results are deterministic.

## Notes

The manuscript reports load reduction as a negative deviation from baseline (`Delta P1 < 0`) and rebound power as a positive deviation (`Delta P2 > 0`). The script stores the load-reduction magnitude as a positive value; `results/published_manuscript_table.csv` applies the manuscript convention.

Please cite the associated manuscript when using this code. Bibliographic details and a DOI will be added after publication.

No open-source license has been assigned yet. All rights are reserved by the authors.
