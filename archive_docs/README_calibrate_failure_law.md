## Calibrate Failure Law: Datasets, Metrics, Plots, and CI

This utility estimates the failure-law calibration parameters (lambda, tau) that map model uncertainty `hbar_s` to a probability of failure `P(fail)` via a logistic transform.

It now supports:
- TruthfulQA and HaluEval dataset ingestion (optional, with graceful fallbacks)
- Train/validation/test splits to avoid leakage
- Advanced metrics (ROC-AUC, Brier Score, ECE, Welch's t-test)
- Optional ROC and reliability (calibration) plots
- Concurrency and rate limiting for throughput
- Structured JSON/CSV outputs for reporting and CI

### Installation

Core deps are standard library + `requests` (already used by the repo). Optional features require extra packages:

```bash
pip install scikit-learn scipy matplotlib datasets tqdm
```

If you do not install these, the script still runs with reduced functionality and will skip metrics/plots that are unavailable.

### CLI

```bash
python scripts/calibrate_failure_law.py \
  --base http://127.0.0.1:3000/api/v1 \
  [--pairs path/to/pairs.json | --dataset default|truthfulqa|halueval] \
  [--halueval_task qa|dialogue|summarization|general] \
  [--halueval_url https://raw.githubusercontent.com/.../subset.json] \
  [--max_samples 1000] \
  [--seed 42] \
  [--val_split 0.2 --test_split 0.2] \
  [--concurrency 8 --rate_limit 0] \
  [--no_plots --plot_dir ./plots] \
  [--output_json ./calibration_summary.json --save_csv ./scores.csv]
```

Key arguments:
- `--pairs`: Provide your own JSON of objects with fields `prompt`, `failing`, `passing`.
- `--dataset`: Built-in sources: `default` (toy), `truthfulqa`, `halueval`.
- `--halueval_task`: Task subset for HaluEval. Defaults to `qa`.
- `--halueval_url`: Override the default HaluEval raw JSON URL if mirrors move.
- `--max_samples`: Randomly sample up to N pairs for quick runs.
- `--seed`: Seed for reproducible shuffles and splits.
- `--val_split`, `--test_split`: Fractions on the pair level. Calibration uses train; metrics reported for train/val/test.
- `--concurrency`: Number of concurrent requests to `--base` `/analyze` endpoint.
- `--rate_limit`: Per-thread requests-per-second throttle. Set to 0 to disable.
- `--no_plots`: Disable all plotting (useful in CI).
- `--plot_dir`: Directory to save `roc.png` and `calibration.png` when plotting is enabled.
- `--output_json`: Save a summary JSON (lambda, tau, losses, split sizes, metrics).
- `--save_csv`: Save per-sample rows with `split,hbar_s,pfail,label` (prompts are not stored to avoid leakage).

### Datasets

- TruthfulQA (generation): Loaded via `datasets` library (`truthful_qa`, config `generation`). We normalize fields into `(prompt, passing, failing)`. If the library is not installed, the script skips this dataset.
- HaluEval: Fetched from public raw JSON files on GitHub. If URLs change, pass `--halueval_url`. We normalize common fields like `question/prompt/src`, `correct/answer/reference`, `incorrect/negative/hallucination`.

Notes:
- These sources evolve; validate schema before large runs.
- Respect licenses and attribution for public datasets.

### Metrics

Aggregated globally per split:
- ROC-AUC: Discrimination between failing (1) and passing (0)
- Brier Score: Mean squared error of predicted `P(fail)`
- ECE and MCE: Calibration errors with uniform binning (default 15 bins)
- Welch's t-test: Difference of mean probabilities between failing vs passing groups

If `scikit-learn`/`scipy` are not available, only ECE/MCE are computed via native logic.

### Plots

With `--plot_dir` set and `--no_plots` omitted, the script writes:
- `roc.png`: ROC curve (with AUC in the legend if available)
- `calibration.png`: Reliability diagram (predicted vs observed)

The script forces a headless backend (`Agg`) to keep CI stable.

### Outputs

- JSON summary (if `--output_json`):
  - `lambda`, `tau`, `train_loss`
  - `num_pairs` per split
  - `metrics` per split
- CSV (if `--save_csv`): rows of `split,hbar_s,pfail,label`

### Example runs

Toy default dataset:

```bash
python scripts/calibrate_failure_law.py --dataset default --output_json out.json --plot_dir plots
```

TruthfulQA, 2k pairs, 8 workers:

```bash
python scripts/calibrate_failure_law.py \
  --dataset truthfulqa --max_samples 2000 \
  --concurrency 8 --plot_dir plots --output_json tqa_summary.json --save_csv tqa_scores.csv
```

HaluEval QA subset with explicit mirror URL:

```bash
python scripts/calibrate_failure_law.py \
  --dataset halueval --halueval_task qa \
  --halueval_url https://raw.githubusercontent.com/HaluEval/HaluEval/main/data/qa.json \
  --max_samples 500 --plot_dir plots --output_json he_summary.json
```

### Agentic coding practices applied

- Modularity: distinct functions for loading datasets, calibration, metrics, plotting
- Graceful degradation: optional dependencies are lazily imported and gated
- Reproducibility: CLI seed, explicit splits, CSV/JSON outputs
- Resilience: sanitization of inputs, safe float conversions, headless plotting
- Throughput: concurrency and optional rate limit to match backend capacity

### Integration tips

- Point `--base` to your running API with a compatible `/analyze` endpoint.
- For CI, prefer `--no_plots` plus `--output_json` and `--save_csv`.
- For very large datasets, start with a small `--max_samples` to validate setup, then scale up.


