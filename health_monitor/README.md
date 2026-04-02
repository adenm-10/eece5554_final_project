# Stereo Health Monitor

End-to-end pipeline for training a model that predicts stereo camera degradation severity from image sequences.

## Architecture

```
Per frame:
  left_img, right_img → ResNet18 (shared) → left_emb, right_emb (512 each)
  diff_emb = left_emb - right_emb
  frame_emb = concat(left_emb, right_emb, diff_emb) → Linear(1536→128)

Sequence:
  20 frame embeddings → GRU(128, 64) → last hidden (64,)

Classifier:
  Linear(64→32) + ReLU + Dropout → Linear(32→1) + Sigmoid → severity ∈ [0, 1]
```

Severity is a continuous score derived from per-window RTE:
- `severity = rte_stereo / (rte_stereo + rte_mono_baseline)` (default: ratio policy)
- Crossover at 0.5 — below means stereo is performing better than mono, above means worse
- Degradations are applied on-the-fly at load time; no duplicate images are stored

## File Overview

| File | Purpose |
|---|---|
| `config.py` | Shared constants (paths, window size, image size, thresholds) |
| `build_dataset_index.py` | Computes per-window severity scores from SLAM results; writes `dataset.csv` + `config.json` |
| `stereo_health_dataset.py` | PyTorch Dataset; discovers all trajs/conditions, applies degradation on-the-fly |
| `train.py` | Model definition (FrameEncoder + StereoHealthMonitor) and training loop |
| `analyze_dataset.py` | Prints severity distribution per condition with ASCII histogram |

## Pipeline

All commands run from the **project root**.

### Step 1 — Run clean baseline (stereo + mono)
```bash
python run_pipeline.py --config experiments/clean.yaml
```
Produces `results/clean/<traj>/clean/stereo/` and `clean/mono/` trajectories.
The mono trajectory is used as the per-window RTE baseline.

### Step 2 — Run degradation experiments (stereo only)
```bash
python run_pipeline.py --config experiments/new_blur1.yaml
```
Produces `results/new_blur1/<traj>/<condition>/stereo/` trajectories for each condition.

### Step 3 — Build dataset index
```bash
python health_monitor/build_dataset_index.py \
    --baseline clean \
    --experiments new_blur1
```
- Reads traj metadata from `results/<exp>/<traj>/metadata.yaml`
- Reads per-run condition params from `results/<exp>/<traj>/<cond>/stereo/metadata.yaml`
- Computes per-window RTE for each (traj, condition)
- Writes to `data/health_monitor_dataset/sequence/<traj>/`:
  - `meta.json` — cam dir paths for the dataset loader
  - `degradations/<cond>/config.json` — degradation params (applied on-the-fly)
  - `degradations/<cond>/dataset.csv` — `window_start_ts, rte_stereo, rte_mono, severity`

Multiple baselines and experiments can be passed:
```bash
python health_monitor/build_dataset_index.py \
    --baseline clean room2_clean \
    --experiments new_blur1 noise_sweep
```

#### Severity policies
```bash
# Default: ratio  (severity = rte_s / (rte_s + rte_m))
python health_monitor/build_dataset_index.py --baseline clean --experiments new_blur1

# Log-ratio  (scale-agnostic sigmoid, alpha controls sharpness)
python health_monitor/build_dataset_index.py --baseline clean --experiments new_blur1 \
    --severity-policy log_ratio --severity-params alpha=3.0
```

### Step 4 — Analyze dataset
```bash
python health_monitor/analyze_dataset.py
```
Prints per-traj/condition breakdown and overall severity histogram.

Optional override:
```bash
python health_monitor/analyze_dataset.py --dataset /path/to/health_monitor_dataset
```

### Step 5 — Train
```bash
# Quick sanity check (100 windows, no multiprocessing)
python health_monitor/train.py --quick

# Full training (30 epochs)
python health_monitor/train.py
```
Outputs:
- `checkpoints/best_model.pt` — best validation checkpoint
- `training_curves.png` — loss / MAE / crossover accuracy vs epoch
- `val_predictions.png` — scatter plot and distribution histogram

## Dataset Layout

```
data/health_monitor_dataset/
└── sequence/
    └── <traj>/
        ├── meta.json                        # cam0_eq_dir, cam1_eq_dir paths
        └── degradations/
            └── <condition>/
                ├── config.json              # degradation type + params
                └── dataset.csv              # window_start_ts, rte_stereo, rte_mono, severity
```

Images are **not** stored here — the dataset loader reads them directly from the original
TUM sequence directories (paths resolved via `meta.json`).

## Experiment YAML Format

```yaml
name: new_blur1
seed: 42
timeout: 600

sequences:
  traj1: data/TUM_original/dataset-room1_512_16   # name: path

conditions:
  blur_ks3:                  # condition name (used as output folder)
    type: gaussian_blur
    kernel_size: 3
    sigma: 50
  blur_ks7:
    type: gaussian_blur
    kernel_size: 7
    sigma: 50

slam_modes:
  - stereo                   # degradation experiments: stereo only
```

For clean baselines, `slam_modes` should include both `stereo` and `mono`.

## Resetting

To rebuild from scratch (keeps original TUM data):
```bash
rm -rf results/ data/health_monitor_dataset/ checkpoints/
```
