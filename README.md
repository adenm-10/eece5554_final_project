# Stereo Camera Health Monitor for ORB-SLAM3

Characterizes how five types of image degradation affect stereo-inertial SLAM accuracy,
identifies the crossover point where switching to monocular becomes beneficial, and trains
a learned health monitor that predicts degradation severity from raw stereo image pairs.

**Dataset:** TUM-VI (room1, room2) | **SLAM:** ORB-SLAM3 | **Monitor:** ResNet18 + GRU

---

## What This Project Does

When one stereo camera lens degrades — through blur, occlusion, dirt, or underexposure —
ORB-SLAM3 continues running without any alert. Trajectory error accumulates silently until
tracking fails. This project:

1. **Phase 1 — Characterisation:** Sweeps five degradation types across two TUM-VI room
   sequences, measuring ATE RMSE at each severity level. Identifies failure thresholds and
   classifies which types produce a learnable warning zone vs. a binary cliff.

2. **Phase 2 — Health Monitor:** Trains a ResNet18 + GRU model on 20-frame stereo windows
   labeled with a self-supervised severity score derived from the Phase 1 SLAM outputs.
   The model predicts severity ∈ [0, 1]; a score ≥ 0.5 means the system should switch to
   mono-inertial mode.

### Severity Metric

```
severity = RTE_stereo / (RTE_stereo + RTE_mono)
```

- `0.0` — stereo performing perfectly relative to mono
- `0.5` — crossover: stereo and mono are equivalent → switch
- `1.0` — stereo has failed entirely

No manual annotation is required. Labels are derived entirely from SLAM trajectory outputs.

---

## Project Structure

```
eece5554_final_project/
│
├── run_pipeline.py              # Phase 1: run ORB-SLAM3 degradation sweeps
│
├── src/                         # Core pipeline utilities
│   ├── config.py                # Paths, seeds, constants
│   ├── experiment.py            # Loads experiment YAML, dispatches SLAM runs
│   ├── slam.py                  # Launches ORB-SLAM3 via subprocess (headless)
│   ├── evaluate.py              # Runs evo_ape, parses ATE metrics
│   ├── convert.py               # TUM-VI mocap → TUM trajectory format
│   ├── noise.py                 # Image degradation functions (applied on-the-fly)
│   └── plot.py                  # ATE vs degradation plots
│
├── experiments/                 # Experiment YAML configs (one per sweep)
│   ├── clean.yaml               # Clean stereo + mono baselines
│   ├── blur_sweep_B.yaml        # Gaussian blur sweep (Option B: kernel = 6σ)
│   ├── motion_blur_sweep.yaml   # Motion blur sweep
│   ├── snp_sweep.yaml           # Salt & pepper noise sweep
│   ├── brightness_sweep.yaml    # Brightness scale sweep
│   └── occlusion_sweep.yaml     # Occlusion fraction sweep
│
├── health_monitor/              # Phase 2: model training pipeline
│   ├── config.py                # Shared paths and constants
│   ├── build_dataset_index.py   # Computes per-window severity from SLAM results
│   ├── stereo_health_dataset.py # PyTorch Dataset; applies degradation on-the-fly
│   ├── train.py                 # Model definition (ResNet18+GRU) + training loop
│   ├── analyze_dataset.py       # Severity distribution analysis per condition
│   └── README.md                # Health monitor pipeline details
│
├── analysis/                    # Analysis scripts and reports
│   ├── phase1_analysis.py       # Generates all Phase 1 plots (both trajectories)
│   ├── model_sanity_check.py    # Probes trained model on known conditions
│   ├── tsne_pairs.py            # t-SNE on frozen ResNet18 embeddings
│   ├── tsne_trained_model.py    # t-SNE on trained model GRU hidden state
│   ├── key_findings.tex         # Phase 1 key findings report
│   ├── phase2_key_findings.tex  # Phase 2 key findings report
│   ├── report.tex               # Final project report
│   └── sanitycheck.tex          # Model sanity check report with figures
│
├── results/                     # SLAM run outputs (auto-created, not committed)
│   └── <exp>/<traj>/<cond>/stereo|mono/
│       ├── trajectory.txt
│       └── stats.yaml           # ATE RMSE and other metrics
│
├── data/                        # Datasets (not committed)
│   ├── TUM_original/            # Raw TUM-VI sequences
│   └── health_monitor_dataset/  # Built dataset index (window CSVs, configs)
│
└── checkpoints/                 # Saved model weights (not committed)
    └── best_model.pt
```

---

## Setup

**Prerequisites:** Linux, Python 3.10+, CUDA GPU (recommended), ORB-SLAM3 built.

### 1. Build ORB-SLAM3

```bash
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3 && chmod +x build.sh && ./build.sh && cd ..
```

### 2. Install Python dependencies

```bash
pip install torch torchvision opencv-python-headless numpy pandas matplotlib \
            scikit-learn evo pyyaml
```

Use `opencv-python-headless` (not `opencv-python`) to avoid Qt conflicts with Pangolin.

### 3. Download TUM-VI sequences

```bash
mkdir -p data/TUM_original && cd data/TUM_original
wget https://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar
wget https://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room2_512_16.tar
tar xf dataset-room1_512_16.tar && tar xf dataset-room2_512_16.tar
cd ../..
```

Copy timestamps and IMU files from ORB-SLAM3:

```bash
cp -r ORB_SLAM3/Examples/Stereo-Inertial/TUM_TimeStamps data/TUM_original/
cp -r ORB_SLAM3/Examples/Stereo-Inertial/TUM_IMU       data/TUM_original/
```

---

## Phase 1 — Degradation Sweep

### Step 1: Run clean baselines (stereo + mono)

```bash
python run_pipeline.py --config experiments/clean.yaml
```

Produces `results/clean/<traj>/clean/stereo/` and `clean/mono/` for each trajectory.
The mono trajectory ATE becomes the switching threshold for that sequence.

### Step 2: Run degradation sweeps

```bash
python run_pipeline.py --config experiments/blur_sweep_B.yaml
python run_pipeline.py --config experiments/motion_blur_sweep.yaml
python run_pipeline.py --config experiments/snp_sweep.yaml
python run_pipeline.py --config experiments/brightness_sweep.yaml
python run_pipeline.py --config experiments/occlusion_sweep.yaml
```

Each sweep applies degradation to the right camera only, across increasing severity levels.
Once a condition causes tracking failure, subsequent conditions are auto-skipped.

### Step 3: Generate analysis plots

```bash
python analysis/phase1_analysis.py
```

Outputs 6 PNGs to `analysis/`: one per degradation type + a combined 5-panel figure.

---

## Phase 2 — Health Monitor Training

### Step 4: Build dataset index

```bash
python health_monitor/build_dataset_index.py \
    --baseline clean \
    --experiments blur_sweep_B motion_blur_sweep snp_sweep brightness_sweep occlusion_sweep \
    --skip-failed
```

Reads SLAM results, computes per-window RTE, and writes severity labels to
`data/health_monitor_dataset/sequence/<traj>/degradations/<cond>/dataset.csv`.
Images are **not** copied — the dataset loader reads them from the original TUM directories.

Inspect the severity distribution:

```bash
python health_monitor/analyze_dataset.py
```

### Step 5: Train the model

```bash
# Quick sanity check (100 windows, ~1 min)
python health_monitor/train.py --quick

# Full training (30 epochs, ~2–4 hours on GPU)
python health_monitor/train.py
```

Outputs saved to:
- `checkpoints/best_model.pt` — best validation checkpoint
- `health_monitor/training_curves.png` — loss / MAE / crossover accuracy vs epoch
- `health_monitor/val_predictions.png` — scatter and distribution histogram

### Step 6: Sanity check

```bash
python analysis/model_sanity_check.py
```

Probes the trained model on 65 windows across 13 known conditions (5 per condition).
Outputs per-window filmstrip PNGs and a crossover classification summary figure to
`analysis/`. Prints MAE and crossover accuracy to stdout.

---

## Model Architecture

```
Per frame (left + right):
  left_img, right_img → ResNet18 (shared, layer4 unfrozen) → left_emb, right_emb (512-d each)
  diff_emb = left_emb - right_emb          ← captures left-right asymmetry
  frame_emb = Linear(1536 → 128)([left, right, diff])

Sequence (20 frames):
  frame_embs (20 × 128) → GRU(hidden=64) → last hidden state (64-d)

Output:
  Linear(64→32) → ReLU → Dropout(0.3) → Linear(32→1) → Sigmoid → severity ∈ [0, 1]

Total params: ~11.2M  |  Trainable: layer4 + head
```

**Training details:** Huber loss (δ=0.1), Adam (backbone LR 1e-5, head LR 1e-3),
ReduceLROnPlateau, 30 epochs, batch 16, weighted random sampler (4 severity bins),
10,830 windows from traj1 + traj2.

---

## Key Results

| Metric | Value |
|---|---|
| Val MAE | 0.135 |
| Crossover accuracy (val set) | ~75% |
| Crossover accuracy (sanity check, 65 windows) | 80% |
| Gaussian blur failure cliff | σ = 6.1 (room1), σ = 6.0 (room2) |
| Motion blur failure cliff | 28 px (room1), 27 px (room2) |
| S&P / Occlusion pattern | Flat ATE → binary cliff (RANSAC absorbs corruption) |
| Brightness asymmetry | Overexposure benign; underexposure (α < 0.03) catastrophic |

---

## Experiment YAML Format

```yaml
name: blur_sweep_B
seed: 42
timeout: 600

sequences:
  traj1: data/TUM_original/dataset-room1_512_16
  traj2: data/TUM_original/dataset-room2_512_16

conditions:
  blur_lv1:
    type: gaussian_blur
    kernel_size: 3
    sigma: 0.5
  blur_lv2:
    type: gaussian_blur
    kernel_size: 7
    sigma: 1.0

slam_modes:
  - stereo    # degradation sweeps: stereo only
              # clean.yaml includes both stereo and mono
```

Supported degradation types: `gaussian_blur`, `motion_blur`, `salt_and_pepper`,
`brightness`, `occlusion`.
