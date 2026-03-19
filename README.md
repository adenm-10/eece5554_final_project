# ORB-SLAM3 Stereo Noise Robustness Benchmark

Measures how Gaussian noise on one stereo camera degrades ORB-SLAM3 accuracy on TUM-VI, using mono-inertial performance as a baseline threshold.

## Setup

**Prerequisites:** Linux, cmake, gcc/g++, OpenCV 4.x, Eigen3, Python 3.8+.

**1. Build ORB-SLAM3:**

```bash
git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git
cd ORB_SLAM3
chmod +x build.sh
./build.sh
cd ..
```

This builds the library, vocabulary, and example binaries. See the [ORB-SLAM3 README](https://github.com/UZ-SLAMLab/ORB_SLAM3) for dependency details (Pangolin, OpenCV, Eigen3, DBoW2, g2o).

**2. Install Python dependencies:**

```bash
pip install opencv-python-headless numpy matplotlib evo
```

Use `opencv-python-headless` (not `opencv-python`) to avoid Qt plugin conflicts with Pangolin.

**3. Download TUM-VI dataset:**

```bash
mkdir -p data/TUM_original
cd data/TUM_original
wget https://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-room1_512_16.tar
tar xf dataset-room1_512_16.tar
cd ../..
```

The TUM-VI timestamp files (`TUM_TimeStamps/`) and IMU files (`TUM_IMU/`) ship with ORB-SLAM3 under `ORB_SLAM3/Examples/Stereo-Inertial/`. The pipeline references them from `data/TUM_original/` — copy or symlink them there:

```bash
cp -r ORB_SLAM3/Examples/Stereo-Inertial/TUM_TimeStamps data/TUM_original/
cp -r ORB_SLAM3/Examples/Stereo-Inertial/TUM_IMU data/TUM_original/
```

## Usage

```bash
python run_pipeline.py
```

This will:

1. Run mono-inertial once on clean data as a baseline
2. Generate noisy copies of cam1 (right camera) at each sigma level
3. Run stereo-inertial on each noisy dataset
4. Evaluate all trajectories against mocap ground truth using `evo_ape`
5. Print a summary table and save a plot to `results/noise_benchmark/`

Edit `src/config.py` to change noise levels, timeout, or dataset paths.

## Project Structure

```
eece5554_final_project/
├── run_pipeline.py          # Entry point — runs the full pipeline
├── src/
│   ├── config.py            # Paths, noise levels, constants
│   ├── noise.py             # Generates noisy cam1 images (symlinks cam0/imu)
│   ├── convert.py           # TUM-VI mocap and ORB-SLAM3 output → TUM format
│   ├── slam.py              # Launches stereo/mono ORB-SLAM3 via subprocess
│   ├── evaluate.py          # Runs evo_ape, parses ATE metrics
│   └── plot.py              # Plots stereo RMSE vs noise with mono threshold
├── ORB_SLAM3/               # Built ORB-SLAM3 (vocabulary, examples, libs)
├── data/
│   ├── TUM_original/        # Clean TUM-VI dataset + timestamps
│   └── noisy_datasets/      # Generated noisy datasets (auto-created)
└── results/
    └── noise_benchmark/     # Trajectory files, evo results, plots, summary.csv
```

## Output

After a run, `results/noise_benchmark/` contains:

- `summary.csv` — ATE metrics for each noise level, with degradation percentages
- `ate_vs_noise.png` — Stereo RMSE curve with mono baseline as a dashed threshold line
- `stereo_sigma*/` — Per-run logs, trajectories, and evo results
- `mono_baseline/` — Mono-inertial baseline run

## Configuration

All tunables are in `src/config.py`:

| Variable | Default | Description |
|---|---|---|
| `SIGMAS` | `[0, 5, 10, 20, 40, 80]` | Noise standard deviations (pixel intensity, 0–255) |
| `TIMEOUT` | `600` | Max seconds per ORB-SLAM3 run |
| `SEED` | `42` | RNG seed for reproducible noise |
| `ORIG_DATASET` | `data/TUM_original/dataset-room1_512_16` | Path to clean TUM-VI sequence |

## Notes

- Noisy datasets symlink cam0, imu0, and mocap0 from the original to save disk space. Only cam1 images are copied with noise.
- ORB-SLAM3 runs headless (`QT_QPA_PLATFORM=offscreen`). To see the viewer, remove that line from `src/slam.py`.
- The mono baseline uses cam0 (always clean) and serves as an upper bound — once stereo crosses this line, the noisy right camera is actively hurting performance.
- Re-running skips noise generation for datasets that already exist.

## Using a Different TUM-VI Sequence

All TUM-VI sequences are available in EuRoC format at:

```
https://vision.in.tum.de/tumvi/exported/euroc/512_16/
```

To switch to a different sequence (e.g. corridor1):

**1. Download and extract:**

```bash
cd data/TUM_original
wget https://vision.in.tum.de/tumvi/exported/euroc/512_16/dataset-corridor1_512_16.tar
tar xf dataset-corridor1_512_16.tar
```

**2. Update three paths in `src/config.py`:**

```python
ORIG_DATASET = PROJECT_ROOT / "data" / "TUM_original" / "dataset-corridor1_512_16"
TIMESTAMPS = PROJECT_ROOT / "data" / "TUM_original" / "TUM_TimeStamps" / "dataset-corridor1_512.txt"
GT_CSV = ORIG_DATASET / "mav0" / "mocap0" / "data.csv"
```

The IMU path (`IMU_CSV`) doesn't need changing — it's relative to `ORIG_DATASET`.

**3. Clean old results and re-run:**

```bash
rm -rf results/noise_benchmark/ data/noisy_datasets/
python run_pipeline.py
```

Available sequence types and their ground truth coverage:

| Type | Sequences | Ground truth |
|---|---|---|
| room | room1–room6 (~1.3–1.6 GB) | Full trajectory |
| corridor | corridor1–corridor5 (~1–3.7 GB) | Start and end segments |
| magistrale | magistrale1–magistrale6 (~5.4–9.6 GB) | Start and end segments |
| outdoors | outdoors1–outdoors8 (~8.4–19 GB) | Start and end segments |
| slides | slides1–slides3 (~2.9–4.2 GB) | Start and end segments |

The **room** sequences are recommended for benchmarking since they have ground truth for the entire trajectory. Use 1024x1024 versions (swap `512_16` for `1024_16` in URLs) for higher resolution at ~3x the file size.