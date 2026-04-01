# activate environment
source <your-env>/bin/activate

# ── Step 1: Run clean baseline (stereo + mono) ────────────────────────────────
python run_pipeline.py --config experiments/clean.yaml

# ── Step 2: Run degradation experiments (stereo only) ────────────────────────
python run_pipeline.py --config experiments/new_blur1.yaml

# ── Step 3: Build dataset index ───────────────────────────────────────────────
# reads YAMLs → writes config.json + dataset.csv per condition
python health_monitor/build_dataset_index.py \
    --baseline clean \
    --experiments new_blur1

# ── Step 4: Analyze dataset ───────────────────────────────────────────────────
python health_monitor/analyze_dataset.py

# ── Step 5: Train ─────────────────────────────────────────────────────────────
# quick test first
python health_monitor/train.py --quick

# full training
python health_monitor/train.py
