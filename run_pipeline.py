#!/usr/bin/env python3
"""
Entry point: run from eece5554_final_project/.

Runs mono-inertial once as a baseline, then stereo-inertial across all
noise levels. Plots stereo ATE vs noise with mono as a threshold line.

Usage:
    python run_pipeline.py
"""

from src.config import SIGMAS, RESULTS_DIR, NOISY_BASE, GT_CSV, ORIG_DATASET
from src.noise import generate_noisy_dataset
from src.convert import convert_gt_to_tum
from src.slam import run_stereo, run_mono
from src.evaluate import evaluate
from src.plot import plot_results, print_summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    NOISY_BASE.mkdir(parents=True, exist_ok=True)

    # Convert ground truth once
    gt_tum = RESULTS_DIR / "gt_room1.txt"
    print("Converting ground truth...")
    convert_gt_to_tum(GT_CSV, gt_tum)

    # ── Step 1: Run mono baseline (once, on clean data) ──
    print(f"\n{'='*50}")
    print(f"  MONO BASELINE (clean data)")
    print(f"{'='*50}")

    mono_tag = "mono_baseline"
    mono_dir = RESULTS_DIR / mono_tag
    mono_dir.mkdir(parents=True, exist_ok=True)

    run_mono(ORIG_DATASET, mono_tag, mono_dir)
    mono_stats = evaluate(mono_dir, gt_tum, mono_tag)

    if mono_stats:
        print(f"    Mono baseline ATE RMSE: {mono_stats['rmse']:.4f} m")
    else:
        print(f"    Mono baseline FAILED")

    # ── Step 2: Run stereo across noise levels ──
    stereo_results = {}

    print(f"\nRunning stereo-inertial across {len(SIGMAS)} noise levels")
    print(f"Sigmas: {SIGMAS}\n")

    for sigma in SIGMAS:
        print(f"\n{'='*50}")
        print(f"  SIGMA = {sigma}")
        print(f"{'='*50}")

        dataset_dir = generate_noisy_dataset(sigma)

        tag = f"stereo_sigma{sigma}"
        run_dir = RESULTS_DIR / tag
        run_dir.mkdir(parents=True, exist_ok=True)

        run_stereo(dataset_dir, tag, run_dir)
        stats = evaluate(run_dir, gt_tum, tag)
        stereo_results[sigma] = stats

        if stats:
            print(f"    ATE RMSE: {stats['rmse']:.4f} m")
        else:
            print(f"    FAILED")

    # ── Step 3: Report and plot ──
    print_summary(stereo_results, mono_stats)
    plot_results(stereo_results, mono_stats)


if __name__ == "__main__":
    main()