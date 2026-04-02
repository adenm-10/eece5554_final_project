#!/usr/bin/env python3
"""
ORB-SLAM experiment runner.

Usage:
    python run_pipeline.py --config experiments/orbslam_blur.yaml
"""

import argparse
import time
import yaml

from src.experiment import load_orbslam_config
from src.noise import generate_noisy_dataset_euroc
from src.config import GT_CSV
from src.convert import convert_gt_to_tum
from src.slam import run_stereo, run_mono
from src.evaluate import evaluate
from src.plot import plot_results, print_summary


def main():
    parser = argparse.ArgumentParser(description="ORB-SLAM noise benchmark")
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    args = parser.parse_args()

    cfg = load_orbslam_config(args.config)

    print(f"Experiment : {cfg.name}")
    print(f"Trajs      : {list(cfg.trajs.keys())}")
    print(f"Conditions : {cfg.conditions}")
    print(f"SLAM modes : {cfg.slam_modes}")
    print(f"Results    : {cfg.results_dir}")
    print()

    cfg.results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Generate EuRoC-format datasets per (traj, condition) ──
    print("=" * 60)
    print("  GENERATING DATASETS")
    print("=" * 60)

    dataset_paths = {}  # (traj, condition) -> Path
    for traj_name, traj_src in cfg.trajs.items():

        # write traj-level metadata once per traj
        traj_meta_path = cfg.results_dir / traj_name / "metadata.yaml"
        if not traj_meta_path.exists():
            traj_meta_path.parent.mkdir(parents=True, exist_ok=True)
            with open(traj_meta_path, "w") as f:
                yaml.dump({
                    "traj":          traj_name,
                    "experiment":    cfg.name,
                    "sequence_path": str(traj_src),
                }, f, default_flow_style=False)

        for cond in cfg.conditions:
            ds = generate_noisy_dataset_euroc(
                condition=cond,
                orig_dataset=traj_src,
                out_base=cfg.euroc_out(traj_name),
                seed=cfg.seed,
            )
            dataset_paths[(traj_name, cond)] = ds

    # ── 2. Convert ground truth ──
    gt_tum = cfg.results_dir / "gt.txt"
    if not gt_tum.exists():
        print("\nConverting ground truth...")
        convert_gt_to_tum(GT_CSV, gt_tum)

    # ── 3. Run SLAM + evaluate ──
    print("\n" + "=" * 60)
    print("  RUNNING SLAM")
    print("=" * 60)

    all_results = {}
    manifest = []

    for traj, cond, mode in cfg.run_combos():
        tag = f"{traj}_{cond}_{mode}"
        run_dir = cfg.run_dir(traj, cond, mode)
        dataset_dir = dataset_paths[(traj, cond)]

        print(f"\n{'─'*50}")
        print(f"  {tag}")
        print(f"{'─'*50}")

        # Skip if cached
        stats_file = run_dir / "stats.yaml"
        if stats_file.exists():
            with open(stats_file) as f:
                cached = yaml.safe_load(f)
            all_results[(traj, cond, mode)] = cached
            print(f"  Cached. RMSE={cached.get('rmse', 'N/A')}")
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        if mode == "stereo":
            run_stereo(dataset_dir, tag, run_dir)
        elif mode == "mono":
            run_mono(dataset_dir, tag, run_dir)

        stats = evaluate(run_dir, gt_tum, tag)
        elapsed = time.time() - t0

        entry = {"traj": traj, "condition": cond, "mode": mode, "elapsed_s": round(elapsed, 1)}
        if stats:
            entry.update(stats)
            print(f"  ATE RMSE: {stats['rmse']:.4f} m  ({elapsed:.0f}s)")
        else:
            entry["status"] = "FAILED"
            print(f"  FAILED ({elapsed:.0f}s)")

        with open(stats_file, "w") as f:
            yaml.dump(entry, f, default_flow_style=False)

        # write per-run metadata snapshot
        with open(run_dir / "metadata.yaml", "w") as f:
            yaml.dump({
                "traj":             traj,
                "condition":        cond,
                "mode":             mode,
                "experiment":       cfg.name,
                "condition_params": cfg.conditions_cfg.get(cond),
            }, f, default_flow_style=False)

        all_results[(traj, cond, mode)] = entry
        manifest.append(entry)

    # ── 4. Manifest ──
    manifest_path = cfg.results_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump({
            "experiment": cfg.name,
            "seed": cfg.seed,
            "conditions": cfg.conditions,
            "slam_modes": cfg.slam_modes,
            "runs": manifest,
        }, f, default_flow_style=False)

    # ── 5. Summary + plot ──
    print_summary(all_results, cfg.results_dir)
    plot_results(all_results, cfg.results_dir, experiment_name=cfg.name)

    print(f"\nManifest: {manifest_path}")
    print(f"Done.")


if __name__ == "__main__":
    main()