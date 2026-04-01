"""
Dataset analyzer — summarizes severity distribution per condition.

Run after build_dataset_index.py:
    python health_monitor/analyze_dataset.py
    python health_monitor/analyze_dataset.py --dataset data/health_monitor_dataset
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEVERITY_BINS = [
    (0.0,  0.25, "healthy     "),
    (0.25, 0.50, "mild        "),
    (0.50, 0.75, "degraded    "),
    (0.75, 1.00, "severe      "),
    (1.00, 1.01, "failed(=1.0)"),
]


def analyze(dataset_root: Path):
    degrad_root = dataset_root / "sequence" / "traj1" / "degradations"
    if not degrad_root.exists():
        raise FileNotFoundError(f"No degradations found at {degrad_root}")

    cond_dirs = sorted(d for d in degrad_root.iterdir() if d.is_dir())
    all_data = []

    print(f"\n{'Condition':<28} {'Windows':>7} {'Failed':>7} {'Mean sev':>9} "
          f"{'<0.25':>7} {'0.25-0.5':>9} {'0.5-0.75':>9} {'>0.75':>7} {'=1.0':>6}")
    print("─" * 100)

    for cond_dir in cond_dirs:
        csv_path = cond_dir / "dataset.csv"
        if not csv_path.exists():
            print(f"  {cond_dir.name:<26}  no dataset.csv — run build_dataset_index.py")
            continue

        df  = pd.read_csv(csv_path)
        sev = df["severity"]
        n   = len(sev)

        n_failed = (sev == 1.0).sum()
        n_inf    = np.isinf(df["rte_stereo"]).sum()

        counts = []
        for lo, hi, _ in SEVERITY_BINS:
            if hi > 1.0:
                counts.append((sev == 1.0).sum())
            else:
                counts.append(((sev >= lo) & (sev < hi)).sum())

        print(f"  {cond_dir.name:<26}  {n:>6}  {n_inf:>6}  {sev.mean():>8.3f}  "
              f"{counts[0]:>6}  {counts[1]:>8}  {counts[2]:>8}  {counts[3]:>6}  {counts[4]:>5}")

        all_data.append({"condition": cond_dir.name, "severity": sev})

    # overall histogram across all conditions
    if all_data:
        combined = pd.concat([d["severity"] for d in all_data], ignore_index=True)
        print("─" * 100)
        n = len(combined)
        n_inf = sum(np.isinf(pd.read_csv(
            degrad_root / d["condition"] / "dataset.csv")["rte_stereo"]).sum()
            for d in all_data)
        counts = []
        for lo, hi, _ in SEVERITY_BINS:
            if hi > 1.0:
                counts.append((combined == 1.0).sum())
            else:
                counts.append(((combined >= lo) & (combined < hi)).sum())
        print(f"  {'ALL':<26}  {n:>6}  {n_inf:>6}  {combined.mean():>8.3f}  "
              f"{counts[0]:>6}  {counts[1]:>8}  {counts[2]:>8}  {counts[3]:>6}  {counts[4]:>5}")

        print(f"\nSeverity breakdown (all conditions, n={n}):")
        for (lo, hi, label), count in zip(SEVERITY_BINS, counts):
            bar = "█" * int(40 * count / n)
            print(f"  {label}  {count:>5} ({100*count/n:5.1f}%)  {bar}")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None,
                        help="Path to health_monitor_dataset root "
                             "(default: data/health_monitor_dataset)")
    args = parser.parse_args()

    dataset_root = Path(args.dataset) if args.dataset \
                   else PROJECT_ROOT / "data" / "health_monitor_dataset"

    analyze(dataset_root)
