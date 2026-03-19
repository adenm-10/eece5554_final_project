import csv
import matplotlib.pyplot as plt
from src.config import SIGMAS, RESULTS_DIR


def plot_results(stereo_results, mono_stats=None):
    """Plot stereo ATE RMSE vs noise, with mono baseline as a horizontal line."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot mono baseline as horizontal threshold line
    if mono_stats is not None:
        mono_rmse = mono_stats["rmse"]
        ax.axhline(y=mono_rmse, color="#FF5722", linestyle="--", linewidth=2,
                    label=f"Mono-inertial baseline ({mono_rmse:.4f} m)")

    # Plot stereo results
    sigmas = []
    rmses = []
    for sigma, stats in sorted(stereo_results.items()):
        if stats is not None:
            sigmas.append(sigma)
            rmses.append(stats["rmse"])

    if sigmas:
        ax.plot(sigmas, rmses, "o-", color="#2196F3", linewidth=2,
                markersize=8, label="Stereo-inertial")

        # Shade region where stereo is worse than mono
        if mono_stats is not None:
            ax.fill_between(sigmas, mono_rmse, rmses,
                            where=[r > mono_rmse for r in rmses],
                            alpha=0.15, color="#FF5722",
                            label="Stereo worse than mono")

    ax.set_xlabel("Gaussian Noise σ (pixel intensity)", fontsize=12)
    ax.set_ylabel("ATE RMSE (m)", fontsize=12)
    ax.set_title("ORB-SLAM3: Stereo-Inertial Degradation vs. Right-Camera Noise\n"
                 "(TUM-VI room1, mono baseline as threshold)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=-2)

    plot_path = RESULTS_DIR / "ate_vs_noise.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved: {plot_path}")
    plt.show()


def print_summary(stereo_results, mono_stats=None):
    """Print a summary table and save to CSV."""
    csv_path = RESULTS_DIR / "summary.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "sigma", "ate_rmse", "ate_mean", "ate_median", "ate_std"])

        print(f"\n{'='*65}")
        print(f"{'Mode':<10} {'Sigma':>6} {'RMSE':>10} {'Mean':>10} {'Median':>10} {'Std':>10}")
        print(f"{'-'*65}")

        # Mono baseline row
        if mono_stats:
            rmse, mean, median, std = mono_stats["rmse"], mono_stats["mean"], mono_stats["median"], mono_stats["std"]
            writer.writerow(["mono", "baseline", f"{rmse:.6f}", f"{mean:.6f}", f"{median:.6f}", f"{std:.6f}"])
            print(f"{'mono':<10} {'base':>6}   {rmse:>10.4f} {mean:>10.4f} {median:>10.4f} {std:>10.4f}  (threshold)")
            print(f"{'-'*65}")

        # Stereo rows
        baseline_rmse = stereo_results.get(0, {})
        if baseline_rmse:
            baseline_rmse = baseline_rmse.get("rmse")

        for sigma in SIGMAS:
            stats = stereo_results.get(sigma)
            if stats is None:
                writer.writerow(["stereo", sigma, "FAIL", "FAIL", "FAIL", "FAIL"])
                print(f"{'stereo':<10} {sigma:>6}   {'FAIL':>10}")
                continue

            rmse, mean, median, std = stats["rmse"], stats["mean"], stats["median"], stats["std"]
            writer.writerow(["stereo", sigma, f"{rmse:.6f}", f"{mean:.6f}", f"{median:.6f}", f"{std:.6f}"])

            notes = ""
            if baseline_rmse and baseline_rmse > 0 and sigma > 0:
                pct = ((rmse - baseline_rmse) / baseline_rmse) * 100
                notes = f"  ({pct:+.1f}%)"
            if mono_stats and rmse > mono_stats["rmse"]:
                notes += "  *** WORSE THAN MONO ***"

            print(f"{'stereo':<10} {sigma:>6}   {rmse:>10.4f} {mean:>10.4f} {median:>10.4f} {std:>10.4f}{notes}")

        print(f"{'='*65}")

    print(f"Summary saved: {csv_path}")