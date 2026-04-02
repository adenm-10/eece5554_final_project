"""
Build dataset.csv and config.json for each degradation condition.

Reads experiment YAMLs to get condition params, finds SLAM trajectories in
results/, writes config.json + dataset.csv into the health_monitor_dataset.

Usage:
    python health_monitor/build_dataset_index.py \
        --baseline clean \
        --experiments new_blur1 degrade

    --baseline    : name of experiment that holds the clean mono/stereo run
                    (looks for results/<baseline>/traj1/clean/mono/traj_tum.txt)
    --experiments : one or more experiment names whose degradation conditions
                    should be included (looks for results/<exp>/traj1/<cond>/stereo/traj_tum.txt)
    --dataset     : optional override for health_monitor_dataset root
"""

import argparse
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
CAM0_DIR      = PROJECT_ROOT / "data" / "TUM_original" / "dataset-room1_512_16" / "mav0" / "cam0_eq" / "data"
GT_CSV        = PROJECT_ROOT / "data" / "TUM_original" / "dataset-room1_512_16" / "mav0" / "mocap0" / "data.csv"
RESULTS_ROOT  = PROJECT_ROOT / "results"
EXPERIMENTS   = PROJECT_ROOT / "experiments"

WINDOW_SIZE   = 20
MAX_DIFF_NS   = 10_000_000   # 10 ms
MIN_COVERAGE  = 0.75


# ── YAML loader ───────────────────────────────────────────────────────────────

def load_conditions_from_yaml(exp_name: str) -> dict:
    """
    Load conditions dict from experiments/<exp_name>.yaml.
    Returns {condition_name: params_dict_or_None}.
    Skips 'clean' — that is the baseline, not a degradation.
    """
    yaml_path = EXPERIMENTS / f"{exp_name}.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Experiment YAML not found: {yaml_path}")

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    conditions_raw = raw.get("conditions", {})

    # support both old list format and new dict format
    if isinstance(conditions_raw, list):
        # old format: conditions is a plain list of strings
        return {c: None for c in conditions_raw if c != "clean"}
    else:
        # new format: conditions is a dict with params
        return {k: v for k, v in conditions_raw.items() if k != "clean"}


def params_to_config(params) -> dict:
    """
    Convert a YAML condition params dict to the config.json contract.
    None or missing params means clean/no degradation.
    Passes params through as-is — type strings come directly from the YAML.
    """
    if params is None:
        return {"type": "none"}
    return dict(params)   # shallow copy, preserves all fields


# ── Severity policies ─────────────────────────────────────────────────────────

class SeverityPolicy(ABC):
    """
    Base class for severity scoring.

    Subclasses receive rte_stereo and rte_mono for a single window and return
    a severity in [0, 1].  SLAM failure (rte_stereo=inf) is handled by the
    caller and always returns 1.0 — policies only need to handle finite values.

    To add a new policy:
      1. Subclass SeverityPolicy, implement compute().
      2. Register it in SEVERITY_POLICIES below.
      3. Pass --severity-policy <name> [--severity-params key=val ...] on the CLI.
    """

    @abstractmethod
    def compute(self, rte_s: float, rte_m: float) -> float:
        """Return severity in [0, 1] given finite stereo and mono RTE."""
        ...

    @classmethod
    def from_config(cls, name: str, params: dict) -> "SeverityPolicy":
        if name not in SEVERITY_POLICIES:
            raise ValueError(
                f"Unknown severity policy '{name}'. "
                f"Available: {list(SEVERITY_POLICIES.keys())}"
            )
        return SEVERITY_POLICIES[name](**params)


class RatioPolicy(SeverityPolicy):
    """
    severity = rte_s / (rte_s + rte_m)

    Crossover at rte_s == rte_m → 0.5.
    Simple and interpretable but has weak signal near crossover.
    No parameters.
    """

    def compute(self, rte_s: float, rte_m: float) -> float:
        denom = rte_s + rte_m
        return float(np.clip(rte_s / denom, 0.0, 1.0)) if denom > 1e-9 else 0.0


class LogRatioPolicy(SeverityPolicy):
    """
    severity = sigmoid(alpha * log(rte_s / rte_m))

    Sequence-agnostic: the log-ratio cancels absolute scale so the metric
    behaves consistently across indoor/outdoor/corridor sequences.

    Crossover is at rte_s == rte_m (ratio=1, log=0) → sigmoid(0) = 0.5.
    Post-crossover signal grows quickly because log is sensitive to
    multiplicative changes.

    Parameters
    ----------
    alpha : float (default 3.0)
        Controls slope at the crossover.  Higher = sharper decision boundary.
        alpha=3 means rte_s = 2*rte_m gives severity ≈ 0.90.
    """

    def __init__(self, alpha: float = 3.0):
        self.alpha = alpha

    def compute(self, rte_s: float, rte_m: float) -> float:
        if rte_m < 1e-9:
            return 1.0
        log_ratio = np.log(rte_s / rte_m)
        return float(1.0 / (1.0 + np.exp(-self.alpha * log_ratio)))


# Registry — add new policies here
SEVERITY_POLICIES: dict[str, type[SeverityPolicy]] = {
    "ratio":     RatioPolicy,
    "log_ratio": LogRatioPolicy,
}


# ── Trajectory I/O ────────────────────────────────────────────────────────────

def load_tum_traj(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.array([], dtype=np.int64), np.empty((0, 3))
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    return (data[:, 0] * 1e9).astype(np.int64), data[:, 1:4]


def load_gt(path):
    df  = pd.read_csv(path, comment="#", header=None)
    ts  = df.iloc[:, 0].values.astype(np.int64)
    pos = df.iloc[:, 1:4].values.astype(np.float64)
    return ts, pos


def nearest_match(query_ts, ref_ts):
    indices  = np.searchsorted(ref_ts, query_ts)
    indices  = np.clip(indices, 0, len(ref_ts) - 1)
    left     = np.clip(indices - 1, 0, len(ref_ts) - 1)
    use_left = np.abs(ref_ts[left] - query_ts) < np.abs(ref_ts[indices] - query_ts)
    indices[use_left] = left[use_left]
    valid = np.abs(ref_ts[indices] - query_ts) <= MAX_DIFF_NS
    return indices, valid


# ── RTE helpers ───────────────────────────────────────────────────────────────

def window_rte(pos_est, pos_gt):
    errors = np.linalg.norm(np.diff(pos_est, axis=0) - np.diff(pos_gt, axis=0), axis=1)
    return float(np.sqrt(np.mean(errors ** 2)))


def make_failed_rows(cam_ts, mono_window_rtes):
    rows = []
    n_windows = len(cam_ts) // WINDOW_SIZE
    for w in range(n_windows):
        rte_m = mono_window_rtes.get(w)
        rows.append({
            "window_start_ts": int(cam_ts[w * WINDOW_SIZE]),
            "rte_stereo":      float("inf"),
            "rte_mono":        round(rte_m, 6) if rte_m is not None else float("nan"),
            "severity":        1.0,
        })
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def build_index(dataset_root: Path, baseline_name: str, experiment_names: list[str],
                policy: SeverityPolicy | None = None):
    if policy is None:
        policy = RatioPolicy()

    mono_traj_path = RESULTS_ROOT / baseline_name / "traj1" / "clean" / "mono" / "traj_tum.txt"
    if not mono_traj_path.exists():
        raise FileNotFoundError(
            f"Mono baseline not found: {mono_traj_path}\n"
            f"Run: python run_pipeline.py --config experiments/clean.yaml"
        )

    # camera frames
    cam_frames = sorted(CAM0_DIR.glob("*.png"))
    if not cam_frames:
        raise FileNotFoundError(f"No frames found in {CAM0_DIR}")
    cam_ts   = np.array([int(f.stem) for f in cam_frames], dtype=np.int64)
    n_frames = len(cam_ts)
    print(f"Camera frames : {n_frames}")

    # ground truth
    gt_ts, gt_pos = load_gt(GT_CSV)
    print(f"GT poses      : {len(gt_ts)}")

    # mono baseline
    mono_ts, mono_pos = load_tum_traj(mono_traj_path)
    if len(mono_ts) == 0:
        raise RuntimeError(f"Mono baseline trajectory is empty: {mono_traj_path}")
    print(f"Mono traj     : {len(mono_ts)} poses")

    # precompute per-window mono RTE once
    n_windows = n_frames // WINDOW_SIZE
    mono_window_rtes = {}
    for w in range(n_windows):
        win_ts = cam_ts[w * WINDOW_SIZE : (w + 1) * WINDOW_SIZE]
        g_idx, g_valid = nearest_match(win_ts, gt_ts)
        m_idx, m_valid = nearest_match(win_ts, mono_ts)
        both = m_valid & g_valid
        if both.sum() >= int(MIN_COVERAGE * WINDOW_SIZE):
            mono_window_rtes[w] = window_rte(
                mono_pos[m_idx[both]], gt_pos[g_idx[both]]
            )
        else:
            mono_window_rtes[w] = None
    print()

    # collect conditions from all experiment YAMLs
    all_conditions = {}   # {cond_name: (config_dict, traj_path)}
    for exp_name in experiment_names:
        cond_params = load_conditions_from_yaml(exp_name)
        for cond, params in cond_params.items():
            traj_path = RESULTS_ROOT / exp_name / "traj1" / cond / "stereo" / "traj_tum.txt"
            if not traj_path.exists():
                print(f"[WARN] No trajectory for {exp_name}/{cond} — skipping")
                continue
            if cond in all_conditions:
                print(f"[WARN] Condition '{cond}' appears in multiple experiments, "
                      f"using first occurrence")
                continue
            all_conditions[cond] = (params_to_config(params), traj_path)

    if not all_conditions:
        print("No conditions found. Check --experiments and results/ folder.")
        return

    # process each condition
    degrad_root = dataset_root / "sequence" / "traj1" / "degradations"
    for cond, (cfg, traj_path) in sorted(all_conditions.items()):
        print(f"[{cond}]")
        cond_dir = degrad_root / cond
        cond_dir.mkdir(parents=True, exist_ok=True)

        # write config.json
        with open(cond_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

        # load stereo trajectory
        stereo_ts, stereo_pos = load_tum_traj(traj_path)

        if len(stereo_ts) == 0:
            print(f"  SLAM failed — all windows severity=1.0, rte_stereo=inf")
            rows = make_failed_rows(cam_ts, mono_window_rtes)
        else:
            print(f"  Stereo poses : {len(stereo_ts)}")
            rows = []
            for w in range(n_windows):
                win_ts = cam_ts[w * WINDOW_SIZE : (w + 1) * WINDOW_SIZE]
                s_idx, s_valid = nearest_match(win_ts, stereo_ts)
                g_idx, g_valid = nearest_match(win_ts, gt_ts)
                both   = s_valid & g_valid
                rte_m  = mono_window_rtes[w]

                if both.sum() / WINDOW_SIZE < MIN_COVERAGE:
                    rows.append({
                        "window_start_ts": int(win_ts[0]),
                        "rte_stereo":      float("inf"),
                        "rte_mono":        round(rte_m, 6) if rte_m else float("nan"),
                        "severity":        1.0,
                    })
                    continue

                rte_s = window_rte(stereo_pos[s_idx[both]], gt_pos[g_idx[both]])
                if rte_m is None:
                    severity = 1.0
                else:
                    severity = policy.compute(rte_s, rte_m)

                rows.append({
                    "window_start_ts": int(win_ts[0]),
                    "rte_stereo":      round(rte_s, 6),
                    "rte_mono":        round(rte_m, 6) if rte_m else float("nan"),
                    "severity":        round(severity, 4),
                })

        df = pd.DataFrame(rows)
        df.to_csv(cond_dir / "dataset.csv", index=False)
        print(f"  Windows  : {len(rows)}  →  {cond_dir / 'dataset.csv'}")
        sev = df["severity"]
        print(f"  Severity   min={sev.min():.3f}  max={sev.max():.3f}  mean={sev.mean():.3f}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True,
                        help="Experiment name for the clean baseline "
                             "(e.g. 'clean')")
    parser.add_argument("--experiments", nargs="+", required=True,
                        help="Experiment name(s) containing degradation conditions "
                             "(e.g. 'new_blur1 degrade')")
    parser.add_argument("--dataset", default=None,
                        help="Path to health_monitor_dataset root "
                             "(default: data/health_monitor_dataset)")
    parser.add_argument("--severity-policy", default="ratio",
                        choices=list(SEVERITY_POLICIES.keys()),
                        help="Severity scoring policy (default: ratio)")
    parser.add_argument("--severity-params", nargs="*", default=[],
                        metavar="KEY=VALUE",
                        help="Policy hyperparameters, e.g. alpha=3.0")
    args = parser.parse_args()

    # parse KEY=VALUE pairs into a typed dict
    severity_params = {}
    for kv in args.severity_params:
        k, _, v = kv.partition("=")
        try:
            severity_params[k] = float(v)
        except ValueError:
            severity_params[k] = v

    policy = SeverityPolicy.from_config(args.severity_policy, severity_params)
    print(f"Severity policy : {args.severity_policy}  params={severity_params or '{}'}")

    dataset_root = Path(args.dataset) if args.dataset \
                   else PROJECT_ROOT / "data" / "health_monitor_dataset"

    build_index(dataset_root, args.baseline, args.experiments, policy=policy)
