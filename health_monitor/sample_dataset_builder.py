"""
Build a synthetic health monitor dataset from one TUM-VI sequence.

Strategy:
  - 1000 windows with ks in [0, 9]   → severity < 0.5  (stereo healthy)
  - 1000 windows with ks in [11, 19] → severity >= 0.5 (stereo degraded)
  - severity is linearly interpolated from known ATE anchor points
  - blur is applied to all 10 frames in a window together (same ks)
  - windows are shuffled but internal frame order is preserved
"""

import cv2
import numpy as np
import torch
import pickle
from pathlib import Path
from torch.utils.data import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
CAM0_DIR   = Path("datasets/TUM-VI/dataset-room1_512_16/mav0/cam0_eq/data")
CAM1_DIR   = Path("datasets/TUM-VI/dataset-room1_512_16/mav0/cam1_eq/data")
OUT_PATH   = Path("datasets/health_monitor_dataset.pkl")

WINDOW_SIZE      = 10
N_HEALTHY        = 1000   # windows with ks in [0, 9]
N_DEGRADED       = 1000   # windows with ks in [11, 19]
IMG_SIZE         = (224, 224)
SEED             = 42

# ── ATE anchor points (from your real experiments) ────────────────────────────
# format: ks → (ate_stereo, ate_mono)
# ks=0  : clean run
# ks=7  : from your blur experiment
# ks=11+: tracking failure → ate_stereo treated as very large
ATE_ANCHORS = {
    0:  (0.007, 0.012),
    7:  (0.025, 0.012),
    11: (None,  0.012),   # None = tracking failure
}
ATE_MONO_BASELINE = 0.012   # mono is stable across blur levels
ATE_FAILURE_VALUE = 0.500   # assigned ate_stereo when tracking fails

def severity_from_ate(ate_stereo, ate_mono):
    return float(np.clip(ate_stereo / (ate_stereo + ate_mono), 0.0, 1.0))

def build_severity_lut():
    """
    Build a lookup table: ks_value (int) → severity (float)
    using linear interpolation between anchor points.
    Failure (ks >= 11) gets severity = 1.0
    """
    # known severities at anchor ks values
    ks_known  = []
    sev_known = []

    for ks, (ate_s, ate_m) in sorted(ATE_ANCHORS.items()):
        ks_known.append(ks)
        if ate_s is None:
            sev_known.append(1.0)
        else:
            sev_known.append(severity_from_ate(ate_s, ate_m))

    # interpolate for every integer ks from 0 to 19
    lut = {}
    for ks in range(20):
        if ks >= 11:
            lut[ks] = 1.0
        else:
            lut[ks] = float(np.interp(ks, ks_known, sev_known))

    return lut

# ── Image utilities ───────────────────────────────────────────────────────────
def apply_blur(img, ks):
    """Apply Gaussian blur with kernel size ks. ks=0 means no blur."""
    if ks == 0:
        return img
    k = ks if ks % 2 == 1 else ks + 1
    return cv2.GaussianBlur(img, (k, k), sigmaX=0)   # sigma=0 → auto from k

def load_image(path):
    """Load grayscale image, resize, normalise to [0,1], return (1,H,W) tensor."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load {path}")
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return torch.tensor(img).unsqueeze(0)   # (1, H, W)

# ── Window sampling ───────────────────────────────────────────────────────────
def sample_windows(cam0_paths, cam1_paths, ks_values, n_windows, severity_lut, rng):
    """
    Sample n_windows windows.
    Each window:
      - picks a random starting frame from the sequence
      - picks a random ks from ks_values
      - applies that ks blur to all 10 cam1 frames in the window
      - cam0 stays clean
    Returns list of dicts: {cam0, cam1, severity, ks}
    """
    n_frames = min(len(cam0_paths), len(cam1_paths))
    max_start = n_frames - WINDOW_SIZE

    if max_start <= 0:
        raise ValueError(f"Sequence too short: {n_frames} frames, need {WINDOW_SIZE}")

    windows = []
    for _ in range(n_windows):
        start = rng.integers(0, max_start)
        ks    = int(rng.choice(ks_values))

        cam0_frames = []
        cam1_frames = []

        for i in range(start, start + WINDOW_SIZE):
            # cam0: always clean
            img0 = cv2.imread(str(cam0_paths[i]), cv2.IMREAD_GRAYSCALE)
            img0 = cv2.resize(img0, IMG_SIZE)

            # cam1: blurred with this window's ks
            img1 = cv2.imread(str(cam1_paths[i]), cv2.IMREAD_GRAYSCALE)
            img1 = cv2.resize(img1, IMG_SIZE)
            img1 = apply_blur(img1, ks)

            cam0_frames.append(img0.astype(np.float32) / 255.0)
            cam1_frames.append(img1.astype(np.float32) / 255.0)

        windows.append({
            "cam0":     np.stack(cam0_frames),   # (10, H, W)
            "cam1":     np.stack(cam1_frames),   # (10, H, W)
            "severity": severity_lut[ks],
            "ks":       ks,
        })

    return windows

# ── Dataset class ─────────────────────────────────────────────────────────────
class StereoHealthDataset(Dataset):
    """
    Each item:
      cam0     : (10, 1, H, W) tensor — clean left frames
      cam1     : (10, 1, H, W) tensor — blurred right frames
      severity : scalar tensor in [0, 1]
    """
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        w = self.windows[idx]
        cam0 = torch.tensor(w["cam0"]).unsqueeze(1)        # (10, 1, H, W)
        cam1 = torch.tensor(w["cam1"]).unsqueeze(1)        # (10, 1, H, W)
        severity = torch.tensor(w["severity"], dtype=torch.float32)
        return cam0, cam1, severity

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)

    # build severity lookup table
    severity_lut = build_severity_lut()
    print("Severity LUT:")
    for ks in sorted(severity_lut):
        flag = "DEGRADED" if severity_lut[ks] >= 0.5 else "healthy "
        print(f"  ks={ks:2d}  severity={severity_lut[ks]:.3f}  [{flag}]")

    # load image paths
    cam0_paths = sorted(CAM0_DIR.glob("*.png"))
    cam1_paths = sorted(CAM1_DIR.glob("*.png"))

    # diagnostic — print before crashing
    print(f"\nCAM0_DIR: {CAM0_DIR}")
    print(f"  exists : {CAM0_DIR.exists()}")
    print(f"  frames : {len(cam0_paths)}")
    if cam0_paths:
        print(f"  first  : {cam0_paths[0].name}")

    print(f"\nCAM1_DIR: {CAM1_DIR}")
    print(f"  exists : {CAM1_DIR.exists()}")
    print(f"  frames : {len(cam1_paths)}")
    if cam1_paths:
        print(f"  first  : {cam1_paths[0].name}")

    if len(cam0_paths) == 0 or len(cam1_paths) == 0:
        raise FileNotFoundError(
            "No images found. Check CAM0_DIR and CAM1_DIR at the top of the script.\n"
            f"  CAM0_DIR = {CAM0_DIR}\n"
            f"  CAM1_DIR = {CAM1_DIR}"
        )

    print(f"\nSequence: {len(cam0_paths)} cam0 frames, {len(cam1_paths)} cam1 frames")

    # split ks values by severity threshold (0.5 = crossover point)
    healthy_ks  = [ks for ks, sev in severity_lut.items() if sev <  0.5]
    degraded_ks = [ks for ks, sev in severity_lut.items() if sev >= 0.5]
    print(f"\nHealthy  ks values (severity < 0.5): {sorted(healthy_ks)}")
    print(f"Degraded ks values (severity >= 0.5): {sorted(degraded_ks)}")

    # sample windows
    print(f"\nSampling {N_HEALTHY} healthy windows...")
    healthy_windows = sample_windows(
        cam0_paths, cam1_paths,
        healthy_ks, N_HEALTHY,
        severity_lut, rng
    )

    print(f"Sampling {N_DEGRADED} degraded windows...")
    degraded_windows = sample_windows(
        cam0_paths, cam1_paths,
        degraded_ks, N_DEGRADED,
        severity_lut, rng
    )

    # combine and shuffle — but internal window frame order is preserved
    all_windows = healthy_windows + degraded_windows
    rng.shuffle(all_windows)

    # sanity check
    severities = [w["severity"] for w in all_windows]
    ks_vals    = [w["ks"] for w in all_windows]
    print(f"\nDataset summary:")
    print(f"  Total windows : {len(all_windows)}")
    print(f"  Healthy (<0.5): {sum(s < 0.5 for s in severities)}")
    print(f"  Degraded(>=0.5): {sum(s >= 0.5 for s in severities)}")
    print(f"  Severity min  : {min(severities):.3f}")
    print(f"  Severity max  : {max(severities):.3f}")
    print(f"  Severity mean : {np.mean(severities):.3f}")
    print(f"  KS values seen: {sorted(set(ks_vals))}")

    # save
    dataset = StereoHealthDataset(all_windows)
    with open(OUT_PATH, "wb") as f:
        pickle.dump(all_windows, f)
    print(f"\nSaved to {OUT_PATH}")

    # verify loading
    print("\nVerifying shapes from DataLoader...")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    cam0, cam1, severity = next(iter(loader))
    print(f"  cam0     : {cam0.shape}")       # (8, 10, 1, 224, 224)
    print(f"  cam1     : {cam1.shape}")       # (8, 10, 1, 224, 224)
    print(f"  severity : {severity.shape}")   # (8,)
    print(f"  severity values: {severity}")

if __name__ == "__main__":
    main()