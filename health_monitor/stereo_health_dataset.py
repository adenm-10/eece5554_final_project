"""
StereoHealthDataset — path-based, degradation on-the-fly.

Reads dataset.csv files from health_monitor_dataset/sequence/traj1/degradations/.
For each 20-frame window:
  - cam0: clean frames loaded from TUM_original/cam0
  - cam1: frames loaded from TUM_original/cam1, degradation applied from config.json
  - severity: precomputed RTE-based score from dataset.csv

Returns:
  cam0     : (T, 1, H, W) float32 tensor  — clean left
  cam1     : (T, 1, H, W) float32 tensor  — degraded right
  severity : scalar float32 tensor
"""

import json
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = PROJECT_ROOT / "data" / "health_monitor_dataset"
CAM0_DIR     = PROJECT_ROOT / "data" / "TUM_original" / "dataset-room1_512_16" / "mav0" / "cam0" / "data"
CAM1_DIR     = PROJECT_ROOT / "data" / "TUM_original" / "dataset-room1_512_16" / "mav0" / "cam1" / "data"
IMG_SIZE     = (224, 224)
WINDOW_SIZE  = 20

# ── Degradation functions (must match src/noise.py behaviour) ─────────────────

def _apply_occlusion(img, frac):
    h, w = img.shape[:2]
    patch_area = int(h * w * frac)
    patch_h    = int(np.sqrt(patch_area * h / w))
    patch_w    = int(patch_area / max(patch_h, 1))
    patch_h, patch_w = min(patch_h, h), min(patch_w, w)
    y = np.random.randint(0, h - patch_h + 1)
    x = np.random.randint(0, w - patch_w + 1)
    out = img.copy()
    out[y:y + patch_h, x:x + patch_w] = 0
    return out


def _apply_config(img, config):
    """
    Apply degradation described by config dict to a grayscale uint8 image.
    Type strings match src/noise.py: 'gaussian_blur', 'gaussian_noise', 'occlusion'.
    'none' or missing type means no degradation.
    """
    t = config.get("type", "none")

    if t in ("none", "clean"):
        return img

    if t == "gaussian_blur":
        ks = int(config["kernel_size"])
        sigma = float(config.get("sigma", 0))
        if ks % 2 == 0:
            ks += 1
        return cv2.GaussianBlur(img, (ks, ks), sigma)

    if t == "gaussian_noise":
        sigma = float(config["sigma"])
        noise = np.random.normal(0, sigma, img.shape)
        return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    if t == "occlusion":
        return _apply_occlusion(img, float(config["frac"]))

    raise ValueError(f"Unknown degradation type: {t!r}")


def _load_image(path):
    """Load grayscale PNG, resize to IMG_SIZE, normalise to [0,1]."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return cv2.resize(img, IMG_SIZE)


# ── Dataset ───────────────────────────────────────────────────────────────────

class StereoHealthDataset(Dataset):
    """
    Args:
        dataset_root : Path to health_monitor_dataset/
        cam0_dir     : Path to clean cam0/data/
        cam1_dir     : Path to clean cam1/data/
        conditions   : list of condition names to include (None = all)
        seed         : RNG seed for stochastic degradations (occlusion, noise)
    """

    def __init__(self, dataset_root=DATASET_ROOT, cam0_dir=CAM0_DIR,
                 cam1_dir=CAM1_DIR, conditions=None, seed=42):
        self.cam0_dir = Path(cam0_dir)
        self.cam1_dir = Path(cam1_dir)
        self.rng      = np.random.default_rng(seed)

        # sorted frame lists (timestamp → path)
        self.cam0_frames = sorted(self.cam0_dir.glob("*.png"))
        self.cam1_frames = sorted(self.cam1_dir.glob("*.png"))
        self._cam0_ts    = {int(f.stem): f for f in self.cam0_frames}
        self._cam1_ts    = {int(f.stem): f for f in self.cam1_frames}
        self._sorted_ts  = sorted(self._cam0_ts.keys())

        degrad_root = Path(dataset_root) / "sequence" / "traj1" / "degradations"

        self.windows = []   # list of (start_ts, severity, config_dict, cond_name)

        import pandas as pd
        for cond_dir in sorted(degrad_root.iterdir()):
            if conditions and cond_dir.name not in conditions:
                continue
            csv_path    = cond_dir / "dataset.csv"
            config_path = cond_dir / "config.json"
            if not csv_path.exists() or not config_path.exists():
                continue

            with open(config_path) as f:
                cfg = json.load(f)

            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                self.windows.append((
                    int(row["window_start_ts"]),
                    float(row["severity"]),
                    cfg,
                    cond_dir.name,
                ))

        if not self.windows:
            raise RuntimeError(
                f"No windows loaded. Run build_dataset_index.py first.\n"
                f"Looked in: {degrad_root}"
            )

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        start_ts, severity, cfg, _ = self.windows[idx]

        # find start index in sorted timestamp list
        try:
            start_idx = self._sorted_ts.index(start_ts)
        except ValueError:
            # nearest match fallback
            arr = np.array(self._sorted_ts)
            start_idx = int(np.argmin(np.abs(arr - start_ts)))

        end_idx = min(start_idx + WINDOW_SIZE, len(self._sorted_ts))
        frame_ts = self._sorted_ts[start_idx:end_idx]

        cam0_imgs = []
        cam1_imgs = []

        for ts in frame_ts:
            img0 = _load_image(self._cam0_ts[ts])
            img1 = _load_image(self._cam1_ts[ts])
            img1 = _apply_config(img1, cfg)

            cam0_imgs.append(img0.astype(np.float32) / 255.0)
            cam1_imgs.append(img1.astype(np.float32) / 255.0)

        cam0 = torch.tensor(np.stack(cam0_imgs)).unsqueeze(1)   # (T, 1, H, W)
        cam1 = torch.tensor(np.stack(cam1_imgs)).unsqueeze(1)

        return cam0, cam1, torch.tensor(severity, dtype=torch.float32)
