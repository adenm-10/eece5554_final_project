#!/usr/bin/env python3
"""
Generate noisy TUM-VI datasets by adding Gaussian noise to cam1 (right) images only.
Produces one output dataset per noise level, with cam0/imu0 symlinked from original.

Usage:
    python generate_noisy_tumvi.py \
        --input /path/to/dataset-corridor1_512_16 \
        --output /path/to/noisy_datasets \
        --sigma 5 10 20 40 80
"""

import argparse
import os
import shutil
from pathlib import Path

import cv2
import numpy as np


def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise with given std dev, clipping to [0, 255]."""
    noise = rng.normal(0, sigma, img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def generate_noisy_dataset(input_dir: Path, output_base: Path, sigma: float, seed: int = 42):
    """
    Create a new TUM-VI-structured dataset where:
      - cam0 (left) images are symlinked from the original (untouched)
      - cam1 (right) images have Gaussian noise added
      - imu0 data and timestamp files are symlinked from the original
    """
    tag = f"sigma_{sigma:04.1f}".replace(".", "p")
    out_dir = output_base / f"{input_dir.name}_{tag}"
    mav_in = input_dir / "mav0"
    mav_out = out_dir / "mav0"

    if not mav_in.exists():
        raise FileNotFoundError(f"Expected mav0/ directory in {input_dir}")

    print(f"\n{'='*60}")
    print(f"Generating dataset: sigma={sigma}")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {out_dir}")
    print(f"{'='*60}")

    # --- Symlink cam0 (left, clean) ---
    cam0_out = mav_out / "cam0"
    cam0_out.mkdir(parents=True, exist_ok=True)

    # Symlink the cam0 data directory and csv
    cam0_data_link = cam0_out / "data"
    if not cam0_data_link.exists():
        cam0_data_link.symlink_to((mav_in / "cam0" / "data").resolve())

    cam0_csv_src = mav_in / "cam0" / "data.csv"
    cam0_csv_dst = cam0_out / "data.csv"
    if cam0_csv_src.exists() and not cam0_csv_dst.exists():
        cam0_csv_dst.symlink_to(cam0_csv_src.resolve())

    # --- Symlink imu0 ---
    imu0_out = mav_out / "imu0"
    if not imu0_out.exists():
        imu0_out.symlink_to((mav_in / "imu0").resolve())

    # --- Generate noisy cam1 (right) ---
    cam1_in_data = mav_in / "cam1" / "data"
    cam1_out_data = mav_out / "cam1" / "data"
    cam1_out_data.mkdir(parents=True, exist_ok=True)

    # Copy cam1 csv
    cam1_csv_src = mav_in / "cam1" / "data.csv"
    cam1_csv_dst = mav_out / "cam1" / "data.csv"
    if cam1_csv_src.exists() and not cam1_csv_dst.exists():
        cam1_csv_dst.symlink_to(cam1_csv_src.resolve())

    image_files = sorted(cam1_in_data.glob("*.png"))
    if not image_files:
        raise FileNotFoundError(f"No PNG images found in {cam1_in_data}")

    # Skip if all noisy images already exist
    existing = list(cam1_out_data.glob("*.png"))
    if len(existing) == len(image_files):
        print(f"  Already generated ({len(existing)} images), skipping.")
        return out_dir

    rng = np.random.default_rng(seed)
    total = len(image_files)

    for i, img_path in enumerate(image_files):
        out_path = cam1_out_data / img_path.name
        if out_path.exists():
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"  WARNING: Could not read {img_path}, skipping")
            continue

        noisy_img = add_gaussian_noise(img, sigma, rng)
        cv2.imwrite(str(out_path), noisy_img)

        if (i + 1) % 200 == 0 or (i + 1) == total:
            print(f"  [{i+1}/{total}] images processed")

    print(f"Done: {out_dir}")
    return out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate noisy TUM-VI datasets (noise on cam1 only)"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Path to original TUM-VI sequence (e.g. dataset-corridor1_512_16)")
    parser.add_argument("--output", type=str, required=True,
                        help="Base directory for noisy dataset outputs")
    parser.add_argument("--sigma", type=float, nargs="+", default=[5, 10, 20, 40, 80],
                        help="Gaussian noise std dev values (pixel intensity units, 0-255 scale)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for reproducibility")

    args = parser.parse_args()
    input_dir = Path(args.input).resolve()
    output_base = Path(args.output).resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    generated = []
    for sigma in args.sigma:
        out = generate_noisy_dataset(input_dir, output_base, sigma, seed=args.seed)
        generated.append((sigma, out))

    print(f"\n{'='*60}")
    print("All datasets generated:")
    for sigma, path in generated:
        print(f"  sigma={sigma:5.1f}  ->  {path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()