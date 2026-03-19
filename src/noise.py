import cv2
import numpy as np
from src.config import ORIG_DATASET, NOISY_BASE, SEED


def add_gaussian_noise(img, sigma, rng):
    noise = rng.normal(0, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def generate_noisy_dataset(sigma):
    """Create a noisy copy of cam1. cam0/imu0/mocap0 are symlinked.
    Returns the path to the dataset directory."""
    if sigma == 0:
        return ORIG_DATASET

    tag = f"sigma_{sigma}"
    out_dir = NOISY_BASE / f"room1_{tag}"
    mav_out = out_dir / "mav0"

    # Check if already generated
    cam1_out_data = mav_out / "cam1" / "data"
    cam1_in_data = ORIG_DATASET / "mav0" / "cam1" / "data"
    if cam1_out_data.exists():
        n_existing = len(list(cam1_out_data.glob("*.png")))
        n_original = len(list(cam1_in_data.glob("*.png")))
        if n_existing == n_original:
            print(f"  [sigma={sigma}] Already generated, skipping.")
            return out_dir

    print(f"  [sigma={sigma}] Generating noisy dataset...")

    # Symlink cam0
    cam0_out = mav_out / "cam0"
    cam0_out.mkdir(parents=True, exist_ok=True)
    for item in ["data", "data.csv"]:
        src = (ORIG_DATASET / "mav0" / "cam0" / item).resolve()
        dst = cam0_out / item
        if not dst.exists():
            dst.symlink_to(src)

    # Symlink imu0 and mocap0
    for folder in ["imu0", "mocap0"]:
        dst = mav_out / folder
        if not dst.exists():
            dst.symlink_to((ORIG_DATASET / "mav0" / folder).resolve())

    # Generate noisy cam1
    cam1_out_data.mkdir(parents=True, exist_ok=True)

    cam1_csv_src = ORIG_DATASET / "mav0" / "cam1" / "data.csv"
    cam1_csv_dst = mav_out / "cam1" / "data.csv"
    if cam1_csv_src.exists() and not cam1_csv_dst.exists():
        cam1_csv_dst.symlink_to(cam1_csv_src.resolve())

    rng = np.random.default_rng(SEED)
    images = sorted(cam1_in_data.glob("*.png"))
    for i, img_path in enumerate(images):
        out_path = cam1_out_data / img_path.name
        if out_path.exists():
            continue
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        noisy = add_gaussian_noise(img, sigma, rng)
        cv2.imwrite(str(out_path), noisy)
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(images)}]")

    return out_dir
