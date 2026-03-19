import os
import subprocess
from src.config import (
    ORBSLAM_DIR, VOCAB, STEREO_SETTINGS, MONO_SETTINGS,
    STEREO_BIN, MONO_BIN, TIMESTAMPS, IMU_CSV, TIMEOUT,
)


def get_env():
    """Build environment with LD_LIBRARY_PATH for ORB-SLAM3."""
    env = os.environ.copy()
    lib_paths = [
        str(ORBSLAM_DIR / "lib"),
        str(ORBSLAM_DIR / "Thirdparty" / "DBoW2" / "lib"),
        str(ORBSLAM_DIR / "Thirdparty" / "g2o" / "lib"),
    ]
    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(lib_paths + [existing])
    env["QT_QPA_PLATFORM"] = "offscreen"
    return env


def run_stereo(dataset_dir, tag, run_dir):
    """Run stereo-inertial TUM-VI example."""
    cam0 = dataset_dir / "mav0" / "cam0" / "data"
    cam1 = dataset_dir / "mav0" / "cam1" / "data"

    cmd = [
        str(STEREO_BIN),
        str(VOCAB), str(STEREO_SETTINGS),
        str(cam0), str(cam1),
        str(TIMESTAMPS), str(IMU_CSV),
        tag,
    ]

    print(f"    Running stereo-inertial...")
    with open(run_dir / "log.txt", "w") as log:
        try:
            subprocess.run(
                cmd, cwd=str(run_dir), env=get_env(),
                stdout=log, stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {TIMEOUT}s")


def run_mono(dataset_dir, tag, run_dir):
    """Run mono-inertial TUM-VI example."""
    cam0 = dataset_dir / "mav0" / "cam0" / "data"

    cmd = [
        str(MONO_BIN),
        str(VOCAB), str(MONO_SETTINGS),
        str(cam0),
        str(TIMESTAMPS), str(IMU_CSV),
        tag,
    ]

    print(f"    Running mono-inertial...")
    with open(run_dir / "log.txt", "w") as log:
        try:
            subprocess.run(
                cmd, cwd=str(run_dir), env=get_env(),
                stdout=log, stderr=subprocess.STDOUT,
                timeout=TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {TIMEOUT}s")
