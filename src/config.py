from pathlib import Path

# All paths are relative to the project root (one level above src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

ORBSLAM_DIR = PROJECT_ROOT / "ORB_SLAM3"
VOCAB = ORBSLAM_DIR / "Vocabulary" / "ORBvoc.txt"

STEREO_BIN = ORBSLAM_DIR / "Examples" / "Stereo-Inertial" / "stereo_inertial_tum_vi"
STEREO_SETTINGS = ORBSLAM_DIR / "Examples" / "Stereo-Inertial" / "TUM-VI.yaml"

MONO_BIN = ORBSLAM_DIR / "Examples" / "Monocular-Inertial" / "mono_inertial_tum_vi"
MONO_SETTINGS = ORBSLAM_DIR / "Examples" / "Monocular-Inertial" / "TUM-VI.yaml"

ORIG_DATASET = PROJECT_ROOT / "data" / "TUM_original" / "dataset-room1_512_16"
TIMESTAMPS = PROJECT_ROOT / "data" / "TUM_original" / "TUM_TimeStamps" / "dataset-room1_512.txt"
GT_CSV = ORIG_DATASET / "mav0" / "mocap0" / "data.csv"
IMU_CSV = ORIG_DATASET / "mav0" / "imu0" / "data.csv"

NOISY_BASE = PROJECT_ROOT / "data" / "noisy_datasets"
RESULTS_DIR = PROJECT_ROOT / "results" / "noise_benchmark"

SIGMAS = [0, 5, 10, 20, 40, 80]
TIMEOUT = 600
SEED = 42