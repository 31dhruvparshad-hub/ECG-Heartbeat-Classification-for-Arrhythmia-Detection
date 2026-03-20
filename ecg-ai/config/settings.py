# settings.py
from pathlib import Path

# Root directory of project
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = BASE_DIR / "dataset"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# MIT-BIH sampling rate
SAMPLING_RATE = 360

# Heartbeat window (in samples)
PRE_R_PEAK = int(0.25 * SAMPLING_RATE)   # 250 ms before R
POST_R_PEAK = int(0.45 * SAMPLING_RATE)  # 450 ms after R
BEAT_LENGTH = PRE_R_PEAK + POST_R_PEAK

# Classes we will use
CLASSES = ["N", "PAC", "PVC", "RBBB", "LBBB"]
NUM_CLASSES = len(CLASSES)

# Training parameters
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001

# Personalization
PERSONALIZATION_SECONDS = 60
FINE_TUNE_EPOCHS = 5
