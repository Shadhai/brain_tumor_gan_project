import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
TRAIN_DIR = os.path.join(RAW_DIR, "Training")
TEST_DIR  = os.path.join(RAW_DIR, "Testing")

PROCESSED_DIR = os.path.join(BASE_DIR, "dataset", "processed")
SYNTHETIC_DIR = os.path.join(BASE_DIR, "dataset", "synthetic")

# Image settings
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
IMG_CHANNELS = 1

# Classes
CLASSES = ["glioma", "meningioma", "pituitary", "notumor"]
NUM_CLASSES = len(CLASSES)
PITUITARY_IDX = CLASSES.index("pituitary")

# GAN settings
LATENT_DIM = 100
EPOCHS_GAN = 2000
LR_G = 0.0001
LR_D = 0.0004

# Imbalance settings
REAL_PITUITARY = 200

# Experiment datasets
SYNTHETIC_COUNTS = {
    "D1": 0,      # baseline
    "D2": 200,    # low GAN
    "D3": 600,    # mid GAN
    "D4": 1000,   # high GAN
    "D5": 600     # traditional augment
}

# Classifier settings
EPOCHS_CLASSIFIER = 50
LR_CLASSIFIER = 1e-4

# Multi‑seed for statistical significance
SEEDS = [42, 123, 999, 2024, 7]

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "experiments")

for d in [OUTPUT_DIR, MODEL_DIR, EXPERIMENT_DIR,
          os.path.join(OUTPUT_DIR, "logs"),
          os.path.join(OUTPUT_DIR, "plots"),
          os.path.join(OUTPUT_DIR, "confusion_matrices"),
          os.path.join(OUTPUT_DIR, "gradcam")]:
    os.makedirs(d, exist_ok=True)

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")