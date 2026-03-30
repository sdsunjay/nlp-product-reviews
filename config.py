"""Centralized configuration for the NLP product reviews project."""

# --- Paths ---
PATH_TO_TRAINING_CSV = "./data/clean_training3.csv"
OUTPUT_MODEL_DIR = "models"
RESULTS_DIR = "./results"

# --- Model ---
MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 2

# --- Hardware detection ---
import os
import torch as _torch

HAS_CUDA = _torch.cuda.is_available()
HAS_MPS = hasattr(_torch.backends, "mps") and _torch.backends.mps.is_available()

if HAS_CUDA:
    _gpu_mem_gb = _torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    DEVICE = "cuda"
elif HAS_MPS:
    DEVICE = "mps"
    # Allow MPS to use more shared memory
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
else:
    DEVICE = "cpu"

# --- Training hyperparameters ---
NUM_EPOCHS = 8
LEARNING_RATE = 2e-5
if HAS_CUDA and _gpu_mem_gb >= 32:
    PER_DEVICE_TRAIN_BATCH_SIZE = 32
    PER_DEVICE_EVAL_BATCH_SIZE = 32
    GRADIENT_ACCUMULATION_STEPS = 1
    FP16 = True
elif HAS_CUDA:  # 16GB GPU
    PER_DEVICE_TRAIN_BATCH_SIZE = 8
    PER_DEVICE_EVAL_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4
    FP16 = True
elif HAS_MPS:  # Apple Silicon (shared 16GB RAM)
    PER_DEVICE_TRAIN_BATCH_SIZE = 8
    PER_DEVICE_EVAL_BATCH_SIZE = 8
    GRADIENT_ACCUMULATION_STEPS = 4
    FP16 = False
else:  # CPU only
    PER_DEVICE_TRAIN_BATCH_SIZE = 4
    PER_DEVICE_EVAL_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = 8
    FP16 = False
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500

# --- Class weighting (for recall on minority class) ---
# Weight applied to class-1 loss. Higher = fewer missed safety reviews.
# Set to 0 to auto-compute from training label distribution.
CLASS_1_WEIGHT = 8.0

# --- Tokenization ---
MAX_TOKENS = 256
SAFE_CHAR_LIMIT = 2000

# --- Inference threshold ---
# Predict class 1 when P(class=1) exceeds this value.
# Lower = higher recall, more false positives.
DECISION_THRESHOLD = 0.3

# --- Data ---
NUM_EXAMPLES = 3200
RANDOM_SEED = 913
TEST_SIZE = 0.2

# --- AWS ---
S3_BUCKET_NAME = "dsunjay-bucket"
