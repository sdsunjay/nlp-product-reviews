"""Centralized configuration for the NLP product reviews project."""

# --- Paths ---
PATH_TO_TRAINING_CSV = "./data/25_07_04_11_18_clean_training.csv"
OUTPUT_MODEL_DIR = "models"
RESULTS_DIR = "./results"

# --- Model ---
MODEL_NAME = "microsoft/deberta-v3-base"
NUM_LABELS = 2

# --- Training hyperparameters ---
NUM_EPOCHS = 8
LEARNING_RATE = 2e-5
PER_DEVICE_TRAIN_BATCH_SIZE = 16
PER_DEVICE_EVAL_BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 1
FP16 = True
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
NUM_EXAMPLES = 1000
RANDOM_SEED = 913
TEST_SIZE = 0.2

# --- AWS ---
S3_BUCKET_NAME = "dsunjay-bucket"
