"""
config.py – Cấu hình chung cho dự án ERC Nhóm 9
"""
import os

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Documents")

TRAIN_CSV = os.path.join(DATA_DIR, "train_sent_emo.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_sent_emo.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_sent_emo.csv")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

# ─── Model ────────────────────────────────────────────────────────────────────
BERT_MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 7
LABEL_LIST = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
LABEL2ID   = {lab: i for i, lab in enumerate(LABEL_LIST)}
ID2LABEL   = {i: lab for i, lab in enumerate(LABEL_LIST)}

# ─── Hyperparameters ──────────────────────────────────────────────────────────
BATCH_SIZE   = 16
MAX_LEN      = 256      # max token length sau khi ghép context
LEARNING_RATE = 2e-5
EPOCHS        = 10
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01

# Context window (k câu trước) – dùng cho bert_context & bert_speaker
CONTEXT_K = 3

# ─── Misc ─────────────────────────────────────────────────────────────────────
SEED         = 42
DEVICE_STR   = "cuda"   # "cuda" hoặc "cpu", train.py sẽ fallback tự động
EARLY_STOP_PATIENCE = 3
