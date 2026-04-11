import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "Documents")

TRAIN_CSV = os.path.join(DATA_DIR, "train_sent_emo.csv")
VAL_CSV   = os.path.join(DATA_DIR, "val_sent_emo.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test_sent_emo.csv")

CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

BERT_MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 7
LABEL_LIST = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]
LABEL2ID   = {lab: i for i, lab in enumerate(LABEL_LIST)}
ID2LABEL   = {i: lab for i, lab in enumerate(LABEL_LIST)}

BATCH_SIZE    = 32
MAX_LEN       = 128      # câu MELD ngắn, k=5 ghép lại cũng chỉ ~100 token
LEARNING_RATE = 1e-5
HEAD_LEARNING_RATE = 5e-5
EPOCHS        = 10
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01

CONTEXT_K = 3            # số câu trước dùng làm context

SEED                = 42
DEVICE_STR          = "cuda"
EARLY_STOP_PATIENCE = 4
