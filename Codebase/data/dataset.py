"""
dataset.py – Load và xử lý dữ liệu MELD cho bài toán ERC.

Hỗ trợ 3 chế độ input:
    - 'baseline'  : câu đơn u_t
    - 'context'   : u_{t-k} [SEP] ... [SEP] u_t
    - 'speaker'   : "Speaker_A: text [SEP] Speaker_B: text ..."

Tối ưu tốc độ:
    - Tokenize toàn bộ dữ liệu 1 lần khi khởi tạo (pre-tokenize),
      thay vì tokenize từng sample trong __getitem__ (lặp lại mỗi epoch).
    - DataLoader dùng num_workers + pin_memory để load nhanh hơn.
"""
import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from collections import defaultdict

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


# ─── Helper ───────────────────────────────────────────────────────────────────

def _load_and_group(csv_path: str) -> dict:
    """Đọc CSV và nhóm utterance theo Dialogue_ID, sắp xếp theo Utterance_ID."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"])

    dialogues = defaultdict(list)
    for _, row in df.iterrows():
        dialogues[row["Dialogue_ID"]].append({
            "utterance" : str(row["Utterance"]).strip(),
            "speaker"   : str(row["Speaker"]).strip(),
            "emotion"   : str(row["Emotion"]).strip().lower(),
        })
    return dict(dialogues)


def _build_text(dialogue: list, t: int, mode: str, context_k: int) -> str:
    """Tạo chuỗi input text theo mode."""
    if mode == "baseline":
        return dialogue[t]["utterance"]

    start = max(0, t - context_k)
    if mode == "context":
        parts = [dialogue[i]["utterance"] for i in range(start, t + 1)]
    else:  # speaker
        parts = [
            f"{dialogue[i]['speaker']}: {dialogue[i]['utterance']}"
            for i in range(start, t + 1)
        ]
    return " [SEP] ".join(parts)


# ─── Main Dataset Class ───────────────────────────────────────────────────────

class MELDDataset(Dataset):
    """
    Dataset MELD – PRE-TOKENIZED để tránh tokenize lại ở mỗi __getitem__.

    Toàn bộ dữ liệu được tokenize 1 lần khi __init__, sau đó lưu dưới dạng
    tensor, giúp tăng tốc đáng kể khi DataLoader load nhiều epoch.

    Args:
        csv_path   : đường dẫn CSV (train/val/test)
        tokenizer  : BertTokenizer
        mode       : 'baseline' | 'context' | 'speaker'
        context_k  : số câu context trước câu hiện tại
        max_len    : max token length
        label2id   : dict {emotion_str -> int}
    """

    def __init__(
        self,
        csv_path  : str,
        tokenizer : BertTokenizer,
        mode      : str = "context",
        context_k : int = config.CONTEXT_K,
        max_len   : int = config.MAX_LEN,
        label2id  : dict = config.LABEL2ID,
    ):
        assert mode in ("baseline", "context", "speaker"), \
            f"mode phải là 'baseline', 'context' hoặc 'speaker', nhận được: {mode}"

        # ── Bước 1: Thu thập tất cả text + label ────────────────────────────
        all_texts  = []
        all_labels = []

        dialogues = _load_and_group(csv_path)
        for dialogue in dialogues.values():
            for t, utt_info in enumerate(dialogue):
                label_str = utt_info["emotion"]
                if label_str not in label2id:
                    continue
                text = _build_text(dialogue, t, mode, context_k)
                all_texts.append(text)
                all_labels.append(label2id[label_str])

        # ── Bước 2: Batch tokenize toàn bộ 1 lần (nhanh hơn nhiều) ──────────
        encodings = tokenizer(
            all_texts,
            max_length     = max_len,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )

        # Lưu thành tensor để __getitem__ chỉ cần index, không xử lý gì thêm
        self.input_ids      = encodings["input_ids"]       # (N, L)
        self.attention_mask = encodings["attention_mask"]  # (N, L)
        self.labels         = torch.tensor(all_labels, dtype=torch.long)  # (N,)

    def get_labels(self) -> list:
        """Trả về danh sách nhãn (dùng để tính class weights)."""
        return self.labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Chỉ index tensor – cực kỳ nhanh, không cần xử lý gì
        return {
            "input_ids"      : self.input_ids[idx],
            "attention_mask" : self.attention_mask[idx],
            "label"          : self.labels[idx],
        }


# ─── Convenience factory ─────────────────────────────────────────────────────

def get_dataloaders(
    mode        : str = "context",
    context_k   : int = config.CONTEXT_K,
    batch_size  : int = config.BATCH_SIZE,
    max_len     : int = config.MAX_LEN,
    num_workers : int = 0,
):
    """
    Tạo DataLoader cho train / val / test.

    Args:
        num_workers: số CPU worker để load data song song.
                     Windows thường dùng 0 (tránh lỗi multiprocessing).
                     Linux/Mac có thể tăng lên 2-4.

    Returns:
        (train_loader, val_loader, test_loader, train_labels)
    """
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    print("  Đang pre-tokenize train...", flush=True)
    train_ds = MELDDataset(config.TRAIN_CSV, tokenizer, mode, context_k, max_len)
    print("  Đang pre-tokenize val...",   flush=True)
    val_ds   = MELDDataset(config.VAL_CSV,   tokenizer, mode, context_k, max_len)
    print("  Đang pre-tokenize test...",  flush=True)
    test_ds  = MELDDataset(config.TEST_CSV,  tokenizer, mode, context_k, max_len)

    # pin_memory=True: copy data vào pinned memory → nhanh hơn khi transfer sang GPU
    pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size        = batch_size,
        shuffle           = True,
        num_workers       = num_workers,
        pin_memory        = pin,
        persistent_workers= num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size        = batch_size,
        shuffle           = False,
        num_workers       = num_workers,
        pin_memory        = pin,
        persistent_workers= num_workers > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size        = batch_size,
        shuffle           = False,
        num_workers       = num_workers,
        pin_memory        = pin,
        persistent_workers= num_workers > 0,
    )

    return train_loader, val_loader, test_loader, train_ds.get_labels()
