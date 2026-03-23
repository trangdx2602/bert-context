import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from collections import defaultdict

import config


def _load_and_group(csv_path: str) -> dict:
    """Đọc CSV và nhóm utterance theo từng hội thoại, sắp xếp theo thứ tự xuất hiện."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"])

    dialogues = defaultdict(list)
    for _, row in df.iterrows():
        dialogues[row["Dialogue_ID"]].append({
            "utterance": str(row["Utterance"]).strip(),
            "speaker":   str(row["Speaker"]).strip(),
            "emotion":   str(row["Emotion"]).strip().lower(),
        })
    return dict(dialogues)


def _build_text(dialogue: list, t: int, mode: str, context_k: int) -> str:
    """Tạo chuỗi input theo mode (baseline / context / speaker)."""
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


class MELDDataset(Dataset):
    """
    Dataset MELD với pre-tokenization.
    Toàn bộ dữ liệu được tokenize 1 lần khi khởi tạo, lưu dưới dạng tensor,
    giúp __getitem__ chỉ cần index mà không xử lý gì thêm.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: BertTokenizer,
        mode: str = "context",
        context_k: int = config.CONTEXT_K,
        max_len: int = config.MAX_LEN,
        label2id: dict = config.LABEL2ID,
    ):
        assert mode in ("baseline", "context", "speaker"), \
            f"mode phải là 'baseline', 'context' hoặc 'speaker', nhận được: {mode}"

        # Thu thập text và nhãn từ tất cả hội thoại
        all_texts  = []
        all_labels = []

        dialogues = _load_and_group(csv_path)
        for dialogue in dialogues.values():
            for t, utt_info in enumerate(dialogue):
                label_str = utt_info["emotion"]
                if label_str not in label2id:
                    continue
                all_texts.append(_build_text(dialogue, t, mode, context_k))
                all_labels.append(label2id[label_str])

        # Tokenize toàn bộ 1 lần (nhanh hơn nhiều so với tokenize trong __getitem__)
        encodings = tokenizer(
            all_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        self.input_ids      = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels         = torch.tensor(all_labels, dtype=torch.long)

    def get_labels(self) -> list:
        return self.labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label":          self.labels[idx],
        }


def get_test_loader(
    mode: str = "context",
    context_k: int = config.CONTEXT_K,
    batch_size: int = config.BATCH_SIZE,
    max_len: int = config.MAX_LEN,
    num_workers: int = 0,
):
    """Chỉ load test set — dùng trong evaluate.py."""
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    print("  Đang pre-tokenize test...", flush=True)
    test_ds = MELDDataset(config.TEST_CSV, tokenizer, mode, context_k, max_len)
    pin = torch.cuda.is_available()
    return DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin,
        persistent_workers=(num_workers > 0),
    )


def get_dataloaders(
    mode: str = "context",
    context_k: int = config.CONTEXT_K,
    batch_size: int = config.BATCH_SIZE,
    max_len: int = config.MAX_LEN,
    num_workers: int = 0,
):
    """
    Load train + val — dùng trong train.py.
    num_workers=0 trên Windows, 2 trên Colab/Linux.
    """
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    print("  Đang pre-tokenize train...", flush=True)
    train_ds = MELDDataset(config.TRAIN_CSV, tokenizer, mode, context_k, max_len)
    print("  Đang pre-tokenize val...",   flush=True)
    val_ds   = MELDDataset(config.VAL_CSV,   tokenizer, mode, context_k, max_len)

    pin = torch.cuda.is_available()

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin,
            persistent_workers=(num_workers > 0),
        )

    train_loader = make_loader(train_ds, shuffle=True)
    val_loader   = make_loader(val_ds,   shuffle=False)

    return train_loader, val_loader, train_ds.get_labels()
