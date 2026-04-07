import pandas as pd
import torch
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

import config


def _load_and_group(csv_path: str) -> dict:
    """Doc CSV va nhom utterance theo tung hoi thoai, sap xep theo thu tu xuat hien."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.sort_values(["Dialogue_ID", "Utterance_ID"])

    dialogues = defaultdict(list)
    for _, row in df.iterrows():
        dialogues[row["Dialogue_ID"]].append(
            {
                "utterance": str(row["Utterance"]).strip(),
                "speaker": str(row["Speaker"]).strip(),
                "emotion": str(row["Emotion"]).strip().lower(),
            }
        )
    return dict(dialogues)


def _build_text(
    dialogue: list,
    t: int,
    mode: str,
    context_k: int,
    target_prefix: str = "",
) -> str:
    """Tao chuoi input theo mode (baseline / context / speaker)."""
    if mode == "baseline":
        return dialogue[t]["utterance"]

    start = max(0, t - context_k)
    if mode == "context":
        parts = [dialogue[i]["utterance"] for i in range(start, t)]
        target = dialogue[t]["utterance"]
        parts.append(f"{target_prefix}{target}" if target_prefix else target)
    else:
        parts = [
            f"{dialogue[i]['speaker']}: {dialogue[i]['utterance']}"
            for i in range(start, t + 1)
        ]
    return " [SEP] ".join(parts)


class MELDDataset(Dataset):
    """
    Dataset MELD voi pre-tokenization.
    Toan bo du lieu duoc tokenize 1 lan khi khoi tao, giup __getitem__ chi con index.
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer: BertTokenizer,
        mode: str = "context",
        context_k: int = config.CONTEXT_K,
        max_len: int = config.MAX_LEN,
        label2id: dict = config.LABEL2ID,
        target_prefix: str = "",
    ):
        assert mode in ("baseline", "context", "speaker"), (
            f"mode phai la 'baseline', 'context' hoac 'speaker', nhan duoc: {mode}"
        )

        all_texts = []
        all_labels = []
        raw_lengths = []

        dialogues = _load_and_group(csv_path)
        for dialogue in dialogues.values():
            for t, utt_info in enumerate(dialogue):
                label_str = utt_info["emotion"]
                if label_str not in label2id:
                    continue
                text = _build_text(
                    dialogue,
                    t,
                    mode,
                    context_k,
                    target_prefix=target_prefix,
                )
                raw_lengths.append(len(tokenizer.tokenize(text)))
                all_texts.append(text)
                all_labels.append(label2id[label_str])

        encodings = tokenizer(
            all_texts,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        truncated_count = sum(length > max_len for length in raw_lengths)
        self.stats = {
            "num_samples": len(all_labels),
            "avg_tokens": (sum(raw_lengths) / len(raw_lengths)) if raw_lengths else 0.0,
            "max_tokens": max(raw_lengths) if raw_lengths else 0,
            "truncated_count": truncated_count,
            "truncated_ratio": (truncated_count / len(raw_lengths)) if raw_lengths else 0.0,
        }

        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.tensor(all_labels, dtype=torch.long)

    def get_labels(self) -> list:
        return self.labels.tolist()

    def get_stats(self) -> dict:
        return self.stats

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "label": self.labels[idx],
        }


def get_test_loader(
    mode: str = "context",
    context_k: int = config.CONTEXT_K,
    batch_size: int = config.BATCH_SIZE,
    max_len: int = config.MAX_LEN,
    num_workers: int = 0,
    target_prefix: str = "",
):
    """Chi load test set, dung trong evaluate.py."""
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)
    print("  Dang pre-tokenize test...", flush=True)
    test_ds = MELDDataset(
        config.TEST_CSV,
        tokenizer,
        mode,
        context_k,
        max_len,
        target_prefix=target_prefix,
    )
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
    target_prefix: str = "",
):
    """
    Load train + val, dung trong train.py.
    num_workers=0 tren Windows, 2 tren Colab/Linux.
    """
    tokenizer = BertTokenizer.from_pretrained(config.BERT_MODEL_NAME)

    print("  Dang pre-tokenize train...", flush=True)
    train_ds = MELDDataset(
        config.TRAIN_CSV,
        tokenizer,
        mode,
        context_k,
        max_len,
        target_prefix=target_prefix,
    )
    print("  Dang pre-tokenize val...", flush=True)
    val_ds = MELDDataset(
        config.VAL_CSV,
        tokenizer,
        mode,
        context_k,
        max_len,
        target_prefix=target_prefix,
    )

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
    val_loader = make_loader(val_ds, shuffle=False)

    return (
        train_loader,
        val_loader,
        train_ds.get_labels(),
        train_ds.get_stats(),
        val_ds.get_stats(),
    )
