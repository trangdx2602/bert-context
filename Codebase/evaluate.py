"""
evaluate.py – Đánh giá mô hình ERC trên tập test (Nhóm 9)

Cách dùng:
    python evaluate.py --model bert_context --context_k 3
    python evaluate.py --model bert_context --context_k 3 --checkpoint checkpoints/bert_context_k3_best.pt
    python evaluate.py --model bert_context --context_k 3 --no_amp   # tắt AMP
"""
import argparse
import torch
from torch.amp import autocast
from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    confusion_matrix,
)
from tqdm import tqdm

import config
from utils import set_seed, load_checkpoint
from data.dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ERC model")
    parser.add_argument("--model",      type=str, default="bert_context",
                        choices=["bert_context"])
    parser.add_argument("--context_k",  type=int, default=config.CONTEXT_K)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max_len",    type=int, default=config.MAX_LEN)
    parser.add_argument("--num_workers",type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--seed",       type=int, default=config.SEED)
    parser.add_argument("--no_amp",     action="store_true",
                        help="Tắt Mixed Precision")
    return parser.parse_args()


def load_model(model_name: str):
    if model_name == "bert_context":
        from models.bert_context import ContextAwareBERT
        return ContextAwareBERT()
    raise ValueError(f"Model không hỗ trợ: {model_name}")


@torch.no_grad()
def predict(model, loader, device, use_amp: bool = False):
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)

        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    return all_labels, all_preds


def main():
    args = parse_args()
    set_seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")
    print(f"\nDevice: {device}  |  AMP: {'ON' if use_amp else 'OFF'}")

    # Data (chỉ cần test loader)
    ds_mode = "context" if "context" in args.model else "baseline"
    _, _, test_loader, _ = get_dataloaders(
        mode        = ds_mode,
        context_k   = args.context_k,
        batch_size  = args.batch_size,
        max_len     = args.max_len,
        num_workers = args.num_workers,
    )
    print(f"Test set: {len(test_loader.dataset):,} samples\n")

    # Checkpoint
    ckpt_path = args.checkpoint or (
        f"{config.CHECKPOINT_DIR}/{args.model}_k{args.context_k}_best.pt"
    )

    # Model
    model = load_model(args.model)
    model.to(device)
    load_checkpoint(model, ckpt_path, device=device)

    # Predict
    all_labels, all_preds = predict(model, test_loader, device, use_amp)

    # Metrics
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"  Model      : {args.model}  (context_k={args.context_k})")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Weighted F1: {wf1:.4f}")
    print(f"{'='*60}\n")

    print("Classification Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=config.LABEL_LIST,
        zero_division=0,
    ))

    # Confusion matrix (optional)
    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix (rows=true, cols=pred):")
    header = "        " + "  ".join(f"{l[:4]:>5}" for l in config.LABEL_LIST)
    print(header)
    for i, row in enumerate(cm):
        label = config.LABEL_LIST[i][:7]
        print(f"{label:<8} " + "  ".join(f"{v:>5}" for v in row))


if __name__ == "__main__":
    main()
