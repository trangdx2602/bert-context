"""
Script danh gia mo hinh ERC tren tap test - Nhom 9.
"""
import argparse

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.amp import autocast
from tqdm import tqdm

import config
from data.dataset import get_test_loader
from utils import load_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate ERC model")
    parser.add_argument("--model", type=str, default="bert_context", choices=["bert_context"])
    parser.add_argument(
        "--input_mode",
        type=str,
        default="context",
        choices=["baseline", "context", "speaker"],
    )
    parser.add_argument("--context_k", type=int, default=config.CONTEXT_K)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--max_len", type=int, default=config.MAX_LEN)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "cls_mean"])
    parser.add_argument("--head_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--target_prefix", type=str, default="")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no_amp", action="store_true", help="Tat Mixed Precision")
    return parser.parse_args()


def build_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    return f"{args.model}_{args.input_mode}_k{args.context_k}"


def load_model(
    model_name: str,
    dropout_prob: float = 0.1,
    pooling: str = "cls",
    head_type: str = "linear",
):
    if model_name == "bert_context":
        from models.bert_context import ContextAwareBERT

        return ContextAwareBERT(
            dropout_prob=dropout_prob,
            pooling=pooling,
            head_type=head_type,
        )
    raise ValueError(f"Model khong ho tro: {model_name}")


@torch.no_grad()
def predict(model, loader, device, use_amp: bool = False):
    model.eval()
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)

        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return all_labels, all_preds


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")
    print(f"\nDevice: {device}  |  AMP: {'ON' if use_amp else 'OFF'}")

    test_loader = get_test_loader(
        mode=args.input_mode,
        context_k=args.context_k,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers,
        target_prefix=args.target_prefix,
    )
    print(f"Test set: {len(test_loader.dataset):,} samples\n")

    run_name = build_run_name(args)
    ckpt_path = args.checkpoint or f"{config.CHECKPOINT_DIR}/{run_name}_best.pt"

    model = load_model(
        args.model,
        dropout_prob=args.dropout,
        pooling=args.pooling,
        head_type=args.head_type,
    )
    model.to(device)
    load_checkpoint(model, ckpt_path, device=device)

    all_labels, all_preds = predict(model, test_loader, device, use_amp)
    wf1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)

    print(f"\n{'=' * 60}")
    print(f"  Model      : {args.model}  (mode={args.input_mode}, context_k={args.context_k})")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Weighted F1: {wf1:.4f}")
    print(f"  Macro F1   : {macro_f1:.4f}")
    print(f"{'=' * 60}\n")

    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=config.LABEL_LIST,
            zero_division=0,
        )
    )

    cm = confusion_matrix(all_labels, all_preds)
    print("Confusion Matrix (rows=true, cols=pred):")
    header = "        " + "  ".join(f"{l[:4]:>5}" for l in config.LABEL_LIST)
    print(header)
    for i, row in enumerate(cm):
        label = config.LABEL_LIST[i][:7]
        print(f"{label:<8} " + "  ".join(f"{v:>5}" for v in row))


if __name__ == "__main__":
    main()
