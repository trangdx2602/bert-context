"""
Script huan luyen mo hinh ERC - Nhom 9.
"""
import argparse
import os

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

import config
from data.dataset import get_dataloaders
from utils import EarlyStopping, FocalLoss, compute_class_weights, save_checkpoint, set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train ERC model")
    parser.add_argument("--model", type=str, default="bert_context", choices=["bert_context"])
    parser.add_argument(
        "--input_mode",
        type=str,
        default="context",
        choices=["baseline", "context", "speaker"],
        help="Cach tao input tu hoi thoai",
    )
    parser.add_argument("--context_k", type=int, default=config.CONTEXT_K)
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--lr_head", type=float, default=config.HEAD_LEARNING_RATE)
    parser.add_argument("--max_len", type=int, default=config.MAX_LEN)
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout cua classifier head")
    parser.add_argument("--loss", type=str, default="ce", choices=["ce", "focal"])
    parser.add_argument(
        "--class_weight_mode",
        type=str,
        default="balanced",
        choices=["none", "balanced", "sqrt_inv"],
        help="Cach ap dung class weights",
    )
    parser.add_argument(
        "--train_sampler_mode",
        type=str,
        default="none",
        choices=["none", "balanced", "sqrt_inv"],
        help="Can bang xac suat lay mau trong train loader",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing cho CrossEntropyLoss",
    )
    parser.add_argument("--pooling", type=str, default="cls", choices=["cls", "cls_mean"])
    parser.add_argument("--head_type", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument(
        "--target_prefix",
        type=str,
        default="",
        help="Tien to chen vao utterance dich, vi du 'TARGET: '",
    )
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--disable_tensorboard", action="store_true")
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument(
        "--selection_metric",
        type=str,
        default="macro_f1",
        choices=["weighted_f1", "macro_f1"],
        help="Metric dung de luu checkpoint va early stopping",
    )
    return parser.parse_args()


def build_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    return f"{args.model}_{args.input_mode}_k{args.context_k}"


def load_model(
    model_name: str,
    freeze_bert: bool = False,
    dropout_prob: float = 0.1,
    pooling: str = "cls",
    head_type: str = "linear",
):
    if model_name == "bert_context":
        from models.bert_context import ContextAwareBERT

        return ContextAwareBERT(
            freeze_bert=freeze_bert,
            dropout_prob=dropout_prob,
            pooling=pooling,
            head_type=head_type,
        )
    raise ValueError(f"Model khong ho tro: {model_name}")


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn, scaler, device, accum_steps, use_amp):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps
        all_preds.extend(logits.detach().argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, weighted_f1, macro_f1, acc


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp: bool, num_labels: int):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    per_class_f1 = f1_score(
        all_labels,
        all_preds,
        labels=list(range(num_labels)),
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(num_labels)))
    return avg_loss, weighted_f1, macro_f1, acc, per_class_f1, cm


def summarize_dataset_stats(name: str, stats: dict):
    print(
        f"  {name:<5}: avg_tokens={stats['avg_tokens']:.1f}  "
        f"max_tokens={stats['max_tokens']}  "
        f"truncated={stats['truncated_count']}/{stats['num_samples']} "
        f"({stats['truncated_ratio']:.1%})"
    )


def build_class_weights(labels: list, num_labels: int, mode: str):
    if mode == "none":
        return None
    base = compute_class_weights(labels, num_labels)
    if mode == "balanced":
        return base
    if mode == "sqrt_inv":
        return torch.sqrt(base)
    raise ValueError(f"class_weight_mode khong ho tro: {mode}")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")
    eff_batch = args.batch_size * args.accum_steps

    print(f"\n{'=' * 60}")
    print(f"  Model            : {args.model}")
    print(f"  Input mode       : {args.input_mode}")
    print(f"  Context k        : {args.context_k}")
    print(f"  Device           : {device}")
    print(f"  Mixed Precision  : {'ON' if use_amp else 'OFF'}")
    print(f"  Batch size       : {args.batch_size}  (effective: {eff_batch})")
    print(f"  Accum steps      : {args.accum_steps}")
    print(f"  Max len          : {args.max_len}")
    print(f"  LR BERT          : {args.lr}")
    print(f"  LR Head          : {args.lr_head}")
    print(f"  Dropout          : {args.dropout}")
    print(f"  Loss             : {args.loss}")
    print(f"  Class weights    : {args.class_weight_mode}")
    print(f"  Train sampler    : {args.train_sampler_mode}")
    print(f"  Label smoothing  : {args.label_smoothing}")
    print(f"  Pooling          : {args.pooling}")
    print(f"  Head type        : {args.head_type}")
    print(f"  Target prefix    : {repr(args.target_prefix)}")
    print(f"  Selection metric : {args.selection_metric}")
    print(f"  Epochs           : {args.epochs}")
    print(f"{'=' * 60}\n")

    run_name = build_run_name(args)
    tb_dir = args.log_dir or os.path.join(config.BASE_DIR, "runs", run_name)
    writer = None
    if not args.disable_tensorboard and SummaryWriter is not None:
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text(
            "run_config",
            "\n".join(
                [
                    f"model: {args.model}",
                    f"input_mode: {args.input_mode}",
                    f"context_k: {args.context_k}",
                    f"batch_size: {args.batch_size}",
                    f"accum_steps: {args.accum_steps}",
                    f"effective_batch: {eff_batch}",
                    f"lr: {args.lr}",
                    f"lr_head: {args.lr_head}",
                    f"epochs: {args.epochs}",
                    f"max_len: {args.max_len}",
                    f"freeze_bert: {args.freeze_bert}",
                    f"amp: {use_amp}",
                    f"class_weight_mode: {args.class_weight_mode}",
                    f"train_sampler_mode: {args.train_sampler_mode}",
                    f"label_smoothing: {args.label_smoothing}",
                    f"pooling: {args.pooling}",
                    f"head_type: {args.head_type}",
                    f"target_prefix: {args.target_prefix}",
                    f"selection_metric: {args.selection_metric}",
                ]
            ),
        )
        print(f"[TensorBoard] Logging vao: {tb_dir}\n")
    elif not args.disable_tensorboard:
        print("[TensorBoard] Package chua duoc cai dat, bo qua ghi log.\n")

    print("[1/4] Dang pre-tokenize va load du lieu...")
    train_loader, val_loader, train_labels, train_stats, val_stats = get_dataloaders(
        mode=args.input_mode,
        context_k=args.context_k,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers,
        target_prefix=args.target_prefix,
        train_sampler_mode=args.train_sampler_mode,
    )
    print(f"  Train: {len(train_loader.dataset):,} samples")
    print(f"  Val  : {len(val_loader.dataset):,} samples")
    summarize_dataset_stats("Train", train_stats)
    summarize_dataset_stats("Val", val_stats)

    class_weights = build_class_weights(train_labels, config.NUM_LABELS, args.class_weight_mode)
    if class_weights is not None:
        print("[2/4] Class weights:", [f"{w:.3f}" for w in class_weights.tolist()])
        class_weights = class_weights.to(device)
    else:
        print("[2/4] Class weights: OFF")

    if args.loss == "focal":
        loss_fn = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing,
        )

    print("\n[3/4] Load model...")
    model = load_model(
        args.model,
        freeze_bert=args.freeze_bert,
        dropout_prob=args.dropout,
        pooling=args.pooling,
        head_type=args.head_type,
    )
    model.to(device)

    total_updates = max(1, ((len(train_loader) + args.accum_steps - 1) // args.accum_steps) * args.epochs)
    warmup_steps = int(total_updates * config.WARMUP_RATIO)

    optimizer = AdamW(
        model.get_param_groups(args.lr, args.lr_head),
        weight_decay=config.WEIGHT_DECAY,
        fused=(device.type == "cuda"),
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = GradScaler(enabled=use_amp)

    early_stopping = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, mode="max")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{run_name}_best.pt")

    print("\n[4/4] Bat dau training...\n")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        tr_loss, tr_weighted_f1, tr_macro_f1, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            scaler,
            device,
            args.accum_steps,
            use_amp,
        )
        va_loss, va_weighted_f1, va_macro_f1, va_acc, va_per_class_f1, va_cm = evaluate(
            model,
            val_loader,
            loss_fn,
            device,
            use_amp,
            config.NUM_LABELS,
        )

        print(
            f"  Train -> loss: {tr_loss:.4f}  "
            f"weighted_F1: {tr_weighted_f1:.4f}  macro_F1: {tr_macro_f1:.4f}  Acc: {tr_acc:.4f}"
        )
        print(
            f"  Val   -> loss: {va_loss:.4f}  "
            f"weighted_F1: {va_weighted_f1:.4f}  macro_F1: {va_macro_f1:.4f}  Acc: {va_acc:.4f}"
        )
        print(
            "  Val F1 per class:",
            ", ".join(
                f"{label}={score:.3f}"
                for label, score in zip(config.LABEL_LIST, va_per_class_f1)
            ),
        )

        if writer is not None:
            writer.add_scalar("train/loss", tr_loss, epoch)
            writer.add_scalar("train/weighted_f1", tr_weighted_f1, epoch)
            writer.add_scalar("train/macro_f1", tr_macro_f1, epoch)
            writer.add_scalar("train/accuracy", tr_acc, epoch)
            writer.add_scalar("val/loss", va_loss, epoch)
            writer.add_scalar("val/weighted_f1", va_weighted_f1, epoch)
            writer.add_scalar("val/macro_f1", va_macro_f1, epoch)
            writer.add_scalar("val/accuracy", va_acc, epoch)
            writer.add_scalar("train/learning_rate_bert", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("train/learning_rate_head", optimizer.param_groups[1]["lr"], epoch)
            for idx, label in enumerate(config.LABEL_LIST):
                writer.add_scalar(f"val_f1_per_class/{label}", va_per_class_f1[idx], epoch)

        selection_value = va_macro_f1 if args.selection_metric == "macro_f1" else va_weighted_f1
        is_best = early_stopping.best_value is None or selection_value > early_stopping.best_value
        if is_best:
            save_checkpoint(
                model,
                optimizer,
                epoch,
                selection_value,
                ckpt_path,
                metric_name=args.selection_metric,
                metrics={
                    "val_weighted_f1": va_weighted_f1,
                    "val_macro_f1": va_macro_f1,
                    "val_accuracy": va_acc,
                },
            )

        print("  Val confusion matrix:")
        header = "        " + "  ".join(f"{l[:4]:>5}" for l in config.LABEL_LIST)
        print(header)
        for i, row in enumerate(va_cm):
            label = config.LABEL_LIST[i][:7]
            print(f"{label:<8} " + "  ".join(f"{v:>5}" for v in row))

        stop = early_stopping.step(selection_value)
        if stop:
            print(f"  Early stopping triggered after epoch {epoch}.")
            break

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
