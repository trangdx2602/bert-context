"""
Script huấn luyện mô hình ERC – Nhóm 9.

Cách dùng:
    python train.py --model bert_context --context_k 3
    python train.py --model bert_context --context_k 3 --accum_steps 4 --batch_size 8
"""
import argparse
import os
import torch
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None

import config
from utils import FocalLoss, set_seed, compute_class_weights, EarlyStopping, save_checkpoint
from data.dataset import get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description="Train ERC model")
    parser.add_argument("--model",       type=str,   default="bert_context",
                        choices=["bert_context"])
    parser.add_argument("--input_mode",  type=str,   default="context",
                        choices=["baseline", "context", "speaker"],
                        help="Cach tao input tu hoi thoai")
    parser.add_argument("--context_k",   type=int,   default=config.CONTEXT_K)
    parser.add_argument("--epochs",      type=int,   default=config.EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max_len",     type=int,   default=config.MAX_LEN)
    parser.add_argument("--dropout",     type=float, default=0.1,
                        help="Dropout cua classifier head")
    parser.add_argument("--loss",        type=str,   default="ce",
                        choices=["ce", "focal"],
                        help="Ham loss de train")
    parser.add_argument("--focal_gamma", type=float, default=2.0,
                        help="Gamma cho focal loss")
    parser.add_argument("--seed",        type=int,   default=config.SEED)
    parser.add_argument("--freeze_bert", action="store_true",
                        help="Freeze BERT, chỉ train linear head")
    parser.add_argument("--accum_steps", type=int,   default=1,
                        help="Gradient accumulation steps (effective batch = batch_size × accum_steps)")
    parser.add_argument("--num_workers", type=int,   default=0,
                        help="DataLoader workers (0 trên Windows, 2-4 trên Linux)")
    parser.add_argument("--no_amp",      action="store_true",
                        help="Tắt Mixed Precision (dùng khi không có GPU hoặc debug)")
    parser.add_argument("--disable_tensorboard", action="store_true",
                        help="Khong ghi log TensorBoard")
    parser.add_argument("--log_dir", type=str, default=None,
                        help="Thu muc chua TensorBoard logs")
    parser.add_argument("--run_name", type=str, default=None,
                        help="Ten run/checkpoint de tranh de len nhau")
    return parser.parse_args()


def build_run_name(args) -> str:
    if args.run_name:
        return args.run_name
    return f"{args.model}_{args.input_mode}_k{args.context_k}"


def load_model(model_name: str, freeze_bert: bool = False, dropout_prob: float = 0.1):
    if model_name == "bert_context":
        from models.bert_context import ContextAwareBERT
        return ContextAwareBERT(freeze_bert=freeze_bert, dropout_prob=dropout_prob)
    raise ValueError(f"Model không hỗ trợ: {model_name}")


def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,
                    scaler, device, accum_steps: int, use_amp: bool):
    model.train()

    total_loss = 0.0
    all_preds, all_labels = [], []
    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels) / accum_steps

        scaler.scale(loss).backward()

        # Cập nhật optimizer mỗi accum_steps batch (hoặc batch cuối cùng)
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
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, f1, acc


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, use_amp: bool):
    model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=use_amp):
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

        total_loss += loss.item()
        all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, f1, acc


def main():
    args = parse_args()
    set_seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")

    eff_batch = args.batch_size * args.accum_steps
    print(f"\n{'='*60}")
    print(f"  Model            : {args.model}")
    print(f"  Input mode       : {args.input_mode}")
    print(f"  Context k        : {args.context_k}")
    print(f"  Device           : {device}")
    print(f"  Mixed Precision  : {'ON' if use_amp else 'OFF'}")
    print(f"  Batch size       : {args.batch_size}  (effective: {eff_batch})")
    print(f"  Accum steps      : {args.accum_steps}")
    print(f"  Max len          : {args.max_len}")
    print(f"  Dropout          : {args.dropout}")
    print(f"  Loss             : {args.loss}")
    print(f"  Epochs           : {args.epochs}")
    print(f"{'='*60}\n")

    run_name = build_run_name(args)
    tb_dir = args.log_dir or os.path.join(config.BASE_DIR, "runs", run_name)
    writer = None
    if not args.disable_tensorboard and SummaryWriter is not None:
        os.makedirs(tb_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tb_dir)
        writer.add_text(
            "run_config",
            "\n".join([
                f"model: {args.model}",
                f"input_mode: {args.input_mode}",
                f"context_k: {args.context_k}",
                f"batch_size: {args.batch_size}",
                f"accum_steps: {args.accum_steps}",
                f"effective_batch: {eff_batch}",
                f"lr: {args.lr}",
                f"epochs: {args.epochs}",
                f"max_len: {args.max_len}",
                f"freeze_bert: {args.freeze_bert}",
                f"amp: {use_amp}",
            ]),
        )
        print(f"[TensorBoard] Logging vao: {tb_dir}\n")
    elif not args.disable_tensorboard:
        print("[TensorBoard] Package chua duoc cai dat, bo qua ghi log.\n")

    print("[1/4] Đang pre-tokenize và load dữ liệu...")
    train_loader, val_loader, train_labels = get_dataloaders(
        mode=args.input_mode,
        context_k=args.context_k,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=args.num_workers,
    )
    print(f"  Train: {len(train_loader.dataset):,} samples")
    print(f"  Val  : {len(val_loader.dataset):,} samples\n")

    class_weights = compute_class_weights(train_labels, config.NUM_LABELS)
    print("[2/4] Class weights:", [f"{w:.3f}" for w in class_weights.tolist()])
    class_weights = class_weights.to(device)
    if args.loss == "focal":
        loss_fn = FocalLoss(gamma=args.focal_gamma, weight=class_weights)
    else:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    print("\n[3/4] Load model...")
    model = load_model(args.model, freeze_bert=args.freeze_bert, dropout_prob=args.dropout)
    model.to(device)

    # Tổng số lần cập nhật optimizer (tính đến gradient accumulation)
    total_updates = (len(train_loader) // args.accum_steps) * args.epochs
    warmup_steps  = int(total_updates * config.WARMUP_RATIO)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
        fused=(device.type == "cuda"),  # fused AdamW nhanh hơn trên CUDA
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_updates,
    )
    scaler = GradScaler(enabled=use_amp)

    early_stopping = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, mode="max")
    ckpt_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"{run_name}_best.pt"
    )

    print("\n[4/4] Bắt đầu training...\n")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")

        tr_loss, tr_f1, tr_acc = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn,
            scaler, device, args.accum_steps, use_amp
        )
        va_loss, va_f1, va_acc = evaluate(
            model, val_loader, loss_fn, device, use_amp
        )

        print(f"  Train → loss: {tr_loss:.4f}  F1: {tr_f1:.4f}  Acc: {tr_acc:.4f}")
        print(f"  Val   → loss: {va_loss:.4f}  F1: {va_f1:.4f}  Acc: {va_acc:.4f}")

        if writer is not None:
            writer.add_scalar("train/loss", tr_loss, epoch)
            writer.add_scalar("train/weighted_f1", tr_f1, epoch)
            writer.add_scalar("train/accuracy", tr_acc, epoch)
            writer.add_scalar("val/loss", va_loss, epoch)
            writer.add_scalar("val/weighted_f1", va_f1, epoch)
            writer.add_scalar("val/accuracy", va_acc, epoch)
            writer.add_scalar("train/learning_rate", scheduler.get_last_lr()[0], epoch)

        is_best = early_stopping.best_value is None or va_f1 > early_stopping.best_value
        if writer is not None:
            writer.add_scalar("val/is_best_checkpoint", int(is_best), epoch)

        if is_best:
            save_checkpoint(model, optimizer, epoch, va_f1, ckpt_path)

        if early_stopping.step(va_f1):
            print(f"\n  Early stopping tại epoch {epoch}.")
            break

    print(f"\nTraining xong! Checkpoint: {ckpt_path}")
    print(f"Best val F1 = {early_stopping.best_value:.4f}\n")
    if writer is not None:
        writer.flush()
        writer.close()


if __name__ == "__main__":
    main()
