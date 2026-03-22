"""
train.py – Script huấn luyện mô hình ERC (Nhóm 9)

Tối ưu tốc độ:
    - Mixed Precision (AMP): tự động dùng float16 trên GPU → 2-3x nhanh hơn
    - Gradient Accumulation: mô phỏng batch lớn hơn mà không cần VRAM thêm
    - optimizer.zero_grad(set_to_none=True): nhanh hơn zero_grad()

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

import config
from utils import set_seed, compute_class_weights, EarlyStopping, save_checkpoint
from data.dataset import get_dataloaders


# ─── Argument Parser ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train ERC model")
    parser.add_argument("--model",       type=str,   default="bert_context",
                        choices=["bert_context"])
    parser.add_argument("--context_k",   type=int,   default=config.CONTEXT_K)
    parser.add_argument("--epochs",      type=int,   default=config.EPOCHS)
    parser.add_argument("--batch_size",  type=int,   default=config.BATCH_SIZE)
    parser.add_argument("--lr",          type=float, default=config.LEARNING_RATE)
    parser.add_argument("--max_len",     type=int,   default=config.MAX_LEN)
    parser.add_argument("--seed",        type=int,   default=config.SEED)
    parser.add_argument("--freeze_bert", action="store_true",
                        help="Freeze BERT weights, chỉ train linear head")
    parser.add_argument("--accum_steps", type=int,   default=1,
                        help="Gradient accumulation steps. "
                             "Effective batch = batch_size × accum_steps")
    parser.add_argument("--num_workers", type=int,   default=0,
                        help="DataLoader workers (0 trên Windows, 2-4 trên Linux)")
    parser.add_argument("--no_amp",      action="store_true",
                        help="Tắt Mixed Precision (AMP), dùng khi không có GPU hoặc debug")
    return parser.parse_args()


# ─── Load Model ───────────────────────────────────────────────────────────────

def load_model(model_name: str, freeze_bert: bool = False):
    if model_name == "bert_context":
        from models.bert_context import ContextAwareBERT
        return ContextAwareBERT(freeze_bert=freeze_bert)
    raise ValueError(f"Model không hỗ trợ: {model_name}")


# ─── One Epoch Train (với AMP + Gradient Accumulation) ───────────────────────

def train_one_epoch(model, loader, optimizer, scheduler, loss_fn,
                    scaler, device, accum_steps: int, use_amp: bool):
    model.train()

    total_loss = 0.0
    all_preds, all_labels = [], []
    optimizer.zero_grad(set_to_none=True)   # nhanh hơn zero_grad()

    for step, batch in enumerate(tqdm(loader, desc="  Train", leave=False)):
        input_ids      = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels         = batch["label"].to(device, non_blocking=True)

        # ── Forward với AMP (float16 trên GPU) ────────────────────────────
        with autocast(device_type=device.type, enabled=use_amp):
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)
            # Chia loss theo accum_steps để gradient scale đúng
            loss = loss / accum_steps

        # ── Backward ──────────────────────────────────────────────────────
        scaler.scale(loss).backward()

        # ── Optimizer step mỗi accum_steps batch ─────────────────────────
        if (step + 1) % accum_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * accum_steps  # khôi phục loss gốc để log
        preds = logits.detach().argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, f1, acc


# ─── Evaluation ───────────────────────────────────────────────────────────────

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
            logits, _ = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(logits, labels)

        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    f1  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, f1, acc


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and (device.type == "cuda")

    eff_batch = args.batch_size * args.accum_steps
    print(f"\n{'='*60}")
    print(f"  Model            : {args.model}")
    print(f"  Context k        : {args.context_k}")
    print(f"  Device           : {device}")
    print(f"  Mixed Precision  : {'ON' if use_amp else 'OFF'}")
    print(f"  Batch size       : {args.batch_size}  (effective: {eff_batch})")
    print(f"  Accum steps      : {args.accum_steps}")
    print(f"  Epochs           : {args.epochs}")
    print(f"{'='*60}\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    print("[1/4] Đang pre-tokenize và load dữ liệu...")
    ds_mode = "context" if "context" in args.model else "baseline"
    train_loader, val_loader, _, train_labels = get_dataloaders(
        mode        = ds_mode,
        context_k   = args.context_k,
        batch_size  = args.batch_size,
        max_len     = args.max_len,
        num_workers = args.num_workers,
    )
    print(f"  Train: {len(train_loader.dataset):,} samples")
    print(f"  Val  : {len(val_loader.dataset):,} samples\n")

    # ── Class weights ─────────────────────────────────────────────────────────
    class_weights = compute_class_weights(train_labels, config.NUM_LABELS)
    print("[2/4] Class weights:", [f"{w:.3f}" for w in class_weights.tolist()])
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[3/4] Load model...")
    model = load_model(args.model, freeze_bert=args.freeze_bert)
    model.to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    # Tổng số lần cập nhật optimizer (sau gradient accumulation)
    total_updates = (len(train_loader) // args.accum_steps) * args.epochs
    warmup_steps  = int(total_updates * config.WARMUP_RATIO)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
        fused=device.type == "cuda",   # fused AdamW: nhanh hơn trên CUDA
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps  = warmup_steps,
        num_training_steps= total_updates,
    )

    # GradScaler cho Mixed Precision (tự động vô hiệu khi use_amp=False)
    scaler = GradScaler(enabled=use_amp)

    early_stopping = EarlyStopping(patience=config.EARLY_STOP_PATIENCE, mode="max")
    ckpt_path = os.path.join(
        config.CHECKPOINT_DIR,
        f"{args.model}_k{args.context_k}_best.pt"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
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

        if early_stopping.best_value is None or va_f1 > early_stopping.best_value:
            save_checkpoint(model, optimizer, epoch, va_f1, ckpt_path)

        if early_stopping.step(va_f1):
            print(f"\n  Early stopping tại epoch {epoch}.")
            break

    print(f"\nTraining xong! Checkpoint: {ckpt_path}")
    print(f"Best val F1 = {early_stopping.best_value:.4f}\n")


if __name__ == "__main__":
    main()
