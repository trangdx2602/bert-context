# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Emotion Recognition in Conversation (ERC) trên dataset MELD (13k utterances, 7 nhãn, trích từ *Friends*).
Nhiệm vụ của Tráng: **Context-aware BERT** — ghép k câu trước vào input: `u_{t-k} [SEP] ... [SEP] u_t`.
Target: Weighted F1 > 0.60 trên test set.

## Cách chạy

Tất cả script chạy từ thư mục `Codebase/`:

```bash
cd Codebase

# Train
python train.py --input_mode context --context_k 1 \
  --pooling cls --head_type linear \
  --loss ce --class_weight_mode none --label_smoothing 0.1 \
  --lr 1e-5 --lr_head 5e-5 \
  --batch_size 32 --num_workers 2 \
  --run_name bert_context_k1

# Evaluate
python evaluate.py --input_mode context --context_k 1 \
  --pooling cls --head_type linear \
  --num_workers 2 --run_name bert_context_k1

# Kiểm tra import
python -c "import config; from data.dataset import get_dataloaders; from models.bert_context import ContextAwareBERT; print('OK')"
```

Trên **Windows**: `--num_workers 0`. Trên **Colab/Linux**: `--num_workers 2`.
Tắt AMP (debug/CPU): thêm `--no_amp`.

## Kiến trúc

```
Codebase/
├── config.py            # Tất cả hyperparameter và paths
├── utils.py             # set_seed, EarlyStopping, checkpoint, FocalLoss, compute_class_weights
├── train.py             # Training loop (AMP + gradient accumulation + TensorBoard)
├── evaluate.py          # Evaluate trên test set
├── data/
│   └── dataset.py       # MELDDataset (pre-tokenized), get_dataloaders, get_test_loader
└── models/
    └── bert_context.py  # ContextAwareBERT
```

**Data flow:**
- `Documents/` (ngoài `Codebase/`) chứa 3 CSV: `train_sent_emo.csv`, `val_sent_emo.csv`, `test_sent_emo.csv`
- `MELDDataset` tokenize toàn bộ 1 lần khi init (pre-tokenization), `__getitem__` chỉ index tensor
- `get_dataloaders` trả về `(train_loader, val_loader, labels, train_stats, val_stats)` — 5 giá trị
- `get_test_loader` dùng riêng trong `evaluate.py` để không tokenize train/val thừa

**Model (`ContextAwareBERT`):**
- Hỗ trợ `pooling`: `cls` (chỉ [CLS] token) hoặc `cls_mean` (concat [CLS] + mean pooling)
- Hỗ trợ `head_type`: `linear` (768→7) hoặc `mlp` (768→768→7 với GELU)
- `get_param_groups(lr_bert, lr_head)` trả về param groups để dùng differential LR — **optimizer luôn dùng hàm này**, không dùng `model.parameters()` thẳng

**Training (`train.py`):**
- Differential LR: BERT layers dùng `--lr` (1e-5), classifier head dùng `--lr_head` (5e-5)
- `--class_weight_mode`: `none` | `balanced` (sklearn) | `sqrt_inv` (nhẹ hơn balanced)
- `--label_smoothing`: float, truyền thẳng vào `CrossEntropyLoss`
- Checkpoint tên theo `--run_name`, lưu ở `checkpoints/{run_name}_best.pt`
- Validation in per-class F1 + confusion matrix mỗi epoch

## Hyperparameter hiện tại (config.py)

| Tham số | Giá trị |
|---------|---------|
| `LEARNING_RATE` | 1e-5 (BERT) |
| `HEAD_LEARNING_RATE` | 5e-5 (head) |
| `BATCH_SIZE` | 32 |
| `MAX_LEN` | 128 |
| `EPOCHS` | 10 |
| `EARLY_STOP_PATIENCE` | 4 |
| `CONTEXT_K` | 3 |

## Các biến thể input

| `--input_mode` | Input BERT nhận |
|----------------|-----------------|
| `baseline` | `u_t` đơn |
| `context` | `u_{t-k} [SEP] ... [SEP] u_t` |
| `speaker` | `Speaker: u_{t-k} [SEP] ... [SEP] Speaker: u_t` |

`--target_prefix` (vd: `"TARGET: "`) chen vào đầu câu dích `u_t` trong mode `context`.

## Lưu ý quan trọng

- `FocalLoss` nằm trong `utils.py`, được import trong `train.py`
- `UNEXPECTED keys` khi load BERT là bình thường (các layer `cls.*` của BertForMaskedLM không dùng)
- Checkpoint lưu theo `val weighted F1` tốt nhất, không phải epoch cuối
- `num_workers > 0` trên Windows gây lỗi multiprocessing — luôn dùng 0
