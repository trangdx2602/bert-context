# Context-aware BERT – ERC Nhóm 9

Bài toán: **Emotion Recognition in Conversation (ERC)** trên dataset **MELD**  
Model: **Context-aware BERT** – ghép k câu trước vào input để học ngữ cảnh hội thoại  
Người thực hiện: **Tráng**

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Cấu trúc thư mục

```
Codebase/
├── config.py              # Cấu hình chung (paths, hyperparameters)
├── utils.py               # Tiện ích: EarlyStopping, class weights, checkpoint
├── requirements.txt
│
├── data/
│   └── dataset.py         # MELDDataset – load CSV, build context window, tokenize
│
├── models/
│   └── bert_context.py    # ContextAwareBERT model
│
├── train.py               # Training loop
├── evaluate.py            # Đánh giá trên test set
└── checkpoints/           # Checkpoint được tạo tự động khi train
```

---

## Cách chạy

### 1. Huấn luyện

```bash
# Mặc định (context k=3)
python train.py --model bert_context --context_k 3

# Tối ưu: Mixed Precision GPU + Gradient Accumulation (effective batch=32)
python train.py --model bert_context --context_k 3 --batch_size 8 --accum_steps 4

# Thử nghiệm k=1,3,5
python train.py --model bert_context --context_k 1
python train.py --model bert_context --context_k 5

# Chạy trên CPU (tắt AMP)
python train.py --model bert_context --context_k 3 --no_amp
```

### 2. Đánh giá

```bash
# Tự động tìm checkpoint theo model+k
python evaluate.py --model bert_context --context_k 3

# Chỉ định checkpoint cụ thể
python evaluate.py --model bert_context --context_k 3 --checkpoint checkpoints/bert_context_k3_best.pt
```

---

## Ý tưởng Context-aware BERT

| Model | Input | Mục tiêu |
|---|---|---|
| BERT Baseline | `u_t` (câu đơn) | Làm mốc so sánh |
| **Context-aware BERT** | `u_{t-k} [SEP] ... [SEP] u_t` | Học ngữ cảnh hội thoại cục bộ |

BERT nhận toàn bộ chuỗi context và dùng embedding `[CLS]` để phân loại cảm xúc của câu cuối `u_t`.

---

## Thí nghiệm gợi ý

| context_k | Ý nghĩa |
|---|---|
| `k=1` | Chỉ câu ngay trước |
| `k=3` | 3 câu trước (mặc định) |
| `k=5` | 5 câu trước |

Kết quả được đánh giá bằng **Weighted F1-score** trên tập test MELD.

---

## Dataset MELD

- **13,000+** utterances, **7 nhãn**: neutral, surprise, fear, sadness, joy, disgust, anger
- Trích từ series *Friends*
- File: `Documents/train_sent_emo.csv`, `val_sent_emo.csv`, `test_sent_emo.csv`
- Cột quan trọng: `Utterance`, `Speaker`, `Emotion`, `Dialogue_ID`, `Utterance_ID`

---

## Tối ưu tốc độ

| Kỹ thuật | Mô tả | Hiệu quả |
|---|---|---|
| **Pre-tokenize** | Tokenize toàn bộ data 1 lần khi khởi tạo | Giảm tải CPU mỗi epoch |
| **Mixed Precision (AMP)** | float16 thay float32 trên GPU | Tăng tốc 2–3x, giảm VRAM |
| **Gradient Accumulation** | `--accum_steps 4` → effective batch ×4 | Batch lớn mà không tốn VRAM |
| **pin_memory** | DataLoader dùng pinned memory | Transfer CPU→GPU nhanh hơn |
| **fused AdamW** | Fused CUDA kernel cho optimizer | Nhanh hơn ~10% trên GPU |
| **non_blocking transfer** | `.to(device, non_blocking=True)` | Overlap data transfer + compute |

> Trên CPU (không có GPU): AMP tự động tắt. Thêm `--no_amp` nếu cần debug.
