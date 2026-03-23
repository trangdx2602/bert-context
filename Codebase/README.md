# Context-aware BERT – Nhận diện cảm xúc trong hội thoại

**Bài toán:** Emotion Recognition in Conversation (ERC) trên dataset MELD
**Mô hình:** Context-aware BERT – ghép k câu trước vào input để học ngữ cảnh hội thoại
**Người thực hiện:** Tráng – Nhóm 9

---

## Ý tưởng

Thay vì phân loại từng câu đơn lẻ như BERT Baseline, mô hình nhận input là chuỗi k câu trước ghép với câu hiện tại:

```
u_{t-k} [SEP] u_{t-k+1} [SEP] ... [SEP] u_t
```

BERT mã hóa toàn bộ chuỗi, lấy embedding `[CLS]` làm đại diện, rồi phân loại cảm xúc của `u_t`.

| Mô hình | Input |
|---------|-------|
| BERT Baseline | `u_t` |
| **Context-aware BERT** | `u_{t-k} [SEP] ... [SEP] u_t` |

---

## Cấu trúc thư mục

```
Codebase/
├── config.py           # Cấu hình chung (paths, hyperparameters)
├── utils.py            # EarlyStopping, class weights, checkpoint
├── train.py            # Training loop
├── evaluate.py         # Đánh giá trên test set
├── requirements.txt
├── data/
│   └── dataset.py      # MELDDataset – build context window, tokenize
├── models/
│   └── bert_context.py # ContextAwareBERT
└── checkpoints/        # Tự động tạo khi train
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

Đặt 3 file CSV của MELD vào thư mục `Documents/` (cùng cấp với `Codebase/`):
- `train_sent_emo.csv`
- `val_sent_emo.csv`
- `test_sent_emo.csv`

---

## Chạy thực nghiệm

### Huấn luyện

```bash
# k=3 (mặc định)
python train.py --model bert_context --context_k 3

# Dùng gradient accumulation (effective batch = 32)
python train.py --model bert_context --context_k 3 --batch_size 8 --accum_steps 4

# Thử nghiệm k=1, k=5
python train.py --model bert_context --context_k 1
python train.py --model bert_context --context_k 5

# Chạy trên CPU
python train.py --model bert_context --context_k 3 --no_amp
```

### Đánh giá

```bash
python evaluate.py --model bert_context --context_k 3

# Chỉ định checkpoint cụ thể
python evaluate.py --model bert_context --context_k 3 --checkpoint checkpoints/bert_context_k3_best.pt
```

### Chạy trên Google Colab

Mở file `ERC_ContextBERT_Experiments.ipynb` ở thư mục gốc, bật GPU T4 và chạy tuần tự từng cell.

---

## Kết quả thực nghiệm

*(Điền sau khi chạy xong)*

| Mô hình | Context k | Accuracy | Weighted F1 |
|---------|-----------|----------|-------------|
| BERT Baseline | – | | |
| Context-aware BERT | k=1 | | |
| Context-aware BERT | k=3 | | |
| Context-aware BERT | k=5 | | |

---

## Dataset MELD

- ~13,000 utterances, 7 nhãn cảm xúc: `neutral, surprise, fear, sadness, joy, disgust, anger`
- Trích từ series *Friends*, có sẵn tập train / val / test
- Cột dùng: `Utterance`, `Speaker`, `Emotion`, `Dialogue_ID`, `Utterance_ID`
