# Context-aware BERT - Nhan dien cam xuc trong hoi thoai

**Bai toan:** Emotion Recognition in Conversation (ERC) tren dataset MELD  
**Mo hinh:** Context-aware BERT - ghep k cau truoc vao input de hoc ngu canh hoi thoai  
**Nguoi thuc hien:** Trang - Nhom 9

---

## Y tuong

Thay vi phan loai tung cau don le nhu BERT baseline, mo hinh nhan input la chuoi k cau truoc ghep voi cau hien tai:

```text
u_{t-k} [SEP] ... [SEP] u_t
```

Repo nay da bo sung cac che do de toi uu muc tieu vuot `0.60` weighted F1 tren test:
- `input_mode=baseline` de tao moc doi chung cong bang voi bai cua Hai
- `pooling=cls` hoac `cls_mean`
- `head_type=linear` hoac `mlp`
- learning rate tach rieng cho BERT va classifier head
- class weights co the tat, dung balanced, hoac dung `sqrt_inv`
- `target_prefix` de danh dau cau dich ma khong dua thong tin speaker

---

## Cai dat

```bash
pip install -r requirements.txt
```

Dat 3 file CSV cua MELD vao thu muc `Documents/`:
- `train_sent_emo.csv`
- `val_sent_emo.csv`
- `test_sent_emo.csv`

---

## Ke hoach chay thu nghiem

### 1. Baseline doi chung

```bash
python train.py --input_mode baseline --context_k 1 --pooling cls --head_type linear --loss ce --class_weight_mode none --label_smoothing 0.0 --run_name bert_baseline_clean
python evaluate.py --input_mode baseline --context_k 1 --pooling cls --head_type linear --run_name bert_baseline_clean
```

### 2. Nhanh chinh de day context-aware BERT vuot 0.60

```bash
python train.py --input_mode context --context_k 1 --pooling cls --head_type linear --loss ce --class_weight_mode none --label_smoothing 0.0 --lr 1e-5 --lr_head 5e-5 --run_name bert_context_k1_main
python evaluate.py --input_mode context --context_k 1 --pooling cls --head_type linear --run_name bert_context_k1_main
```

### 3. Cac bien the nen thu tiep

```bash
# Tang learning rate cho classifier head
python train.py --input_mode context --context_k 1 --pooling cls --head_type linear --loss ce --class_weight_mode none --label_smoothing 0.0 --lr 1e-5 --lr_head 1e-4 --run_name bert_context_k1_head1e4

# So sanh pooling cls_mean
python train.py --input_mode context --context_k 1 --pooling cls_mean --head_type linear --loss ce --class_weight_mode none --label_smoothing 0.0 --lr 1e-5 --lr_head 5e-5 --run_name bert_context_k1_clsmean

# Class weights nhe hon
python train.py --input_mode context --context_k 1 --pooling cls --head_type linear --loss ce --class_weight_mode sqrt_inv --label_smoothing 0.0 --lr 1e-5 --lr_head 5e-5 --run_name bert_context_k1_sqrtinv

# Danh dau cau dich
python train.py --input_mode context --context_k 1 --pooling cls --head_type linear --loss ce --class_weight_mode none --label_smoothing 0.0 --lr 1e-5 --lr_head 5e-5 --target_prefix "TARGET: " --run_name bert_context_k1_target
```

### 4. Cau hinh ho tro

```bash
# Dung gradient accumulation
python train.py --input_mode context --context_k 1 --pooling cls --head_type linear --batch_size 8 --accum_steps 4 --run_name bert_context_k1_accum

# Chay tren CPU
python train.py --input_mode context --context_k 1 --pooling cls --head_type linear --no_amp
```

---

## Ghi chu khi doc log

- `train.py` se in them thong ke tokenization va ty le truncate cua train/val.
- Validation log se co `weighted F1`, `per-class F1`, va `confusion matrix`.
- Checkpoint van duoc luu theo `val weighted F1` tot nhat.

---

## Dataset MELD

- ~13,000 utterances, 7 nhan cam xuc: `neutral, surprise, fear, sadness, joy, disgust, anger`
- Trich tu series *Friends*, co san tap train / val / test
- Cot dung: `Utterance`, `Speaker`, `Emotion`, `Dialogue_ID`, `Utterance_ID`
