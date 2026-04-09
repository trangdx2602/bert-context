# Emotion Recognition in Conversation - Nhom 9

Du an nay giai bai toan `Emotion Recognition in Conversation (ERC)` tren dataset MELD. Muc tieu la du doan cam xuc cho tung utterance trong hoi thoai, khong chi dua vao cau hien tai ma con dua vao mot cua so ngu canh truoc do.

Nhanh chinh trong repo la `ContextAwareBERT`: du lieu duoc chuyen tu CSV thanh chuoi co context, dua qua `bert-base-uncased`, sau do phan loai ra 1 trong 7 nhan cam xuc. README nay tong hop lai toan bo du an theo cach don gian, de doc va de hoc.

## Project Goal

Trong bai toan ERC, cung mot cau co the mang cam xuc khac nhau neu dat trong ngu canh khac nhau. Vi du, mot cau ngan nhu `Fine.` co the la `neutral`, nhung cung co the la `sadness` hoac `anger` tuy vao cac cau xuat hien truoc no.

Du an nay muon tra loi cau hoi:

`Neu cho BERT doc them mot vai utterance truoc do, mo hinh co nhan dien cam xuc tot hon so voi chi doc cau hien tai hay khong?`

Mo hinh hien tai tap trung vao 3 kieu dau vao:

- `baseline`: chi dung utterance hien tai
- `context`: dung utterance hien tai va `k` utterance truoc do
- `speaker`: giong `context` nhung chen them ten nguoi noi vao tung cau

## Dataset

Du lieu su dung la MELD, duoc dat trong thu muc `Documents/`:

- `train_sent_emo.csv`
- `val_sent_emo.csv`
- `test_sent_emo.csv`

Code hien tai su dung cac cot chinh:

- `Utterance`
- `Speaker`
- `Emotion`
- `Dialogue_ID`
- `Utterance_ID`

Y nghia:

- moi dong trong CSV la mot utterance
- cac dong co cung `Dialogue_ID` thuoc cung mot hoi thoai
- `Utterance_ID` giup giu dung thu tu cau trong hoi thoai
- `Emotion` la nhan dau ra ma model can du doan

Project hien tai du doan 7 nhan:

- `neutral`
- `surprise`
- `fear`
- `sadness`
- `joy`
- `disgust`
- `anger`

## How The Pipeline Works

Luong du lieu tong quat cua du an la:

```text
CSV
-> group theo dialogue
-> tao built_text cho tung utterance
-> tokenize thanh input_ids va attention_mask
-> DataLoader chia batch
-> ContextAwareBERT du doan logits
-> train.py tinh loss de hoc
-> evaluate.py tinh Accuracy va Weighted F1
```

Noi cach khac:

1. CSV duoc doc bang `pandas`.
2. Du lieu duoc sap xep theo `Dialogue_ID`, `Utterance_ID`.
3. Moi utterance duoc bien thanh 1 sample.
4. Sample do co the chua them context truoc no.
5. `BertTokenizer` bien text thanh token ids.
6. `DataLoader` gom nhieu sample thanh batch.
7. `ContextAwareBERT` sinh ra `logits` cho 7 lop cam xuc.
8. Trong luc train, `loss` duoc tinh tu `logits` va `label` de cap nhat trong so.
9. Trong luc evaluate, model duoc cham tren tap test bang `Accuracy` va `Weighted F1`.

## How One Sample Is Formed

Day la phan quan trong nhat cua project: moi utterance la 1 mau can du doan, nhung text dua vao model co the thay doi tuy theo `input_mode`.

Gia su tai thoi diem `t`, utterance hien tai la `u_t`.

### 1. `baseline`

Chi dung cau hien tai:

```text
u_t
```

Vi du:

```text
My duties? All right.
```

### 2. `context`

Dung `k` utterance truoc va utterance hien tai:

```text
u_{t-k} [SEP] ... [SEP] u_t
```

Vi du voi `k = 2`:

```text
That I did. That I did. [SEP] So let's talk a little bit about your duties. [SEP] My duties? All right.
```

### 3. `speaker`

Giong `context`, nhung chen them speaker vao tung utterance:

```text
Speaker_A: u_{t-k} [SEP] ... [SEP] Speaker_B: u_t
```

Vi du voi `k = 2`:

```text
Chandler: That I did. That I did. [SEP] The Interviewer: So let's talk a little bit about your duties. [SEP] Chandler: My duties? All right.
```

Sau khi co `built_text`, tokenizer se bien no thanh:

- `input_ids`
- `attention_mask`
- `label`

Day la dau vao truc tiep cua model.

## Project Structure

```text
.
|- Codebase/
|  |- config.py
|  |- train.py
|  |- evaluate.py
|  |- utils.py
|  |- requirements.txt
|  |- data/
|  |  \- dataset.py
|  \- models/
|     \- bert_context.py
|- Documents/
|  |- train_sent_emo.csv
|  |- val_sent_emo.csv
|  |- test_sent_emo.csv
|  |- Bao_cao_ContextAwareBERT.md
|  \- IMPROVEMENTS.md
|- ERC_ContextBERT.ipynb
\- README.md
```

Vai tro tung file chinh:

- [config.py](/d:/Code_Python/XLNNTN/Project_Nhom9/Codebase/config.py): chua duong dan du lieu, label mapping, va cac tham so mac dinh.
- [dataset.py](/d:/Code_Python/XLNNTN/Project_Nhom9/Codebase/data/dataset.py): doc CSV, group theo hoi thoai, tao `built_text`, tokenize, va tra ve `DataLoader`.
- [bert_context.py](/d:/Code_Python/XLNNTN/Project_Nhom9/Codebase/models/bert_context.py): dinh nghia `ContextAwareBERT` voi pooling `cls` hoac `cls_mean`, va head `linear` hoac `mlp`.
- [train.py](/d:/Code_Python/XLNNTN/Project_Nhom9/Codebase/train.py): train tren train set, danh gia tren validation set sau moi epoch, luu checkpoint tot nhat, va early stop neu can.
- [evaluate.py](/d:/Code_Python/XLNNTN/Project_Nhom9/Codebase/evaluate.py): load checkpoint tot nhat va danh gia tren test set.
- [utils.py](/d:/Code_Python/XLNNTN/Project_Nhom9/Codebase/utils.py): gom cac ham phu tro nhu `set_seed`, `EarlyStopping`, `FocalLoss`, `save_checkpoint`, `load_checkpoint`.
- [ERC_ContextBERT.ipynb](/d:/Code_Python/XLNNTN/Project_Nhom9/ERC_ContextBERT.ipynb): notebook chinh de train, evaluate, va tong hop ket qua thuc nghiem.

## Installation

```bash
pip install -r Codebase/requirements.txt
```

## Run With Notebook

Notebook chinh:

- [ERC_ContextBERT.ipynb](/d:/Code_Python/XLNNTN/Project_Nhom9/ERC_ContextBERT.ipynb)

Flow notebook:

1. Cai dependencies.
2. Dat 3 file CSV cua MELD vao dung vi tri.
3. Chay cac thuc nghiem voi `context_k = 1, 3, 5`.
4. Evaluate tren test set.
5. Tong hop ket qua bang `Accuracy` va `Weighted F1`.

## Run With CLI

Tat ca cac lenh ben duoi duoc chay trong thu muc `Codebase/`.

### 1. Train

```bash
python train.py --model bert_context --input_mode context --context_k 3 --num_workers 0
```

### 2. Evaluate

```bash
python evaluate.py --model bert_context --input_mode context --context_k 3 --num_workers 0
```

### 3. Mot vai bien the huu ich

Baseline chi dung utterance hien tai:

```bash
python train.py --model bert_context --input_mode baseline --context_k 1 --pooling cls --head_type linear --loss ce --class_weight_mode none --label_smoothing 0.0 --run_name bert_baseline_clean
python evaluate.py --model bert_context --input_mode baseline --context_k 1 --pooling cls --head_type linear --run_name bert_baseline_clean
```

Speaker-aware input:

```bash
python train.py --model bert_context --input_mode speaker --context_k 3 --num_workers 0 --run_name bert_speaker_k3
python evaluate.py --model bert_context --input_mode speaker --context_k 3 --run_name bert_speaker_k3
```

## Important Arguments

### Input / Data

- `--input_mode baseline|context|speaker`: chon cach tao input.
- `--context_k`: so utterance truoc do duoc dua vao context.
- `--max_len`: do dai toi da sau khi tokenize.
- `--target_prefix`: them tien to vao utterance dich trong `context` mode.
- `--num_workers`: so worker cua `DataLoader`.

### Model

- `--model`: hien tai ho tro `bert_context`.
- `--pooling cls|cls_mean`: cach tao vector dai dien cho chuoi.
- `--head_type linear|mlp`: kieu classifier head.
- `--dropout`: dropout cua classifier head.
- `--freeze_bert`: dong bang backbone BERT.

### Training

- `--epochs`: so epoch train.
- `--batch_size`: batch size.
- `--accum_steps`: gradient accumulation.
- `--lr`: learning rate cho BERT backbone.
- `--lr_head`: learning rate cho classifier head.
- `--loss ce|focal`: chon loss function.
- `--focal_gamma`: gamma cua focal loss.
- `--class_weight_mode none|balanced|sqrt_inv`: cach ap class weights.
- `--label_smoothing`: label smoothing cho `CrossEntropyLoss`.
- `--no_amp`: tat mixed precision.

### Logging / Checkpoint

- `--run_name`: dat ten run de luu log va checkpoint.
- `--log_dir`: thu muc TensorBoard.
- `--disable_tensorboard`: tat TensorBoard logging.
- `--checkpoint`: chi dinh checkpoint khi evaluate.

## Results From `ERC_ContextBERT.ipynb`

Bang ket qua tong hop trong notebook [ERC_ContextBERT.ipynb](/d:/Code_Python/XLNNTN/Project_Nhom9/ERC_ContextBERT.ipynb):

| Mo hinh | Context k | Accuracy | Weighted F1 |
|---------|-----------|----------|-------------|
| ContextAwareBERT | `k = 1` | 0.6280 | 0.6140 |
| ContextAwareBERT | `k = 3` | **0.6379** | **0.6221** |
| ContextAwareBERT | `k = 5` | 0.6372 | 0.6206 |

## Main Takeaways

- Ca 3 cau hinh `k = 1, 3, 5` deu vuot moc `Weighted F1 = 0.60`.
- `k = 3` la cau hinh tot nhat trong notebook voi `Accuracy = 0.6379` va `Weighted F1 = 0.6221`.
- `k = 5` rat gan voi `k = 3`, cho thay context dai hon co ich nhung chua vuot ro ret.
- `k = 1` thap hon hai cau hinh con lai, goi y rang viec dua them context hoi thoai giup mo hinh nhan dien cam xuc tot hon.

## Related Documents

- Bao cao: [Bao_cao_ContextAwareBERT.md](/d:/Code_Python/XLNNTN/Project_Nhom9/Documents/Bao_cao_ContextAwareBERT.md)
- Ghi chu cai thien: [IMPROVEMENTS.md](/d:/Code_Python/XLNNTN/Project_Nhom9/Documents/IMPROVEMENTS.md)
- Notebook: [ERC_ContextBERT.ipynb](/d:/Code_Python/XLNNTN/Project_Nhom9/ERC_ContextBERT.ipynb)
