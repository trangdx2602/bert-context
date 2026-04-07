# Emotion Recognition in Conversation - Nhom 9

Du an nay giai bai toan `Emotion Recognition in Conversation (ERC)` tren dataset MELD. Nhanh chinh trong repo la `Context-aware BERT`, trong do moi utterance duoc du doan cam xuc dua tren cau hien tai va mot cua so ngu canh truoc do.

## Tong quan

- Bai toan: phan loai cam xuc cho tung utterance trong hoi thoai
- Dataset: MELD
- Backbone: `bert-base-uncased`
- Chi so chinh: `Weighted F1`
- Cach chay: Google Colab notebook hoac CLI trong `Codebase/`

Trong code hien tai, nhanh `Context-aware BERT` da ho tro:
- `input_mode`: `baseline`, `context`, `speaker`
- `pooling`: `cls`, `cls_mean`
- `head_type`: `linear`, `mlp`
- `loss`: `ce`, `focal`
- `class_weight_mode`: `none`, `balanced`, `sqrt_inv`
- `target_prefix`: danh dau utterance dich trong `context` mode
- TensorBoard logging trong luc train

## Cau truc repo

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

## Du lieu

Dat 3 file CSV cua MELD vao thu muc `Documents/`:

- `train_sent_emo.csv`
- `val_sent_emo.csv`
- `test_sent_emo.csv`

Code se doc cac cot chinh:
- `Utterance`
- `Speaker`
- `Emotion`
- `Dialogue_ID`
- `Utterance_ID`

Trong `context` mode, input co dang:

```text
u_{t-k} [SEP] ... [SEP] u_t
```

Trong `speaker` mode, input co dang:

```text
Speaker_A: u_{t-k} [SEP] ... [SEP] Speaker_B: u_t
```

## Cai dat

```bash
pip install -r Codebase/requirements.txt
```

## Chay bang notebook

Notebook chinh:
- [`ERC_ContextBERT.ipynb`](/d:/Code_Python/XLNNTN/Project_Nhom9/ERC_ContextBERT.ipynb)

Flow notebook:
1. Clone repo va cai dependencies
2. Upload 3 file CSV
3. Train baseline `Context-aware BERT` voi `k = 1, 3, 5`
4. Evaluate tren test set
5. Xem TensorBoard
6. Chay them cac cell cai thien

## Chay bang command line

Tat ca lenh ben duoi duoc chay trong thu muc `Codebase/`.

### 1. Train baseline context-aware

```bash
python train.py --model bert_context --input_mode context --context_k 1 --num_workers 0
```

### 2. Evaluate baseline context-aware

```bash
python evaluate.py --model bert_context --input_mode context --context_k 1 --num_workers 0
```

### 3. Cau hinh cai thien da thu

```bash
python train.py --model bert_context --input_mode context --context_k 1 --max_len 160 --batch_size 16 --accum_steps 2 --lr 1.5e-5 --dropout 0.2 --loss focal --focal_gamma 2.0 --num_workers 0 --run_name bert_context_focal_k1_len160
python evaluate.py --model bert_context --input_mode context --context_k 1 --max_len 160 --dropout 0.2 --num_workers 0 --run_name bert_context_focal_k1_len160
```

### 4. Vi du mot baseline doi chung

```bash
python train.py --model bert_context --input_mode baseline --context_k 1 --pooling cls --head_type linear --loss ce --class_weight_mode none --label_smoothing 0.0 --run_name bert_baseline_clean
python evaluate.py --model bert_context --input_mode baseline --context_k 1 --pooling cls --head_type linear --run_name bert_baseline_clean
```

## Cac tham so quan trong

`train.py` dang ho tro:

- `--input_mode baseline|context|speaker`
- `--context_k`
- `--epochs`
- `--batch_size`
- `--lr`
- `--lr_head`
- `--max_len`
- `--dropout`
- `--loss ce|focal`
- `--class_weight_mode none|balanced|sqrt_inv`
- `--label_smoothing`
- `--pooling cls|cls_mean`
- `--head_type linear|mlp`
- `--target_prefix`
- `--freeze_bert`
- `--accum_steps`
- `--disable_tensorboard`
- `--log_dir`
- `--run_name`

`evaluate.py` dang ho tro:

- `--input_mode`
- `--context_k`
- `--max_len`
- `--dropout`
- `--pooling`
- `--head_type`
- `--target_prefix`
- `--checkpoint`
- `--run_name`

## TensorBoard

Neu co cai `tensorboard`, `train.py` se log:
- train/val loss
- train/val weighted F1
- train/val accuracy
- learning rate cho BERT va classifier head
- per-class F1 tren validation

Notebook da co san cell de bat TensorBoard tren Colab.

## Ket qua hien tai

Ket qua baseline `Context-aware BERT` da chay tren Colab:

| Cau hinh | Accuracy | Weighted F1 | Best val F1 |
|---------|----------|-------------|-------------|
| `k = 1` | **0.5870** | **0.5974** | 0.5712 |
| `k = 3` | 0.5862 | 0.5918 | **0.5875** |
| `k = 5` | 0.5816 | 0.5912 | 0.5821 |

Thu nghiem cai thien da chay:

| Cau hinh cai thien | Accuracy | Weighted F1 | Best val F1 |
|---------|----------|-------------|-------------|
| `context + focal + k=1 + max_len=160` | 0.5824 | 0.5898 | 0.5571 |
| `context + focal + k=3 + max_len=160` | 0.5678 | 0.5806 | 0.5695 |

Nhan xet nhanh:
- `k = 1` dang la cau hinh tot nhat tren test
- `k = 3` co `best val F1` cao nhat tren validation
- cac run cai thien bang `focal loss`, `max_len=160` va `dropout=0.2` chua vuot baseline

## Ghi chu ve model hien tai

`Codebase/models/bert_context.py` hien tai da ho tro:
- backbone BERT pretrained
- pooling `cls` hoac `cls_mean`
- classifier head `linear` hoac `mlp`
- learning rate tach rieng cho backbone va head qua `get_param_groups`

Mac dinh trong model class hien tai:
- `pooling="cls_mean"`
- `head_type="mlp"`

Tuy nhien ket qua baseline trong notebook duoc bao cao theo cac cau hinh da chay cu the, khong mac dinh suy ra tu model class.

## Tai lieu lien quan

- Bao cao: [`Bao_cao_ContextAwareBERT.md`](/d:/Code_Python/XLNNTN/Project_Nhom9/Documents/Bao_cao_ContextAwareBERT.md)
- Ghi chu cai thien: [`IMPROVEMENTS.md`](/d:/Code_Python/XLNNTN/Project_Nhom9/Documents/IMPROVEMENTS.md)
- Notebook Colab: [`ERC_ContextBERT.ipynb`](/d:/Code_Python/XLNNTN/Project_Nhom9/ERC_ContextBERT.ipynb)
