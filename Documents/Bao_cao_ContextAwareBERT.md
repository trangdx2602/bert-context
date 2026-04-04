# Bao Cao Mon Hoc: Emotion Recognition in Conversation voi Context-aware BERT

## 1. Gioi thieu de tai

De tai cua nhom tap trung vao bai toan `Emotion Recognition in Conversation (ERC)`, nghia la nhan dien cam xuc cua tung cau thoai trong mot doan hoi thoai.

Khac voi bai toan phan loai van ban thong thuong, ERC kho hon vi:

- Cam xuc cua mot cau phu thuoc rat manh vao nhung cau truoc do.
- Cung mot cau nhung dat trong ngu canh khac nhau co the mang cam xuc khac nhau.
- Hoi thoai co nhieu nguoi noi, nhieu luot trao doi, de gay mo ho neu chi nhin tung cau rieng le.

Vi du, cau "Fine." neu dung mot minh thi rat kho xac dinh cam xuc. Tuy nhien, neu no xuat hien sau mot cuoc cai nhau thi co the mang sac thai `anger` hoac `sadness`; con neu nam sau mot cau noi vui thi co the gan voi `neutral`.

Tu do, muc tieu cua nhom la xay dung mot mo hinh co kha nang su dung ngu canh hoi thoai de cai thien chat luong nhan dien cam xuc.

## 2. Muc tieu cua phuong phap

Phuong phap duoc chon trong de tai la `Context-aware BERT`.

Muc tieu chinh cua phuong phap:

- Du doan nhan cam xuc cho cau thoai hien tai.
- Khong chi dua tren cau hien tai, ma con khai thac them `k` cau truoc de hieu boi canh.
- Kiem tra xem do dai ngu canh co anh huong nhu the nao den ket qua nhan dien cam xuc.

Noi cach khac, de tai muon tra loi cau hoi:

`Neu cho BERT doc them mot vai cau truoc do, mo hinh co nhan dien cam xuc tot hon so voi chi doc mot cau don le hay khong?`

## 3. Du lieu su dung

De tai su dung dataset `MELD` cho bai toan ERC.

### 3.1. Dac diem du lieu

- Du lieu duoc trich tu cac doan hoi thoai trong series `Friends`.
- Moi mau du lieu la mot `utterance` trong hoi thoai.
- Moi utterance duoc gan mot nhan cam xuc.
- Dataset da duoc chia san thanh 3 tap:
  - `train_sent_emo.csv`
  - `val_sent_emo.csv`
  - `test_sent_emo.csv`

### 3.2. Cac cot quan trong

Trong qua trinh xu ly, code su dung cac cot:

- `Dialogue_ID`: ma hoi thoai
- `Utterance_ID`: vi tri cau trong hoi thoai
- `Utterance`: noi dung cau noi
- `Speaker`: nguoi noi
- `Emotion`: nhan cam xuc

### 3.3. Nhan dau ra

Mo hinh du doan 1 trong 7 nhan cam xuc:

- `neutral`
- `surprise`
- `fear`
- `sadness`
- `joy`
- `disgust`
- `anger`

## 4. Dau vao va dau ra cua mo hinh

## 4.1. Dau vao

Trong project nay, dau vao cua mo hinh khong phai chi la cau hien tai `u_t`.

Thay vao do, voi moi thoi diem `t`, he thong tao mot chuoi van ban gom:

- `k` cau truoc do
- Cau hien tai can phan loai

Dang tong quat:

```text
u_{t-k} [SEP] u_{t-k+1} [SEP] ... [SEP] u_t
```

Vi du neu `k = 3`, input vao mo hinh la:

```text
u_{t-3} [SEP] u_{t-2} [SEP] u_{t-1} [SEP] u_t
```

Y nghia:

- `u_t` la cau can du doan cam xuc.
- Cac cau truoc do dong vai tro ngu canh.
- `[SEP]` dung de phan tach cac cau trong chuoi dau vao cho BERT.

Sau khi ghep chuoi, tokenizer cua BERT se bien doi van ban thanh:

- `input_ids`
- `attention_mask`

Day la dau vao truc tiep cua mang BERT.

## 4.2. Dau ra

Dau ra cua mo hinh la mot vector diem `logits` gom 7 gia tri, ung voi 7 lop cam xuc.

Sau khi lay lop co diem cao nhat, mo hinh tra ve 1 nhan cam xuc cuoi cung cho cau hien tai.

Vi vay:

- Dau vao: mot cua so hoi thoai ngan gom cau hien tai va cac cau truoc do
- Dau ra: mot nhan cam xuc cho cau hien tai

## 5. Y tuong cot loi cua phuong phap

Y tuong chinh cua `Context-aware BERT` la:

`Cam xuc cua mot cau trong hoi thoai khong nen duoc xac dinh chi dua tren ban than cau do, ma can duoc xem trong ngu canh vua xay ra truoc no.`

Thay vi phan loai:

```text
u_t -> emotion
```

phuong phap nay thuc hien:

```text
u_{t-k}, ..., u_{t-1}, u_t -> emotion cua u_t
```

Nhu vay, mo hinh co them thong tin de giai quyet tinh mo ho cua ngon ngu hoi thoai.

## 6. Cau truc mo hinh

Mo hinh duoc cai dat trong file `Codebase/models/bert_context.py`.
 
### 6.1. Thanh phan chinh

Mo hinh gom 3 phan:

1. `BERT encoder`
2. `Dropout`
3. `Linear classifier`

### 6.2. Cach mo hinh xu ly

Quy trinh xu ly cua mo hinh nhu sau:

1. Chuoi input da ghep context duoc dua vao `bert-base-uncased`.
2. BERT ma hoa toan bo chuoi thanh cac vector an.
3. Mo hinh lay vector tai token `[CLS]` de dai dien cho toan bo chuoi.
4. Vector nay di qua `Dropout` de giam overfitting.
5. Sau do di qua `Linear layer` de du doan 7 nhan cam xuc.

Co the tom tat bang so do:

```text
Context window text
    -> BERT tokenizer
    -> BERT encoder
    -> [CLS] embedding
    -> Dropout
    -> Linear
    -> Emotion label
```

## 7. Cach tao du lieu cho mo hinh

Phan nay duoc xu ly trong file `Codebase/data/dataset.py`.

### 7.1. Buoc 1: Doc va sap xep du lieu

Chuong trinh doc file CSV, sau do:

- Chuan hoa ten cot
- Sap xep theo `Dialogue_ID` va `Utterance_ID`
- Gom cac utterance theo tung hoi thoai

Muc dich la dam bao thu tu cau noi duoc giu dung theo dong thoi gian hoi thoai.

### 7.2. Buoc 2: Tao context window

Voi moi cau thu `t` trong hoi thoai:

- Neu `mode = baseline`: chi lay cau `u_t`
- Neu `mode = context`: lay tu `u_{t-k}` den `u_t`
- Cac cau duoc noi voi nhau bang token `[SEP]`

Vi vay, moi mau du lieu moi khong con la mot cau don, ma la mot doan context ngan.

### 7.3. Buoc 3: Tokenization

Toan bo text duoc tokenize mot lan ngay khi khoi tao dataset, thay vi tokenize trong tung lan `__getitem__`.

Uu diem:

- Giam thoi gian xu ly trong luc train
- DataLoader nhanh hon
- Toi uu cho viec thuc nghiem nhieu lan

## 8. Qua trinh huan luyen

Phan huan luyen nam trong file `Codebase/train.py`.

### 8.1. Dau vao cho train

Chuong trinh nhan cac tham so quan trong:

- `--model`
- `--context_k`
- `--batch_size`
- `--epochs`
- `--lr`
- `--max_len`
- `--accum_steps`
- `--num_workers`

Trong notebook, nhom thu nghiem voi 3 gia tri:

- `k = 1`
- `k = 3`
- `k = 5`

### 8.2. Cac ky thuat duoc ap dung

Trong huan luyen, project su dung mot so ky thuat ho tro:

- `Class weights`: xu ly mat can bang nhan
- `Mixed Precision (AMP)`: giam bo nho va tang toc tren GPU
- `Gradient accumulation`: tang effective batch size khi GPU co han
- `Gradient clipping`: giup train on dinh hon
- `Early stopping`: dung som neu ket qua validation khong cai thien
- `Checkpoint`: luu mo hinh tot nhat theo `val F1`

### 8.3. Tieu chi danh gia trong train

Sau moi epoch, chuong trinh tinh:

- `loss`
- `accuracy`
- `weighted F1`

Trong do, `weighted F1` duoc dung de theo doi va quyet dinh checkpoint tot nhat.

## 9. Qua trinh danh gia

Phan danh gia nam trong file `Codebase/evaluate.py`.

Sau khi load checkpoint tot nhat, mo hinh se du doan tren tap `test`.

Ket qua in ra gom:

- `Accuracy`
- `Weighted F1`
- `Classification report`
- `Confusion matrix`

Y nghia:

- `Accuracy`: ti le du doan dung tong the
- `Weighted F1`: phu hop hon khi cac lop du lieu khong can bang
- `Classification report`: xem chi tiet precision, recall, F1 cua tung nhan
- `Confusion matrix`: xem mo hinh hay nham giua nhung cam xuc nao

## 10. Khac biet voi cac phuong phap khac

## 10.1. So voi BERT baseline

`BERT baseline` chi dung cau hien tai:

```text
u_t -> emotion
```

Trong khi do `Context-aware BERT` dung them nguc canh:

```text
u_{t-k} [SEP] ... [SEP] u_t -> emotion cua u_t
```

Khac biet chinh:

- Baseline khong co ngu canh hoi thoai
- Context-aware BERT co khai thac ngu canh cuc bo
- Vi vay, Context-aware BERT giam bot tinh mo ho cua cau thoai

Day la diem moi quan trong nhat cua de tai so voi mo hinh co ban.

## 10.2. So voi cac mo hinh RNN/LSTM cho hoi thoai

Mot so phuong phap ERC truyen thong dung:

- RNN
- LSTM
- GRU

Cac mo hinh nay doc hoi thoai theo chuoi thoi gian va luu trang thai an.

So sanh:

- RNN/LSTM mo hinh hoa chuoi hoi thoai theo thu tu thoi gian
- Context-aware BERT cat mot cua so context ngan roi dua vao BERT mot lan

Uu diem cua cach lam trong project:

- Don gian hon
- De cai dat hon
- Tan dung suc manh pretrained BERT

Han che:

- Chi nhin ngu canh cuc bo trong `k` cau truoc
- Khong mo hinh hoa hoi thoai rat dai mot cach truc tiep

## 10.3. So voi cac mo hinh ERC nang cao hon

Trong nghien cuu ERC, co nhieu huong manh hon nhu:

- Theo doi speaker relation
- Graph neural network
- Attention giua cac luot hoi thoai
- Ket hop da phuong thuc: text + audio + video

So voi cac huong nay, project hien tai:

- Chi dung text
- Chua khai thac sau thong tin nguoi noi
- Chua su dung audio/video
- Chua mo hinh hoa quan he phuc tap giua cac utterance

Tuy nhien, doi lai phuong phap:

- Ro rang
- De giai thich
- De huan luyen
- Phu hop voi bai tap mon hoc va thuc nghiem co ban

## 11. Uu diem va han che cua phuong phap

### 11.1. Uu diem

- Y tuong don gian, de trinh bay
- De cai dat trong thuc te
- Tan dung mo hinh pretrained BERT
- Co su dung ngu canh hoi thoai nen tot hon cach chi dung mot cau
- De thuc hien ablation study voi cac gia tri `k`

### 11.2. Han che

- Chi dung text, chua dung audio/video cua MELD
- Chi dung `k` cau truoc, khong xet context xa hon
- Chua khai thac ro vai tro cua `Speaker`
- Dung vector `[CLS]` cua ca doan de du doan cho cau cuoi, nen co the lam tron mat thong tin chi tiet cua cau muc tieu

## 12. Y nghia cua thuc nghiem k = 1, 3, 5

Day la phan thuc nghiem quan trong trong notebook.

Muc tieu cua viec thay doi `context_k` la de kiem tra:

- It ngu canh co du khong
- Nhieu ngu canh hon co giup mo hinh tot hon khong
- Ngu canh qua dai co gay nhieu, loang thong tin hay khong

Y nghia tung gia tri:

- `k = 1`: mo hinh chi nhin 1 cau truoc
- `k = 3`: mo hinh nhin 3 cau truoc
- `k = 5`: mo hinh nhin 5 cau truoc

Neu ket qua `k = 3` tot hon `k = 1`, ta co the ket luan rang ngu canh co ich.
Neu `k = 5` khong tot hon `k = 3`, co the do context da qua dai hoac chua thuc su can thiet.

## 13. Tom tat quy trinh hoat dong cua toan he thong

Co the tom tat toan bo pipeline nhu sau:

1. Doc du lieu MELD tu cac file CSV.
2. Gom utterance theo tung hoi thoai.
3. Voi moi utterance hien tai, lay them `k` utterance truoc do.
4. Noi thanh mot chuoi bang token `[SEP]`.
5. Tokenize chuoi dau vao bang BERT tokenizer.
6. Dua vao `bert-base-uncased`.
7. Lay embedding `[CLS]`.
8. Phan loai qua `Dropout + Linear`.
9. Du doan 1 trong 7 nhan cam xuc.
10. Danh gia bang `Accuracy`, `Weighted F1`, `Classification Report`, `Confusion Matrix`.

## 14. Ket luan

De tai su dung `Context-aware BERT` de giai bai toan nhan dien cam xuc trong hoi thoai. Diem trung tam cua phuong phap la dua them cac cau truoc vao input de mo hinh hieu boi canh hoi thoai, thay vi chi nhin cau hien tai.

Day la mot huong tiep can hop ly cho bai toan ERC vi cam xuc trong hoi thoai mang tinh phu thuoc ngu canh rat cao. So voi BERT baseline, phuong phap nay co tinh thuyet phuc hon ve mat ngon ngu hoc. So voi cac mo hinh ERC nang cao, no don gian hon nhung de trien khai, de giai thich va phu hop voi bai bao cao mon hoc.

Neu can phat trien them trong tuong lai, nhom co the mo rong theo cac huong:

- Them thong tin `Speaker`
- Thu nghiem mo hinh da phuong thuc
- Khai thac context dai han hon
- Su dung cac kien truc ERC chuyen biet hon cho hoi thoai

## 15. Goi y cach trinh bay tren slide

Neu chuyen noi dung nay thanh slide, co the chia thanh 8 slide chinh:

1. Gioi thieu bai toan ERC
2. Dataset MELD va nhan cam xuc
3. Han che cua BERT baseline
4. Y tuong Context-aware BERT
5. Input, output va pipeline
6. Cau truc mo hinh
7. Thuc nghiem voi `k = 1, 3, 5`
8. Uu diem, han che va huong phat trien

## 16. Nguon tham chieu trong project

Noi dung bao cao nay duoc viet dua tren cac thanh phan chinh trong project:

- `Codebase/data/dataset.py`
- `Codebase/models/bert_context.py`
- `Codebase/train.py`
- `Codebase/evaluate.py`
- `Codebase/config.py`
