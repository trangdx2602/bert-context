# Quá trình cải thiện Context-aware BERT

Ghi lại các thay đổi kỹ thuật đã giúp cải thiện Weighted F1 từ ~0.58–0.59 lên **0.62–0.63** trên tập test MELD.

---

## Kết quả trước và sau

| Giai đoạn | Config | Weighted F1 |
|-----------|--------|-------------|
| Ban đầu | LR=2e-5 (đồng nhất), MAX_LEN=256, batch=16, dropout=0.1, class_weights=balanced | ~0.58–0.59 |
| Sau tối ưu | LR BERT=1e-5, LR head=5e-5, MAX_LEN=128, batch=32, dropout=0.2, label_smoothing=0.1, class_weight=none | **0.62–0.63** |

---

## 1. Giảm MAX_LEN từ 256 xuống 128

**Vấn đề:** Câu trong MELD (trích từ *Friends*) thường ngắn, k=5 câu ghép lại cũng chỉ khoảng 80–100 token. Dùng MAX_LEN=256 lãng phí tính toán vì BERT attention có độ phức tạp O(n²) theo chiều dài chuỗi.

**Thay đổi:** `MAX_LEN: 256 → 128`

**Tác dụng:** Tăng tốc training ~4x, cho phép tăng batch size mà không tốn thêm VRAM.

---

## 2. Tăng Batch Size từ 16 lên 32

**Vấn đề:** Batch nhỏ làm gradient noisy, ước lượng không ổn định.

**Thay đổi:** `BATCH_SIZE: 16 → 32`

**Tác dụng:** Gradient ổn định hơn, GPU T4 được tận dụng tối đa nhờ MAX_LEN đã giảm.

---

## 3. Differential Learning Rate (thay đổi quan trọng nhất)

**Vấn đề:** Dùng cùng một learning rate cho toàn bộ model là không tối ưu — BERT đã được pretrain trên lượng dữ liệu khổng lồ, chỉ cần fine-tune nhẹ. Classifier head thì ngược lại, cần học nhanh hơn vì khởi tạo ngẫu nhiên.

**Thay đổi:**
- BERT layers: `lr = 1e-5`
- Classifier head: `lr_head = 5e-5`

Thực hiện qua `get_param_groups(lr_bert, lr_head)` trong `ContextAwareBERT`, truyền vào `AdamW` thay vì `model.parameters()`.

**Tham khảo:** Kỹ thuật này được ghi nhận trong notebook `it4772_emotionDialogue.ipynb` (thành viên nhóm khác), đạt val F1=0.617 với differential LR.

**Tác dụng:** Tránh việc BERT bị "phá" quá nhiều trong quá trình fine-tune, đồng thời classifier head hội tụ nhanh hơn.

---

## 4. Tắt Class Weights

**Vấn đề:** Class weights balanced (sklearn) phạt nặng các lớp thiểu số như `fear` (50 mẫu) và `disgust` (68 mẫu) — nhưng với BERT đã mạnh, penalty này khiến model thiên lệch quá mức về các lớp nhỏ và giảm F1 tổng thể.

**Thay đổi:** `class_weight_mode: balanced → none`

**Tác dụng:** Model tập trung học tốt hơn ở các lớp phổ biến (`neutral`, `joy`, `surprise`) mà không bị nhiễu bởi các lớp cực hiếm.

---

## 5. Label Smoothing = 0.1

**Vấn đề:** Model bị overfit mạnh — train F1 đạt ~0.86 trong khi val F1 chỉ ~0.59. Mô hình học quá tự tin vào nhãn training.

**Thay đổi:** Thêm `label_smoothing=0.1` vào `CrossEntropyLoss`.

Label smoothing thay nhãn one-hot cứng (0/1) bằng phân phối mềm:
- Nhãn đúng: `1 - 0.1 = 0.9`
- Các nhãn khác: `0.1 / (7 - 1) ≈ 0.017`

**Tác dụng:** Phạt sự quá tự tin, giảm khoảng cách train/val F1, cải thiện generalization.

---

## 6. Tăng Dropout từ 0.1 lên 0.2

**Vấn đề:** Dropout thấp không đủ regularize cho model với hàng triệu tham số.

**Thay đổi:** `dropout_prob: 0.1 → 0.2`

**Tác dụng:** Kết hợp với label smoothing, giảm thêm overfitting.

---

## 7. Điều chỉnh Training Schedule

**Thay đổi:**
- `LEARNING_RATE: 2e-5 → 1e-5` — LR thấp hơn phù hợp với fine-tuning sâu hơn
- `EPOCHS: 5 → 10` — cho model đủ thời gian hội tụ với LR nhỏ
- `EARLY_STOP_PATIENCE: 3 → 4` — tránh dừng quá sớm khi val F1 dao động nhẹ

---

## Tổng hợp các thay đổi

| Thay đổi | Trước | Sau |
|----------|-------|-----|
| MAX_LEN | 256 | 128 |
| Batch size | 16 | 32 |
| LR (BERT) | 2e-5 | 1e-5 |
| LR (head) | 2e-5 | 5e-5 |
| Dropout | 0.1 | 0.2 |
| Label smoothing | không có | 0.1 |
| Class weights | balanced | none |
| Epochs | 5 | 10 |
| Early stop patience | 3 | 4 |

---

## 8. Han che cua phien ban ban dau va cach nhom cai tien tiep

### 8.1. Han che cua phien ban ban dau

Phien ban ban dau cua pipeline tap trung rat manh vao `Accuracy` va `Weighted F1`. Day la hai chi so huu ich, nhung voi bo du lieu MELD thi chua du de phan anh chat luong mo hinh mot cach cong bang, vi nhan `neutral` chiem ti le rat lon trong tap du lieu.

He qua la mo hinh co the dat ket qua tong the kha on chi nho du doan tot lop `neutral`, trong khi cac lop hiem nhu `fear` va `disgust` van rat yeu. Khi do:

- `Accuracy` de bi keo len boi lop da so.
- `Weighted F1` cung co the dep hon thuc te vi F1 cao cua `neutral` dong gop nhieu vao trung binh co trong so.
- Checkpoint "tot nhat" co the van nghieng ve mo hinh hoc tot lop da so, chu khong phai mo hinh can bang nhat giua cac nhan.

Ngoai ra, phien ban ban dau moi chi xu ly mat can bang chu yeu o muc `loss`, thong qua `class weights` hoac `focal loss`. Train loader van lay mau theo kieu shuffle thong thuong, nen trong tung mini-batch, cac mau `neutral` van xuat hien ap dao. Dieu nay lam cho qua trinh hoc de tiep tuc bias ve lop da so.

### 8.2. Cach nhom cai tien tiep

De giai quyet han che tren, nhom da bo sung them mot huong cai tien o cap do train pipeline, nham danh gia mo hinh cong bang hon giua cac nhan:

- Them `Macro F1` vao qua trinh theo doi trong luc train va evaluate.
- Cho phep chon metric dung de luu checkpoint va early stopping thong qua `selection_metric`, trong do co the uu tien `macro_f1` thay vi chi dung `weighted_f1`.
- Bo sung `train_sampler_mode` trong dataloader de can bang xac suat lay mau khi tao batch train, ho tro hai che do `balanced` va `sqrt_inv`.
- Giu lai `classification report`, `per-class F1` va `confusion matrix` de phan tich sau hon thay vi ket luan chi dua tren mot chi so tong the.

Y nghia cua cac cai tien nay la:

- `Macro F1` giup danh gia cong bang hon giua cac nhan vi moi lop dong gop ngang nhau vao chi so trung binh.
- `Balanced sampler` giup cac lop hiem xuat hien deu hon trong qua trinh train, giam nguy co mo hinh chi hoc manh lop `neutral`.
- Viec chon checkpoint theo `macro_f1` giup uu tien mo hinh co kha nang tong quat hoa can bang hon tren toan bo cac nhan cam xuc.

Noi cach khac, nhom khong chi cai tien o muc hyperparameter, ma con cai tien o chinh cach lua chon mo hinh tot nhat va cach to chuc du lieu huan luyen. Day la buoc can thiet de giam tinh trang "weighted F1 dep nhung lop hiem van yeu", dong thoi lam cho viec danh gia ket qua thuyet phuc hon.
