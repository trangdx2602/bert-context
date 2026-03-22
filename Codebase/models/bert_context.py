"""
bert_context.py – Context-aware BERT cho bài toán ERC (Nhiệm vụ của Tráng)

Ý tưởng:
    Thay vì chỉ classify câu đơn u_t (như BERT baseline),
    ta ghép k câu trước đó vào input:

        u_{t-k} [SEP] u_{t-k+1} [SEP] ... [SEP] u_t

    Nhờ vậy, BERT có thể học context hội thoại cục bộ để dự đoán
    cảm xúc của u_t chính xác hơn.

Kiến trúc:
    BERT (bert-base-uncased)
        ↓  lấy [CLS] token (768-d)
    Dropout
        ↓
    Linear(768, num_labels)
        ↓
    Logits → CrossEntropyLoss
"""
import torch
import torch.nn as nn
from transformers import BertModel

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class ContextAwareBERT(nn.Module):
    """
    Context-aware BERT: phân loại cảm xúc dựa trên chuỗi context hội thoại.

    Args:
        bert_model_name : tên pretrained model (mặc định bert-base-uncased)
        num_labels      : số nhãn cảm xúc (7 với MELD)
        dropout_prob    : tỉ lệ dropout trước linear layer
        freeze_bert     : True để freeze toàn bộ BERT (chỉ train linear head)
    """

    def __init__(
        self,
        bert_model_name : str   = config.BERT_MODEL_NAME,
        num_labels      : int   = config.NUM_LABELS,
        dropout_prob    : float = 0.1,
        freeze_bert     : bool  = False,
    ):
        super().__init__()

        # Encoder: BERT pretrained
        self.bert = BertModel.from_pretrained(bert_model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768

        # Classifier head
        self.dropout    = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids      : torch.Tensor,  # (B, L)
        attention_mask : torch.Tensor,  # (B, L)
        labels         : torch.Tensor = None,  # (B,) – optional
    ):
        """
        Args:
            input_ids      : token IDs đã padding, shape (batch, seq_len)
            attention_mask : 1 cho token thật, 0 cho padding
            labels         : nhãn ground-truth (nếu có → tính loss luôn)

        Returns:
            logits : (batch, num_labels)
            loss   : CrossEntropyLoss nếu labels không None, else None
        """
        outputs = self.bert(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )

        # Lấy biểu diễn [CLS] token (vị trí 0)
        cls_output = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        cls_output = self.dropout(cls_output)

        logits = self.classifier(cls_output)  # (B, num_labels)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return logits, loss

    def get_param_groups(self, lr_bert: float, lr_head: float):
        """
        Trả về param groups để có thể dùng learning rate khác nhau
        cho BERT và classifier head.

        Ví dụ:
            optimizer = AdamW(model.get_param_groups(lr_bert=2e-5, lr_head=1e-4), ...)
        """
        return [
            {"params": self.bert.parameters(),       "lr": lr_bert},
            {"params": self.classifier.parameters(), "lr": lr_head},
            {"params": self.dropout.parameters(),    "lr": lr_head},
        ]
