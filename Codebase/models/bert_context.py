"""
Context-aware BERT cho bài toán nhận diện cảm xúc trong hội thoại.

Thay vì phân loại từng câu đơn lẻ, mô hình nhận input là chuỗi k câu trước
ghép với câu hiện tại:
    u_{t-k} [SEP] ... [SEP] u_{t-1} [SEP] u_t

BERT sẽ học ngữ cảnh hội thoại cục bộ qua [CLS] token,
sau đó qua Dropout và Linear để dự đoán cảm xúc.
"""
import torch
import torch.nn as nn
from transformers import BertModel

import config


class ContextAwareBERT(nn.Module):

    def __init__(
        self,
        bert_model_name: str = config.BERT_MODEL_NAME,
        num_labels: int = config.NUM_LABELS,
        dropout_prob: float = 0.1,
        freeze_bert: bool = False,
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size  # 768

        self.dropout    = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Lấy biểu diễn [CLS] token (vị trí 0) làm đại diện cho cả chuỗi
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        logits = self.classifier(cls_output)
        return logits

    def get_param_groups(self, lr_bert: float, lr_head: float):
        """Trả về param groups với learning rate riêng cho BERT và classifier head."""
        return [
            {"params": self.bert.parameters(),       "lr": lr_bert},
            {"params": self.classifier.parameters(), "lr": lr_head},
            {"params": self.dropout.parameters(),    "lr": lr_head},
        ]
