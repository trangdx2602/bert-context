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
        pooled_size = hidden_size * 2  # ket hop [CLS] va mean pooling
        mlp_hidden_size = hidden_size

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(pooled_size, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_hidden_size, num_labels),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        cls_output = outputs.last_hidden_state[:, 0, :]

        # Mean pooling tren cac token hop le de bo sung thong tin cho [CLS].
        mask = attention_mask.unsqueeze(-1).type_as(outputs.last_hidden_state)
        masked_hidden = outputs.last_hidden_state * mask
        token_count = mask.sum(dim=1).clamp(min=1.0)
        mean_output = masked_hidden.sum(dim=1) / token_count

        pooled_output = torch.cat([cls_output, mean_output], dim=-1)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)
        return logits

    def get_param_groups(self, lr_bert: float, lr_head: float):
        """Trả về param groups với learning rate riêng cho BERT và classifier head."""
        return [
            {"params": self.bert.parameters(),       "lr": lr_bert},
            {"params": self.classifier.parameters(), "lr": lr_head},
            {"params": self.dropout.parameters(),    "lr": lr_head},
        ]
