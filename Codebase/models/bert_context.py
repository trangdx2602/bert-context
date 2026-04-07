"""
Context-aware BERT cho bai toan nhan dien cam xuc trong hoi thoai.
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
        pooling: str = "cls_mean",
        head_type: str = "mlp",
    ):
        super().__init__()

        self.bert = BertModel.from_pretrained(bert_model_name)
        self.pooling = pooling
        self.head_type = head_type

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        hidden_size = self.bert.config.hidden_size
        if pooling == "cls":
            pooled_size = hidden_size
        elif pooling == "cls_mean":
            pooled_size = hidden_size * 2
        else:
            raise ValueError(f"Pooling khong ho tro: {pooling}")

        self.dropout = nn.Dropout(dropout_prob)
        if head_type == "linear":
            self.classifier = nn.Linear(pooled_size, num_labels)
        elif head_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(pooled_size, hidden_size),
                nn.GELU(),
                nn.Dropout(dropout_prob),
                nn.Linear(hidden_size, num_labels),
            )
        else:
            raise ValueError(f"Head type khong ho tro: {head_type}")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]

        if self.pooling == "cls":
            pooled_output = cls_output
        else:
            mask = attention_mask.unsqueeze(-1).type_as(outputs.last_hidden_state)
            masked_hidden = outputs.last_hidden_state * mask
            token_count = mask.sum(dim=1).clamp(min=1.0)
            mean_output = masked_hidden.sum(dim=1) / token_count
            pooled_output = torch.cat([cls_output, mean_output], dim=-1)

        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

    def get_param_groups(self, lr_bert: float, lr_head: float):
        """Tra ve param groups voi learning rate rieng cho BERT va classifier head."""
        return [
            {"params": self.bert.parameters(), "lr": lr_bert},
            {"params": self.classifier.parameters(), "lr": lr_head},
            {"params": self.dropout.parameters(), "lr": lr_head},
        ]
