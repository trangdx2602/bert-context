import os
import random
import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight


def set_seed(seed: int = 42):
    """Cố định seed để kết quả tái lặp được."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels: list, num_classes: int) -> torch.Tensor:
    """Tính trọng số từng lớp để xử lý mất cân bằng nhãn."""
    classes = list(range(num_classes))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(classes),
        y=np.array(labels)
    )
    return torch.tensor(weights, dtype=torch.float)


class EarlyStopping:
    """Dừng training sớm nếu val metric không cải thiện sau `patience` epoch."""

    def __init__(self, patience: int = 3, min_delta: float = 0.001, mode: str = "max"):
        self.patience   = patience
        self.min_delta  = min_delta
        self.mode       = mode
        self.best_value = None
        self.counter    = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """Cập nhật trạng thái. Trả về True nếu nên dừng."""
        if self.best_value is None:
            self.best_value = value
            return False

        improved = (
            (value > self.best_value + self.min_delta) if self.mode == "max"
            else (value < self.best_value - self.min_delta)
        )

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


def save_checkpoint(model, optimizer, epoch: int, f1: float, path: str):
    """Lưu checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "f1": f1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)
    print(f"  [Checkpoint saved] epoch={epoch}, val_f1={f1:.4f} → {path}")


def load_checkpoint(model, path: str, optimizer=None, device="cpu"):
    """Load checkpoint, trả về (epoch, f1)."""
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [Checkpoint loaded] epoch={ckpt['epoch']}, val_f1={ckpt['f1']:.4f}")
    return ckpt["epoch"], ckpt["f1"]
