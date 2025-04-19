import torch
import torch.nn.functional as F

class BoundedMSELoss(torch.nn.Module):
    """
    Soft‑penalised MSE that discourages predictions outside (min,max).
    """

    def __init__(self, min_val: float = 0.0, max_val: float = 1.0, penalty: float = 10.0):
        super().__init__()
        self.register_buffer("min_val", torch.tensor(min_val))
        self.register_buffer("max_val", torch.tensor(max_val))
        self.penalty = penalty

    def forward(self, preds, target):
        mse = F.mse_loss(preds, target)
        over  = F.relu(preds - self.max_val)
        under = F.relu(self.min_val - preds)
        return mse + self.penalty * torch.mean(over**2 + under**2)
