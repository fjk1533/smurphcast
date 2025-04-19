import torch
from smurphcast.losses.bounded import BoundedMSELoss

def test_loss_increases_out_of_range():
    loss = BoundedMSELoss()
    y_true = torch.tensor([[0.2], [0.8]])
    y_pred_good = torch.tensor([[0.25], [0.75]])
    y_pred_bad  = torch.tensor([[-0.2], [1.2]])
    assert loss(y_pred_good, y_true) < loss(y_pred_bad, y_true)
