import torch
from torch import Tensor


class ShrinkageLoss(torch.nn.Module):
    def __init__(
        self, a: int = 10, c: float = 0.2, reduction: str = "mean"
    ) -> None:
        super(ShrinkageLoss, self).__init__()
        self.a = a
        self.c = c
        self.reduction = reduction

        self.mse = torch.nn.MSELoss(reduction="none")

    def forward(self, inp: Tensor, target: Tensor) -> Tensor:
        l2 = self.mse(inp, target)

        factor = 1 / (1 + torch.exp(self.a * (self.c - l2)))
        loss = l2 * factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
