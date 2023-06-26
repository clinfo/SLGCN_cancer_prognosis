import torch
from lib.utils.shrinkage_loss import ShrinkageLoss
from torch import Tensor
from typing import Type


class LossPrecissionClipper(torch.nn.Module):
    def __init__(
        self,
        loss_fn: Type[ShrinkageLoss],
        clip: float,
        reduction: str = "mean",
    ) -> None:
        super(LossPrecissionClipper, self).__init__()
        self.reduction = reduction
        self.clip = clip

        self.loss_fn = loss_fn(reduction="none")

    def forward(self, inp: Tensor, target: Tensor) -> Tensor:
        print(inp, target)
        loss = self.loss_fn(inp, target)
        loss = torch.clamp(loss, self.clip)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
