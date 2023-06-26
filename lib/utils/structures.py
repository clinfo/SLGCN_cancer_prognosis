import torch
from ..models.convolution import SampleNetResConv
from ..models.fully_connected import (
    SampleNet,
    SampleNetBatchNorm,
    SampleNetDropout,
    SampleNetFullResidual,
)

from .loss_precision_clip import LossPrecissionClipper
from .shrinkage_loss import ShrinkageLoss
from lib.models.convolution import SampleNetResConv
from lib.models.fully_connected import (
    SampleNet,
    SampleNetBatchNorm,
    SampleNetDropout,
    SampleNetFullResidual,
)
from lib.utils.loss_precision_clip import LossPrecissionClipper
from typing import Optional, Type, Union


class ModelsDictionary:
    _dict = {
        "SampleNet": SampleNet,
        "SampleNetDropout": SampleNetDropout,
        "SampleNetFullResidual": SampleNetFullResidual,
        "SampleNetResConv": SampleNetResConv,
        "SampleNetBatchNorm": SampleNetBatchNorm,
    }

    def __getitem__(
        self, item: str
    ) -> Union[
        Type[SampleNetFullResidual],
        Type[SampleNetBatchNorm],
        Type[SampleNetDropout],
        Type[SampleNet],
        Type[SampleNetResConv],
    ]:
        return self._dict[item]

    @property
    def SampleNet(self):
        return self._dict["SampleNet"]

    @property
    def SampleNetDropout(self):
        return self._dict["SampleNetDropout"]

    @property
    def SampleNetFullResidual(self):
        return self._dict["SampleNetFullResidual"]

    @property
    def SampleNetResConv(self):
        return self._dict["SampleNetResConv"]

    @property
    def SampleNetBatchNorm(self):
        return self._dict["SampleNetBatchNorm"]


class LossDictionary(object):
    _dict = {
        "MSE": torch.nn.MSELoss,
        "Shr": ShrinkageLoss,
        "BCEntropy": torch.nn.BCELoss,
    }

    def __init__(
        self, reduction: str = "sum", precision_clip: Optional[float] = None
    ) -> None:
        self.precision_clip = precision_clip
        self.reduction = reduction

    def __getitem__(self, item: str) -> LossPrecissionClipper:
        if self.precision_clip is not None:
            # The reduction should be sum for the clipping to work better
            criterion = LossPrecissionClipper(
                self._dict[item],
                clip=self.precision_clip,
                reduction="sum",
            )
        else:
            criterion = self._dict[item](reduction=self.reduction)

        return criterion

    @property
    def MSE(self):
        return self["MSE"]

    @property
    def Shrinkage(self):
        return self["Shrinkage"]
    
    @property
    def BCEntropy(self):
        return self["BCEntropy"]
