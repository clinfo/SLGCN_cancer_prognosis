import torch
import torch.nn.functional as F
from torch_scatter import scatter
from torch import Tensor


class DepthSeparableConv1d(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ) -> None:
        super(DepthSeparableConv1d, self).__init__()

        self.conv = torch.nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups=in_channels,
            bias=bias,
        )
        self.depth_conv = torch.nn.Conv1d(
            in_channels, out_channels, kernel_size=1, bias=bias
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.depth_conv(x)
        return x
