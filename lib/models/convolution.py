import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch_scatter import scatter

from ..utils.depth_separable_conv import DepthSeparableConv1d
from torch import Tensor
from torch.nn.modules.activation import GELU
from torch.nn.modules.module import Module
from typing import Optional


class ConvResidualBlock1D(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        intermediate_channels: None = None,
        kernel_size: int = 15,
        stride: int = 1,
        padding: int = 1,
        activation: GELU = torch.nn.ELU,
    ) -> None:
        super().__init__()
        if intermediate_channels == None:
            intermediate_channels = out_channels // 2
        self.in_channels, self.out_channels, self.activation = (
            in_channels,
            out_channels,
            activation,
        )
        self.block1 = DepthSeparableConv1d(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn1 = torch.nn.BatchNorm1d(intermediate_channels)
        self.block2 = DepthSeparableConv1d(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.block1(x)
        x = self.activation(x)
        x = self.bn1(x)
        x = self.block2(x)
        x = self.activation(x)
        if self.should_apply_shortcut:
            x += residual

        return self.bn2(x)

    @property
    def should_apply_shortcut(self) -> bool:
        return self.in_channels == self.out_channels


class NN(torch.nn.Module):
    def __init__(
        self,
        embedding_size: int,
        channels_input: int = 1,
        channels_middle: int = 32,
        num_res_blocks: int = 2,
        first_kernel: int = 3,
        activation: GELU = torch.nn.LeakyReLU(),
    ) -> None:
        super(NN, self).__init__()

        downsample_list1 = [
            DepthSeparableConv1d(
                in_channels=1,
                out_channels=channels_middle,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            activation,
            torch.nn.BatchNorm1d(channels_middle),
            torch.nn.MaxPool1d(kernel_size=3),
        ]
        resblock_list1 = []
        for i in range(num_res_blocks):
            resblock_list1.append(
                ConvResidualBlock1D(
                    channels_middle,
                    channels_middle,
                    kernel_size=3,
                    stride=1,
                    activation=activation,
                )
            )

        downsample_list2 = [
            DepthSeparableConv1d(
                in_channels=channels_middle,
                out_channels=embedding_size,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            activation,
            torch.nn.BatchNorm1d(embedding_size),
            torch.nn.MaxPool1d(kernel_size=3),
        ]
        resblock_list2 = []
        for i in range(num_res_blocks):
            resblock_list2.append(
                ConvResidualBlock1D(
                    embedding_size,
                    embedding_size,
                    kernel_size=3,
                    stride=1,
                    activation=activation,
                )
            )

        self.down_sample1 = torch.nn.Sequential(*downsample_list1)
        self.down_sample2 = torch.nn.Sequential(*downsample_list2)
        self.res_blocks1 = torch.nn.Sequential(*resblock_list1)
        self.res_blocks2 = torch.nn.Sequential(*resblock_list2)
        self.last_conv = ConvResidualBlock1D(
            embedding_size,
            embedding_size,
            kernel_size=3,
            stride=1,
            activation=activation,
        )

    def forward(self, x: Tensor, dummy: Optional[Tensor] = None) -> Tensor:
        dummy
        x = self.down_sample1(x)
        x = self.res_blocks1(x)
        x = self.down_sample2(x)
        x = self.res_blocks2(x)
        x = self.last_conv(x)
        return torch.mean(x, dim=2)


class Modal(torch.nn.Module):
    def __init__(
        self,
        embedding_size: int,
        channels_input: int = 1,
        channels_middle: int = 32,
        num_res_blocks: int = 2,
        activation: GELU = torch.nn.LeakyReLU(),
    ) -> None:
        super(Modal, self).__init__()
        self.input = ConvResidualBlock1D(
            channels_input,
            channels_middle,
            kernel_size=3,
            stride=1,
            activation=activation,
        )

        resblock_list = []
        for i in range(num_res_blocks):
            resblock_list.append(
                ConvResidualBlock1D(
                    channels_middle,
                    channels_middle,
                    kernel_size=3,
                    stride=1,
                    activation=activation,
                )
            )

        self.res_blocks = torch.nn.Sequential(*resblock_list)
        self.last_conv = ConvResidualBlock1D(
            channels_middle,
            embedding_size,
            kernel_size=3,
            stride=1,
            activation=activation,
        )

    def forward(self, x: Tensor, dummy: Optional[Tensor] = None) -> Tensor:
        dummy
        x = self.input(x)
        x = self.res_blocks(x)
        x = self.last_conv(x)
        return torch.mean(x, dim=2)


class SampleNetResConv(torch.nn.Module):
    def __init__(
        self,
        num_input_feature: int,
        num_sample_feature: int,
        num_node_feature: int,
        n_label: int,
        embedding_size: int = 32,
        activation: GELU = torch.nn.GELU(),
    ) -> None:
        super(SampleNetResConv, self).__init__()
        self.nn = NN(
            embedding_size,
            channels_input=1,
            channels_middle=32,
            num_res_blocks=1,
            first_kernel=60,
            activation=activation,
        )
        self.nn_graph_modal1 = Modal(
            embedding_size,
            channels_input=1,
            channels_middle=64,
            num_res_blocks=2,
            activation=activation,
        )
        self.nn_graph_modal2 = Modal(
            embedding_size,
            channels_input=1,
            channels_middle=64,
            num_res_blocks=2,
            activation=activation,
        )

        self.nn_feature_modal = NN(
            embedding_size,
            channels_input=1,
            channels_middle=16,
            num_res_blocks=2,
            first_kernel=600,
            activation=activation,
        )
        self.nn_out = NN(
            embedding_size,
            channels_input=1,
            channels_middle=32,
            num_res_blocks=2,
            activation=activation,
        )
        self.final = torch.nn.Linear(embedding_size, n_label)
        self.apply(self._initialize_weights)

    def forward(
        self,
        graph_out: Tensor,
        sample_node_id: Tensor,
        sample_node_feature: Tensor,
        sample_id: Tensor,
        sample_feature: Tensor,
    ) -> Tensor:
        ## Two-modal block
        tensor_checkpoint = torch.ones(
            1, dtype=torch.float32, requires_grad=True
        )
        s_feature = checkpoint.checkpoint(
            self.nn, sample_node_feature.unsqueeze(1), tensor_checkpoint
        )

        g_feature = graph_out[
            sample_node_id[:, 1]
        ]
        x = s_feature + g_feature

        ## Aggregation
        tensor_checkpoint = torch.ones(
            1, dtype=torch.float32, requires_grad=True
        )
        x = checkpoint.checkpoint(
            self.nn_graph_modal1, x.unsqueeze(1), tensor_checkpoint
        )

        def scatter_checkpoint(dim, reduce):
            def custom_forward(*inputs):
                out = scatter(
                    inputs[0],
                    inputs[1],
                    dim=dim,
                    reduce=reduce,
                )
                inputs[2]
                return out

            return custom_forward

        tensor_checkpoint = torch.ones(
            1, dtype=torch.float32, requires_grad=True
        )
        x = checkpoint.checkpoint(
            scatter_checkpoint(dim=0, reduce="sum"),
            x,
            sample_node_id[:, 0],
            tensor_checkpoint,
        )
        x = x.index_select(0, sample_node_id[:, 0].unique())
        out_graph = self.nn_graph_modal2(x.unsqueeze(1))

        ## Multimodal block
        x = scatter(sample_feature, sample_id, dim=0, reduce="mean")
        x = x.index_select(0, sample_id.unique())
        out_feature = self.nn_feature_modal(x.unsqueeze(1))
        out = torch.cat([out_feature, out_graph], dim=-1)
        out = self.nn_out(out.unsqueeze(1))
        return self.final(out)

    @staticmethod
    def _initialize_weights(m: Module) -> None:
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(
                m.weight, gain=torch.nn.init.calculate_gain("leaky_relu")
            )
        if isinstance(m, torch.nn.BatchNorm1d):
            m.reset_parameters()


def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * padding) / stride) + 1
    return output
