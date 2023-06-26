import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch_scatter import scatter
from ..utils.depth_separable_conv import DepthSeparableConv1d
from torch import Tensor
from torch.nn.modules.activation import ELU



class SampleNet(torch.nn.Module):
    def __init__(
        self,
        num_input_feature: int,
        num_sample_feature: int,
        num_node_feature: int,
        n_label: int,
        embedding_size: int = 32,
    ) -> None:
        super(SampleNet, self).__init__()
        apply_initialization = False
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(
                num_input_feature, embedding_size
            ),  # the size of the encoding
            torch.nn.ELU(),
            torch.nn.Linear(embedding_size, num_node_feature),
        )
        self.nn_graph_modal1 = torch.nn.Sequential(
            torch.nn.Linear(num_node_feature, embedding_size),
            torch.nn.ELU(),
            torch.nn.Linear(embedding_size, embedding_size),
        )
        self.nn_graph_modal2 = torch.nn.Sequential(
            torch.nn.Linear(num_node_feature, embedding_size),
            torch.nn.ELU(),
            torch.nn.Linear(embedding_size, embedding_size),
        )
        self.nn_feature_modal = torch.nn.Sequential(
            torch.nn.Linear(num_sample_feature, embedding_size),
            torch.nn.ELU(),
            torch.nn.Linear(embedding_size, embedding_size),
        )
        self.nn_out = torch.nn.Sequential(
            torch.nn.Linear(embedding_size * 2, embedding_size),
            torch.nn.ELU(),
            torch.nn.Linear(embedding_size, n_label),
        )
        if apply_initialization:
            self.apply(self._initialize_weights)

    # sample_node_id: (#sample x #node) = list( sample_id, node_id )
    # sample_node_feature: (#sample x #node) x feature
    def forward(
        self,
        graph_out: Tensor,
        sample_node_id: Tensor,
        sample_node_feature: Tensor,
        sample_id: Tensor,
        sample_feature: Tensor,
    ) -> Tensor:
        ## Two-modal block
        s_feature = self.nn(sample_node_feature)
        g_feature = graph_out[sample_node_id[:, 1]]
        x = s_feature + g_feature

        ## Aggregation
        x = self.nn_graph_modal1(x)
        x = scatter(x, sample_node_id[:, 0], dim=0, reduce="sum")
        x = x.index_select(0, sample_node_id[:, 0].unique())
        out_graph = self.nn_graph_modal2(x)

        ## Multimodal block
        x = scatter(sample_feature, sample_id, dim=0, reduce="mean")
        x = x.index_select(0, sample_id.unique())
        out_feature = self.nn_feature_modal(x)
        out = torch.cat([out_feature, out_graph], dim=-1)
        return self.nn_out(out)


class SampleNetDropout(torch.nn.Module):
    def __init__(
        self,
        num_input_feature: int,
        num_sample_feature: int,
        num_node_feature: int,
        n_label: int,
        embedding_size: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super(SampleNetDropout, self).__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(
                num_input_feature, embedding_size
            ),  # the size of the encoding
            torch.nn.ELU(),
            torch.nn.Linear(embedding_size, num_node_feature),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout),
        )
        self.nn_graph_modal1 = torch.nn.Sequential(
            torch.nn.Linear(num_node_feature, embedding_size),
            torch.nn.ELU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout),
        )
        self.nn_graph_modal2 = torch.nn.Sequential(
            torch.nn.Linear(num_node_feature, embedding_size),
            torch.nn.ELU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout),
        )
        self.nn_feature_modal = torch.nn.Sequential(
            torch.nn.Linear(num_sample_feature, embedding_size),
            torch.nn.ELU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout),
        )
        self.nn_out = torch.nn.Sequential(
            torch.nn.Linear(embedding_size * 2, embedding_size),
            torch.nn.ELU(),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ELU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(embedding_size, n_label),
            torch.nn.Sigmoid(),
        )

    # sample_node_id: (#sample x #node) = list( sample_id, node_id )
    # sample_node_feature: (#sample x #node) x feature
    def forward(
        self,
        graph_out: Tensor,
        sample_node_id: Tensor,
        sample_node_feature: Tensor,
        sample_id: Tensor,
        sample_feature: Tensor,
    ) -> Tensor:
        ## Two-modal block
        s_feature = self.nn(sample_node_feature)
        g_feature = graph_out[sample_node_id[:, 1]]
        x = s_feature + g_feature

        ## Aggregation
        x = self.nn_graph_modal1(x)
        x = scatter(x, sample_node_id[:, 0], dim=0, reduce="sum")
        x = x.index_select(0, sample_node_id[:, 0].unique())
        out_graph = self.nn_graph_modal2(x)

        ## Multimodal block
        x = scatter(sample_feature, sample_id, dim=0, reduce="mean")
        x = x.index_select(0, sample_id.unique())
        out_feature = self.nn_feature_modal(x)
        out = torch.cat([out_feature, out_graph], dim=-1)
        return self.nn_out(out)


class SampleNetBatchNorm(torch.nn.Module):
    def __init__(
        self,
        num_input_feature: int,
        num_sample_feature: int,
        num_node_feature: int,
        n_label: int,
        embedding_size: int = 32,
    ) -> None:
        super(SampleNetBatchNorm, self).__init__()
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(
                num_input_feature, embedding_size
            ),  # the size of the encoding
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(embedding_size),
            torch.nn.Linear(embedding_size, num_node_feature),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(num_node_feature),
        )
        self.nn_graph_modal1 = torch.nn.Sequential(
            torch.nn.Linear(num_node_feature, embedding_size),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(embedding_size),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(embedding_size),
        )
        self.nn_graph_modal2 = torch.nn.Sequential(
            torch.nn.Linear(num_node_feature, embedding_size),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(embedding_size),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(embedding_size),
        )
        self.nn_feature_modal = torch.nn.Sequential(
            torch.nn.Linear(num_sample_feature, embedding_size),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(embedding_size),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(embedding_size),
        )
        self.nn_out = torch.nn.Sequential(
            torch.nn.Linear(embedding_size * 2, embedding_size),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(embedding_size),
            torch.nn.Linear(embedding_size, embedding_size),
            torch.nn.ELU(),
            torch.nn.BatchNorm1d(embedding_size),
            torch.nn.Linear(embedding_size, n_label),
            torch.nn.Sigmoid(),
        )

    # sample_node_id: (#sample x #node) = list( sample_id, node_id )
    # sample_node_feature: (#sample x #node) x feature
    def forward(
        self,
        graph_out: Tensor,
        sample_node_id: Tensor,
        sample_node_feature: Tensor,
        sample_id: Tensor,
        sample_feature: Tensor,
    ) -> Tensor:
        ## Two-modal block
        s_feature = self.nn(sample_node_feature)
        #g_feature = graph_out[sample_node_id[:, 1]]
        x = s_feature #+ g_feature
        
        ## Aggregation
        x = self.nn_graph_modal1(x)
        x = scatter(x, sample_node_id[:, 0], dim=0, reduce="sum")
        x = x.index_select(0, sample_node_id[:, 0].unique())
        out_graph = self.nn_graph_modal2(x)

        ## Multimodal block
        x = scatter(sample_feature, sample_id, dim=0, reduce="mean")
        x = x.index_select(0, sample_id.unique())
        out_feature = self.nn_feature_modal(x)
        out = torch.cat([out_feature, out_graph], dim=-1)
        return self.nn_out(out)

    @staticmethod
    def _initialize_weights(m):
        if isinstance(m, torch.nn.Conv1d):
            torch.nn.init.xavier_uniform_(
                m.weight, gain=torch.nn.init.calculate_gain("leaky_relu")
            )
            torch.nn.init.constant_(m.bias, 0.1)
        elif isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(
                m.weight, gain=torch.nn.init.calculate_gain("leaky_relu")
            )
            torch.nn.init.constant_(m.bias, 0.1)


class ResidualBlock1D(torch.nn.Module):
    def __init__(
        self,
        num_input_feature: int,
        num_output_feature: int,
        intermediate_size: int = 32,
        activation: ELU = torch.nn.ELU,
        apply_norm: bool = False,
        apply_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.apply_norm = False
        self.apply_dropout = False
        self.num_input_feature, self.num_output_feature, self.activation = (
            num_input_feature,
            num_output_feature,
            activation,
        )
        self.block1 = torch.nn.Linear(num_input_feature, intermediate_size)
        self.block2 = torch.nn.Linear(intermediate_size, num_output_feature)
        if self.apply_norm:
            self.bn1 =torch.nn.BatchNorm1d(intermediate_size)
            self.bn2 =torch.nn.BatchNorm1d(num_output_feature)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.block1(x)
        if self.apply_norm:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.block2(x)
        if self.apply_norm:
            x = self.bn2(x)
        if self.should_apply_shortcut:
            x += residual

        x = self.activation(x)
        if self.apply_dropout:
            x = torch.nn.Dropout(p=0.1)(x)
        return x

    @property
    def should_apply_shortcut(self) -> bool:
        return self.num_input_feature == self.num_output_feature


class SampleNetFullResidual(torch.nn.Module):
    def __init__(
        self,
        num_input_feature: int,
        num_sample_feature: int,
        num_node_feature: int,
        n_label: int,
        embedding_size: int = 32,
    ) -> None:
        super(SampleNetFullResidual, self).__init__()
        apply_initialization = False
        self.nn = torch.nn.Sequential(
            ResidualBlock1D(
                num_input_feature,
                num_input_feature,
                num_input_feature,
                activation=torch.nn.ELU(),
            ),
            ResidualBlock1D(
                num_input_feature,
                num_input_feature,
                num_input_feature,
                activation=torch.nn.ELU(),
            ),
            ResidualBlock1D(
                num_input_feature,
                num_node_feature,
                num_node_feature // 2,
                activation=torch.nn.ELU(),
            ),
        )
        self.nn_graph_modal1 = torch.nn.Sequential(
            ResidualBlock1D(
                num_node_feature,
                embedding_size,
                embedding_size,
                activation=torch.nn.ELU(),
            ),
            ResidualBlock1D(
                embedding_size,
                embedding_size,
                embedding_size,
                activation=torch.nn.ELU(),
            ),
            ResidualBlock1D(
                embedding_size,
                embedding_size,
                embedding_size // 2,
                activation=torch.nn.ELU(),
            ),
        )
        self.nn_graph_modal2 = torch.nn.Sequential(
            ResidualBlock1D(
                num_node_feature,
                embedding_size,
                embedding_size,
                activation=torch.nn.ELU(),
            ),
            ResidualBlock1D(
                embedding_size,
                embedding_size,
                embedding_size,
                activation=torch.nn.ELU(),
            ),
            ResidualBlock1D(
                embedding_size,
                embedding_size,
                embedding_size // 2,
                activation=torch.nn.ELU(),
            ),
        )
        self.nn_feature_modal = torch.nn.Sequential(
            ResidualBlock1D(
                num_sample_feature,
                embedding_size,
                embedding_size,
                activation=torch.nn.ELU(),
            ),
            ResidualBlock1D(
                embedding_size,
                embedding_size,
                embedding_size,
                activation=torch.nn.ELU(),
            ),
            ResidualBlock1D(
                embedding_size,
                embedding_size,
                embedding_size // 2,
                activation=torch.nn.ELU(),
            ),
        )
        self.nn_out = torch.nn.Sequential(
            ResidualBlock1D(
                embedding_size * 2,
                embedding_size,
                embedding_size,
                activation=torch.nn.ELU(),
            ),
            ResidualBlock1D(
                embedding_size,
                embedding_size,
                embedding_size,
                activation=torch.nn.ELU(),
            ),
            ResidualBlock1D(
                embedding_size,
                embedding_size,
                embedding_size // 2,
                activation=torch.nn.ELU(),
            ),
            torch.nn.Linear(embedding_size, n_label),
        )
        if apply_initialization:
            self.apply(self._initialize_weights)

    # sample_node_id: (#sample x #node) = list( sample_id, node_id )
    # sample_node_feature: (#sample x #node) x feature
    def forward(
        self,
        graph_out: Tensor,
        sample_node_id: Tensor,
        sample_node_feature: Tensor,
        sample_id: Tensor,
        sample_feature: Tensor,
    ) -> Tensor:
        ## Two-modal block
        s_feature = self.nn(sample_node_feature)
        g_feature = graph_out[sample_node_id[:, 1]]
        x = s_feature + g_feature

        ## Aggregation
        x = self.nn_graph_modal1(x)
        x = scatter(x, sample_node_id[:, 0], dim=0, reduce="mean")
        x = x.index_select(0, sample_node_id[:, 0].unique())
        out_graph = self.nn_graph_modal2(x)

        ## Multimodal block
        x = scatter(sample_feature, sample_id, dim=0, reduce="mean")
        x = x.index_select(0, sample_id.unique())
        out_feature = self.nn_feature_modal(x)
        out = torch.cat([out_feature, out_graph], dim=-1)
        return self.nn_out(out)


def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2 * padding) / stride) + 1
    return output
