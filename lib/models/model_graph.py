import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from torch_cluster import random_walk
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GATConv, GINConv, GraphConv
from torch import Tensor
from torch.nn.parameter import Parameter
from typing import List, Union


class DistMultPred(torch.nn.Module):
    def __init__(self, adj_num, input_feature):
        super(DistMultPred, self).__init__()
        self.adj_num = adj_num
        self.weight = torch.nn.Parameter(torch.zeros((adj_num, input_feature)))
        torch.nn.init.normal_(self.weight.data)

    def forward(self, out, taget_adj):
        return self.weight[taget_adj] * out


class GraphNet(torch.nn.Module):
    def __init__(
        self, node_num: int, adj_num: int, embedding_size: int = 32
    ) -> None:
        super(GraphNet, self).__init__()

        self.adj_num = adj_num
        self.num_layer = 2
        self.embed = torch.nn.Embedding(node_num, embedding_size)
        # self.linear_last = torch.nn.Linear(embedding_size * (self.num_layer+1), embedding_size) # with input
        self.linear_last = torch.nn.Linear(
            embedding_size * (self.num_layer), embedding_size
        )  # without input

        conv_layer_list = []
        linear_layer_list = []
        for _ in range(self.num_layer):
            conv_list1 = []
            linear_list1 = []
            # GIN Block
            # multiple adj.
            for _ in range(self.adj_num):  # Types of edges
                nn = torch.nn.Sequential(
                    torch.nn.Linear(embedding_size, embedding_size),
                    torch.nn.ELU(),
                    torch.nn.Linear(embedding_size, embedding_size),
                )
                conv_list1.append(GINConv(nn))
                linear_list1.append(
                    torch.nn.Linear(embedding_size, embedding_size)
                )
            conv_layer_list.append(torch.nn.ModuleList(conv_list1))
            linear_layer_list.append(torch.nn.ModuleList(linear_list1))
        ## regitering all layers
        self.conv_layer_list = torch.nn.ModuleList(conv_layer_list)
        self.linear_layer_list = torch.nn.ModuleList(linear_layer_list)

    def forward_one_layer(
        self, in_x: Union[Parameter, Tensor], adjs: List[Tensor], layer_id: int
    ) -> Tensor:
        out_x = []
        for j in range(self.adj_num):
            # x is the node features
            # adjs is the adjacency matrix
            x = F.elu(self.conv_layer_list[layer_id][j](in_x, adjs[j]))
            x = F.elu(self.linear_layer_list[layer_id][j](x))
            out_x.append(x)
        y = torch.stack(out_x, dim=0).sum(dim=0)
        return y
    
    def get_embed(self) -> Tensor:
        return self.embed.weight

    def forward_from_embed(self, embed: Tensor, adjs: List[Tensor]) -> Tensor:
        x=embed
        out_x = []  # without input
        # out_x=[x] #with input
        for i in range(self.num_layer):
            x = self.forward_one_layer(x, adjs, i)
            out_x.append(x)
        # y = torch.stack(out_x, dim=0).sum(dim=0)   # sum read out
        y = self.linear_last(torch.cat(out_x, dim=-1))  # concat read out
        return y

    def forward(self, adjs: List[Tensor]) -> Tensor:
        x = self.embed.weight
        return self.forward_from_embed(x,adjs)
        
