import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.lines import Line2D
from numpy import float64, ndarray
from torch import Tensor
from typing import Any, Dict, Iterator, List, Tuple


def load_graph_from_tsv(filename: str) -> Tuple[List[ndarray], List[int]]:
    graphs = {}
    nodes = set()
    for line in open(filename):
        arr = line.strip().split("\t")
        eid = arr[1]  # Types of graphs
        if eid not in graphs:
            graphs[eid] = []
        graphs[eid].append([int(arr[0]), int(arr[2])])
        nodes.add(int(arr[0]))
        nodes.add(int(arr[2]))
    adjs = []  # All the edges in each graph type
    for k, edges in sorted(graphs.items()):  # k stands for the type of graph
        adj = np.transpose(np.array(list(sorted(edges))))
        adjs.append(adj)
    return adjs, list(nodes)


def load_sample_feature_from_tsv(filename):
    sample_idx = []
    feature = []
    for line in open(filename):
        arr = line.strip().split("\t")
        sid = int(arr[0])
        f = list(map(float, arr[1:]))
        sample_idx.append(sid)
        feature.append(f)
    return np.array(sample_idx), np.array(feature)


def load_sample_node_feature_from_tsv(filename):
    sample_node_idx = []
    feature = []
    for line in open(filename):
        arr = line.strip().split("\t")
        sid = int(arr[0])
        nid = int(arr[1])
        f = list(map(float, arr[2:]))
        sample_node_idx.append((sid, nid))
        feature.append(f)
    return np.array(sample_node_idx), np.array(feature)


def load_label_from_tsv(filename):
    labels = []
    for line in open(filename):
        arr = line.strip().split("\t")
        labels.append(int(arr[1]))
    return np.array(labels)

class Converter(object):
    def __init__(
        self, reference_scale: Dict[str, float64], cut_off: int = 0.5
    ) -> None:
        self.reference_scale = reference_scale
        self.cut_off = cut_off
    
    def sc_to_lnIC50(self, values: Tensor) -> Tensor:
        rescaled_lnIC50 = (
            values * self.reference_scale["std"] + self.reference_scale["mean"]
        )

        return rescaled_lnIC50

    def sc_to_deactivated(self, values):
        #rescaled_lnIC50 = (
        #    values * self.reference_scale["std"] + self.reference_scale["mean"]
        #)
        #deactivated = rescaled_lnIC50.lt(self.cut_off)
        #return deactivated
        values = values.lt(self.cut_off)
        return values

class LogCosh(object):
    def __init__(self, reduction="mean"):
        self.reduction = reduction

    def __call__(self, pred, true):
        loss = torch.log(torch.cosh(pred - true))

        if self.reduction == "mean":
            op = torch.mean
        elif self.reduction == "sum":
            op = torch.sum
        else:
            return loss
        return op(loss)
