import os

import torch
from ..models.model_graph import GraphNet

from lib.models.convolution import SampleNetResConv
from lib.models.fully_connected import (
    SampleNet,
    SampleNetBatchNorm,
    SampleNetDropout,
    SampleNetFullResidual,
)
from lib.models.model_graph import GraphNet
from torch import device
from typing import Dict, Tuple, Type, Union


class ModelsContainer(object):
    def __init__(
        self,
        saving_root: str,
        model_name: str,
        random_state: int,
        model_class: Union[
            Type[SampleNetResConv],
            Type[SampleNetBatchNorm],
            Type[SampleNetDropout],
            Type[SampleNetFullResidual],
            Type[SampleNet],
        ],
        modelargs: Dict[str, int],
        graph_model_args: Dict[str, int],
        number_of_models: int,
        graph_train: bool,
        device: device,
    ) -> None:
        self.results = {}
        self.graph_train = graph_train
        if saving_root[-1] == "/":
            saving_root = saving_root[:-1]
        self.saving_root = saving_root
        self.model_name = model_name
        self.random_state = random_state
        self.model_class = model_class
        self.modelargs = modelargs
        self.graph_model_args = graph_model_args
        self.number_of_models = number_of_models
        self.device = device

    def __len__(self):
        return self.number_of_models

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[SampleNetBatchNorm, GraphNet],
        Tuple[SampleNetResConv, GraphNet],
        Tuple[SampleNetDropout, GraphNet],
        Tuple[SampleNetFullResidual, GraphNet],
        Tuple[SampleNet, GraphNet],
    ]:
        path_to_model = "{}/{}_{}_{}.pt".format(
            self.saving_root, self.model_name, self.random_state, idx
        )
        path_to_graph = "{}/graph_{}_{}_{}.pt".format(
            self.saving_root, self.model_name, self.random_state, idx
        )
        model, graph_model = self.create_model()
        model.to(self.device)
        graph_model.to(self.device)
        if os.path.isfile(path_to_model):
            model.load_state_dict(torch.load(path_to_model))

        if self.graph_train and os.path.isfile(path_to_graph):
            graph_model.load_state_dict(torch.load(path_to_graph))

        return model, graph_model

    def add_result(
        self,
        model,
        graph_model,
        idx,
        reference_metric,
        additional_metrics,
        gt,
        pred,
        epoch,
    ):
        if idx not in self.results:
            self.results[idx] = {
                "reference_metric": reference_metric,
                "additional_metrics": additional_metrics,
                "epoch": epoch,
            }
        else:
            saved_best = -10000
            if self.read_log(idx) is not None:
                saved_best = self.read_log(idx)
            if (
                self.results[idx]["reference_metric"] <= reference_metric
                and saved_best <= reference_metric
            ):
                self.save_model(model, graph_model, idx)
                self.results[idx]["reference_metric"] = reference_metric
                self.results[idx]["additional_metrics"] = additional_metrics
                self.results[idx]["epoch"] = epoch

                path_qq = "{}/qq_{}_{}_{}.png".format(
                    self.saving_root, self.model_name, self.random_state, idx
                )

                

                title = "{} E: {} Metr: {:.5f} k: {}".format(
                    self.model_name, epoch, reference_metric, idx
                )
            
                self.save_log(idx, reference_metric, additional_metrics, epoch)

    def create_model(
        self,
    ) -> Union[
        Tuple[SampleNetBatchNorm, GraphNet],
        Tuple[SampleNetResConv, GraphNet],
        Tuple[SampleNetDropout, GraphNet],
        Tuple[SampleNetFullResidual, GraphNet],
        Tuple[SampleNet, GraphNet],
    ]:
        base_model_path = "./model/graph.model.size_{}".format(
            self.graph_model_args["embedding_size"]
        )
        graph_model = GraphNet(**self.graph_model_args)
        graph_model.load_state_dict(torch.load(base_model_path))
        if not self.graph_train:
            graph_model.eval()
            for param in graph_model.parameters():
                param.requires_grad = False
        return self.model_class(**self.modelargs), graph_model

    def save_model(self, model, graph_model, idx):
        path = "{}/{}_{}_{}".format(
            self.saving_root, self.model_name, self.random_state, idx
        )
        path_to_graph = "{}/graph_{}_{}_{}".format(
            self.saving_root, self.model_name, self.random_state, idx
        )
        torch.save(model.state_dict(), "{}.pt".format(path))
        if self.graph_train:
            torch.save(graph_model.state_dict(), "{}.pt".format(path_to_graph))

    def save_log(self, idx, reference_metric, additional_metrics, epoch):
        path = "{}/{}_{}_{}".format(
            self.saving_root, self.model_name, self.random_state, idx
        )

        with open("{}".format(path), "w") as write_handle:
            write_handle.write(
                "Current Best Metric:{}\n".format(reference_metric)
            )
            for k, v in additional_metrics.items():
                write_handle.write("{}: {}\n".format(k, v))
            write_handle.write(
                "K partition: {}, Epoch: {}, Random State: {}\n".format(
                    idx, epoch, self.random_state
                )
            )

    def read_log(self, idx):
        path = "{}/{}_{}_{}".format(
            self.saving_root, self.model_name, self.random_state, idx
        )
        if os.path.isfile(path):
            with open("{}".format(path)) as read_handle:
                first_line = read_handle.readline()

            best_value = first_line.strip("\n")
            best_value = best_value.split("Current Best Metric:")[1]
            return float(best_value)
        else:
            return None
