import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from tqdm import tqdm

from ..utils.focal_loss import FocalLoss
from ..utils.utils import Converter
from lib.models.fully_connected import SampleNetDropout
from lib.models.model_graph import GraphNet
from lib.utils.loss_precision_clip import LossPrecissionClipper
from lib.utils.models_container import ModelsContainer
from numpy import float64

from torch import Tensor, device
from torch.optim.adamw import AdamW
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from typing import Dict, List, Optional, Union


torch.set_printoptions(edgeitems=10000)

class ExplModel(torch.nn.Module):
    def __init__(self, graph_model, sample_model) -> None:
        super(ExplModel, self).__init__()
        self.graph_model = graph_model
        self.sample_model= sample_model
    
    def get_embed(self):
        return self.graph_model.get_embed()

    def set_input(self, adjs, sample_node_idx, sample_idx) -> None:
        self.adjs=adjs
        self.sample_node_idx=sample_node_idx
        self.sample_idx=sample_idx

    def forward(self, embed, sample_node_feature, sample_feature) -> Tensor:
        out = self.graph_model.forward_from_embed(embed,self.adjs)
        pred = self.sample_model(
            out,
            self.sample_node_idx,
            sample_node_feature,
            self.sample_idx,
            sample_feature,
        )
        return pred

def collate_fn_gpu(batch):
    collated = {
        "labels": torch.cat([ele["labels"] for ele in batch]),
        "node_features": (
            torch.cat([ele["node_features"][0] for ele in batch]),
            torch.cat([ele["node_features"][1] for ele in batch]),
        ),
        "sample_features": (
            torch.cat([ele["sample_features"][0] for ele in batch]),
            torch.cat([ele["sample_features"][1] for ele in batch]),
        ),
    }
    return collated

def get_data_loader(subset, batch_size, num_workers, num_pca_features, device):
    subset["train"].dataset.apply_scaling_to_subset(
        subset["train"].indices, num_pca_features
    )

    if num_pca_features is not None:
        subset["train"].dataset.apply_pca_to_subset(
            subset["train"].indices, num_pca_features
        )

    if device.type != "cpu":
        collate_fn_train = collate_fn_gpu
        collate_fn_test = collate_fn_gpu
        subset_train = subset["train"]
        subset_test = subset["test"]
    else:

        def collate_fn_cpu_train(batch):
            return subset["train"][batch]

        def collate_fn_cpu_test(batch):
            return subset["test"][batch]

        collate_fn_train = collate_fn_cpu_train
        collate_fn_test = collate_fn_cpu_test
        subset_train = [*range(len(subset["train"]))]
        subset_test = [*range(len(subset["test"]))]
  
    pin_memory = device.type != "cpu"
    data_loader = torch.utils.data.DataLoader(
        subset_train,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_train,
        pin_memory=pin_memory,
    )

    data_loader_test = torch.utils.data.DataLoader(
        subset_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn_test,
        pin_memory=pin_memory,
    )

    return data_loader, data_loader_test

def train_model(
    kfold: int,
    subset: Dict[str, Subset],
    models_container: ModelsContainer,
    adjs: List[Tensor],
    criterion: LossPrecissionClipper,
    num_epochs: int,
    scaling_values: Dict[str, float64],
    batch_size: int = 10,
    num_workers: int = 0,
    num_pca_features: None = None,
    device: device = "cpu",
    with_explanation: bool = False,
    explanation_strategy: Union[int, str] =3,
):
    sample_model, graph_model = models_container[kfold]
    params = list(sample_model.parameters())
    if models_container.graph_train:
        params += list(graph_model.parameters())
    optimizer = torch.optim.AdamW(
        params,
        #first lr=0.0001
        lr=0.00005,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, verbose=True
    )
    
    data_loader, data_loader_test = get_data_loader(subset, batch_size, num_workers, num_pca_features, device)
    for epoch in range(num_epochs):
        train_one_epoch(
            sample_model,
            graph_model,
            adjs,
            optimizer,
            criterion,
            data_loader,
            epoch,
            device,
            models_container.saving_root,
        )

        reference = evaluate(
            sample_model,
            graph_model,
            adjs,
            data_loader_test,
            models_container,
            criterion,
            kfold,
            epoch,
            scaling_values,
            device,
        )
        lr_scheduler.step(reference)
    if with_explanation:
        expl=compute_explanation(
            sample_model,
            graph_model,
            adjs,
            data_loader_test,
            device,
            strategy=explanation_strategy)
        return reference, expl
    return reference, None 


def compute_one_explanation(
        expl_model,
        embed,
        sample_node_feature,
        sample_feature,
        target,
        n_alpha=5
        )->None:

    delta_alpha=1.0/n_alpha
    integral={}
    integral["sample_node_feature"]=None
    integral["sample_feature"]=None
    integral["embed"]=None
    integral["pred"]=[]
    
    for alpha in np.linspace(0.0, 1.0, n_alpha):
        pred=expl_model.forward(alpha*embed, alpha*sample_node_feature, alpha*sample_feature)
        pred[target].backward()
        integral["pred"].append(pred[target].item())
        ###
        #w=sample_node_feature.grad
        #key="sample_node_feature"
        #integral[key]=None
        #if integral[key] is None:
        #    integral[key] = delta_alpha*w
        #else:
        #    integral[key] +=delta_alpha*w

        #w=sample_feature.grad
        #key="sample_feature"
        #if integral[key] is None:
        #    integral[key] = delta_alpha*w
        #else:
        #    integral[key] +=delta_alpha*w

        #w=embed.grad
        #key="embed"
        #if integral[key] is None:
        #    integral[key] = delta_alpha*w
        #else:
        #    integral[key] +=delta_alpha*w
        ###
    #integral["sample_node_feature"]=(integral["sample_node_feature"]*sample_node_feature).detach().numpy()
    #integral["sample_feature"]     =(integral["sample_feature"]*sample_feature).detach().numpy()
    #integral["embed"]              =(integral["embed"]*embed).detach().numpy()
    return integral

def compute_explanation(
    model: SampleNetDropout,
    graph_model: GraphNet,
    adjs: List[Tensor],
    data_loader: DataLoader,
    device: device,
    strategy: 3,
) -> None:
    expl_model=ExplModel(graph_model, model)
    expl_model.eval()

    ground_truth = []
    predicted_values = []
    cancer_types = []
    total_loss_vali = 0.0
    #change False to True
    classification = True

    integral_list=[]
    for batch_data in data_loader:#tqdm(data_loader, desc="Explanation"):
        batch_data["node_features"] = (
            x.to(device) for x in batch_data["node_features"]
        )
        batch_data["sample_features"] = (
            x.to(device) for x in batch_data["sample_features"]
        )
        batch_data["labels"] = batch_data["labels"].to(device)
        
        batch_data["cancer_type"] = batch_data["cancer_type"].to(device)

        sample_node_idx, sample_node_feature = batch_data["node_features"]
        sample_idx, sample_feature = batch_data["sample_features"]
        sample_label = batch_data["labels"].unsqueeze(1)
        
        cancer_type = batch_data["cancer_type"]
        
        embed=expl_model.get_embed()
        expl_model.set_input(adjs, sample_node_idx, sample_idx)

        embed.requires_grad_()
        sample_node_feature.requires_grad_()
        sample_feature.requires_grad_()

        
        pred=expl_model.forward(embed, sample_node_feature, sample_feature)
        #pred_y=torch.argmax(pred,dim=1)
        n=len(sample_feature)

        expl_index=[]
        if strategy == "all":
            expl_index=list(range(n))
        elif strategy == "binary_class_positive":
            for i in range(n):
                label_i = sample_label[i,0].item()
                if int(label_i)==1:
                    expl_index.append(i)
        elif strategy == "binary_class_pred_positive":
            for i in range(n):
                pred_i =  pred[i,0].item()
                if label_i >= 0.5:
                    expl_index.append(i)
        elif strategy == "binary_class_positive_equal":
            for i in range(n):
                label_i, pred_i = sample_label[i].detach().numpy(), pred[i].detach().numpy()
                if label_i >= 0.5 and int(label_i)==1:
                    expl_index.append(i)
        elif type(strategy) is int:
            expl_index=list(range(n))
            np.random.shuffle(expl_index)
            expl_index=expl_index[:strategy]
        
        print("computing explanation:", expl_index)
        for j,i in enumerate(expl_index):
            label_i, pred_i = sample_label[i].detach().numpy(), pred[i].detach().numpy()
            print("{}/{}: label={} pred={}".format(j,len(expl_index), label_i, pred_i))
            target=(i,0)

            ig=compute_one_explanation(
                expl_model,
                embed,
                sample_node_feature,
                sample_feature,
                target
                )
            ig["i"]=i
            ig["cancer type"] = cancer_type[i]
            ig["label"]=label_i
            ig["pred"]=pred_i
            integral_list.append(ig)
    return integral_list



def train_one_epoch(
    model: SampleNetDropout,
    graph_model: GraphNet,
    adjs: List[Tensor],
    optimizer: AdamW,
    criterion: LossPrecissionClipper,
    data_loader: DataLoader,
    epoch: int,
    device: device,
    saving_root: str,
    verbose: bool = False,
) -> None:
    model.train()
    if next(graph_model.parameters()).requires_grad:
        graph_model.train()

    total_loss = 0.0
    for batch_data in tqdm(data_loader, desc="Train"):
        batch_data["node_features"] = (
            x.to(device) for x in batch_data["node_features"]
        )
        batch_data["sample_features"] = (
            x.to(device) for x in batch_data["sample_features"]
        )
        batch_data["labels"] = batch_data["labels"].to(device)

        sample_node_idx, sample_node_feature = batch_data["node_features"]
        sample_idx, sample_feature = batch_data["sample_features"]
        sample_label = batch_data["labels"]

        out = graph_model(adjs)
        pred = model(
            out,
            sample_node_idx,
            sample_node_feature,
            sample_idx,
            sample_feature,
        )
        sample_label = sample_label.unsqueeze(1)
        loss = criterion(pred, sample_label.type(torch.float))
        if verbose:
            tqdm.write("Batch loss:{:.4f}".format(loss))
        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.detach().item()
        torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
        optimizer.step()
    optimizer.zero_grad()
    log = "Epoch: {:03d}, Train: {:.4f}"
    print(log.format(epoch, total_loss / len(data_loader)))


@torch.no_grad()
def evaluate(
    model: SampleNetDropout,
    graph_model: GraphNet,
    adjs: List[Tensor],
    data_loader: DataLoader,
    models_container: ModelsContainer,
    criterion: LossPrecissionClipper,
    kfold: int,
    epoch: int,
    reference_scale: Dict[str, float64],
    device: device,
) -> None:
    model.eval()
    graph_model.eval()
    ground_truth = []
    predicted_values = []
    cancer_types = []
    
    total_loss_vali = 0.0
    #change False to True
    classification = True
    for batch_data in tqdm(data_loader, desc="Evaluation"):
        batch_data["node_features"] = (
            x.to(device) for x in batch_data["node_features"]
        )
        batch_data["sample_features"] = (
            x.to(device) for x in batch_data["sample_features"]
        )
        batch_data["labels"] = batch_data["labels"].to(device)
        
        batch_data["cancer_type"] = batch_data["cancer_type"].to(device)

        sample_node_idx, sample_node_feature = batch_data["node_features"]
        sample_idx, sample_feature = batch_data["sample_features"]
        sample_label = batch_data["labels"].unsqueeze(1)
        
        cancer_type = batch_data["cancer_type"]
        
        out = graph_model(adjs)
        pred = model(
            out,
            sample_node_idx,
            sample_node_feature,
            sample_idx,
            sample_feature,
        )
        
        vali_loss = criterion(pred, sample_label.type(torch.float))
        
        total_loss_vali += vali_loss.detach().item()

        #pred = pred.squeeze(1)

        predicted_values.extend(pred)
        ground_truth.extend(sample_label)
        cancer_types.extend(cancer_type)

    predicted_values = torch.stack(predicted_values).cpu()
    ground_truth = torch.stack(ground_truth).cpu()
    
    log = "Epoch: {:03d}, Vali: {:.4f}"
    print(log.format(epoch, total_loss_vali / len(data_loader)))
 
    return caclulate_metrics(
        ground_truth,
        predicted_values,
        model,
        graph_model,
        models_container,
        kfold,
        epoch,
        reference_scale,
        cancer_types,
    )


@torch.no_grad()
def caclulate_metrics(
    gt: Tensor,
    pred: Tensor,
    model: SampleNetDropout,
    graph_model: GraphNet,
    models_container: ModelsContainer,
    kfold: int,
    epoch: int,
    reference_scale: Dict[str, float64],
    cancer_types: Tensor,
):
    metrics = {}
    c = Converter(reference_scale)

    try:
        metrics["ROC_AUC"] = roc_auc_score(
            #c.sc_to_deactivated(gt), c.sc_to_deactivated(pred)
            gt, pred
        )
        fpr, tpr, thresholds = roc_curve(
            gt, pred
        )
        metrics["fpr"] = fpr
        metrics["tpr"] = tpr
        metrics["thresholds"] = thresholds
        metrics["cancer_types"] = cancer_types
        metrics["prediction"] = pred
        metrics["ground_truth"] = gt
    except ValueError:
        print("ROC_AUC only 1 class output, thus skipped")
    metrics["F1"] = f1_score(
        c.sc_to_deactivated(gt), c.sc_to_deactivated(pred)
    )

    cm = confusion_matrix(
        c.sc_to_deactivated(gt), c.sc_to_deactivated(pred)
    ).ravel()
    metrics["tn"] = cm[0]
    metrics["fp"] = cm[1]
    metrics["fn"] = cm[2]
    metrics["tp"] = cm[3]
    metrics["Acc"] = (metrics["tp"] + metrics["tn"]) / (
        metrics["tn"] + metrics["fp"] + metrics["fn"] + metrics["tp"]
    )

    for metric, value in metrics.items():
        print("{}: {}".format(metric, value))

    #reference = metrics["R2"]
    reference = metrics["ROC_AUC"]    
    models_container.add_result(
        model,
        graph_model,
        kfold,
        reference,
        metrics,
        c.sc_to_lnIC50(gt),
        c.sc_to_lnIC50(pred),
        epoch,
    )

    return reference
