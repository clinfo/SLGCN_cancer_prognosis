import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from .dataloader.dataloader import CrossValidationHandler, FeaturesDataset
from .engine.engine import train_model
from .utils.models_container import ModelsContainer
from .utils.structures import LossDictionary, ModelsDictionary
from .utils.utils import load_graph_from_tsv

import joblib


def main(parameters):
    if torch.cuda.is_available() and parameters["CUDA"]:
        device = torch.device("cuda")
        num_workers = parameters["num_workers"]
    else:
        device = torch.device("cpu")
        num_workers = 0

    SAVING_ROOT_PATH = "{}/{}".format(
        parameters["SAVING_ROOT_PATH"], parameters["MODEL_KEYWORD_NAME"]
    )
    os.makedirs(SAVING_ROOT_PATH, exist_ok=True)
    FOLDER = parameters["FOLDER"]

    model = ModelsDictionary()[parameters["MODEL"]]
    
    criterion = LossDictionary(
        parameters["REDUCTION"],
        parameters["LOSS_CLIP"],
    )[parameters["LOSS"]]

    print("Model: {}".format(parameters["MODEL_KEYWORD_NAME"]))
    filename = "data/ready/{}/graph.tsv".format(FOLDER)
    dataset = FeaturesDataset(
        node_features_path="data/ready/{}/5years_node_input.tsv".format(FOLDER),
        sample_features_path="data/ready/{}/5years_sampleNOCT.tsv".format(FOLDER),
        labels_path="data/ready/{}/5years_labels.tsv".format(FOLDER),
        device=device,
    )

    print("[LOAD]", filename)
    adjs, nodes = load_graph_from_tsv(filename)
    node_num = len(nodes) + 1
    adjs = [torch.tensor(adj).to(device) for adj in adjs]
    adj_num = len(adjs)
    print("#nodes:", node_num)
    print("#types of edges:", adj_num)
    print("#samples:",len(dataset))

    data_constants = dataset.get_data_constants()
    cross_val = CrossValidationHandler(
        dataset,
        n_splits=parameters["CROSS_VALIDATION_SPLITS"],
        random_state=parameters["RANDOM_STATE"],
    )

    modelargs = {
        "num_input_feature": data_constants["num_feature"],
        "num_sample_feature": data_constants["num_sample_feature"],
        "num_node_feature": parameters["NUM_NODE_FEATURE"],
        "embedding_size": parameters["EMBEDDING_SIZE"],
        "n_label": data_constants["num_label"],
    }
    graph_model_args = {
        "node_num": node_num,
        "adj_num": adj_num,
        "embedding_size": parameters["EMBEDDING_SIZE"],
    }

    if parameters["NUMBER_PRINCIPAL_COMPONENTS"] is not None:
        modelargs["num_sample_feature"] = parameters[
            "NUMBER_PRINCIPAL_COMPONENTS"
        ]

    models_container = ModelsContainer(
        SAVING_ROOT_PATH,
        parameters["MODEL_KEYWORD_NAME"],
        parameters["RANDOM_STATE"],
        model,
        modelargs,
        graph_model_args,
        parameters["CROSS_VALIDATION_SPLITS"],
        parameters["GRAPH_TRAINING"],
        device,
    )

    scaling_values = dataset.label_statistics

    expl_list=[]
    expl_path=parameters["EXPLANATION_PATH"]
    with_explanation = False
    if expl_path is not None:
        with_explanation = True
        
    for kfold, subset in enumerate(cross_val):
        expl_path = expl_path=parameters["EXPLANATION_PATH"]
        expl_path = expl_path + str(kfold) + "fold.jbl"
        expl_list = []
        #expl_path = f"IGall_5y_nograph_withCT_expCTIG_rm100_{kfold}.jbl"
        ref, expl = train_model(
            kfold,
            subset,
            models_container,
            adjs,
            criterion,
            parameters["EPOCHS"],
            scaling_values,
            parameters["BATCH_SIZE"],
            num_workers,
            parameters["NUMBER_PRINCIPAL_COMPONENTS"],
            device,
            with_explanation = with_explanation,
            explanation_strategy=parameters["EXPLANATION_STRATEGY"],
        )
        expl_list.append(expl)
        if expl_path is not None:
            joblib.dump(expl_list,expl_path,compress=True)
    #if expl_path is not None:
    #    joblib.dump(expl_list,expl_path,compress=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Path to the JSON config file", default="config.json"
    )
    args = parser.parse_args()
    with open(args.config) as json_file:
        parameters = json.load(json_file)
    main(parameters)
