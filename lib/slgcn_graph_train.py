import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch_cluster import random_walk
from torch_geometric.datasets import Planetoid

from .models.model_graph import DistMultPred, GraphNet
from .utils.utils import load_graph_from_tsv

device = "cpu"


def train_graph(parameters, args):
    os.makedirs(parameters["SAVING_ROOT_PATH"], exist_ok=True)

    FOLDER = parameters["FOLDER"]
    NUM_NODE_FEATURE = parameters["NUM_NODE_FEATURE"]
    filename = "data/ready/{}/graph.tsv".format(parameters["FOLDER"])
    print("[LOAD]", filename)
    adjs, nodes = load_graph_from_tsv(filename)
    node_num = len(nodes) + 1
    adjs = [torch.tensor(adj).to(device) for adj in adjs]
    adj_num = len(adjs)
    print("#nodes:", node_num)
    print("#types of edges:", adj_num)

    # model
    model = GraphNet(node_num, adj_num, NUM_NODE_FEATURE).to(device)
    params = list(model.parameters())
    if args.weight_model_enabled:
        weight_model = DistMultPred(adj_num, NUM_NODE_FEATURE)
        params += list(weight_model.parameters())

    optimizer = torch.optim.Adam(params, lr=0.0001, weight_decay=5e-4)
    #lr=0, weight_decay=5e-4)

    batch_num_per_adj = 100
    if batch_num_per_adj is None:
        batch_num_per_adj = node_num

    def train_graph_step():
        model.train()

        pos = []
        if args.prediction_edge_type is None:
            target_ids = []
            batches = []
            for target_id in range(adj_num):
                current_size = len(adjs[target_id][0])
                batch = torch.randint(
                    0,
                    current_size,
                    torch.Size([batch_num_per_adj]),
                    dtype=torch.long,
                )
                batches.append(batch)
                starting_position = adjs[target_id][0].index_select(0, batch)
                pos_walk = random_walk(
                    adjs[target_id][0],
                    adjs[target_id][1],
                    starting_position,
                    walk_length=1,
                    coalesced=False,
                )[:, 1]
                pos.append(pos_walk)
                target_ids.append(
                    torch.full(
                        torch.Size([batch_num_per_adj]),
                        target_id,
                        dtype=torch.long,
                    )
                )
            batch_num = batch_num_per_adj * adj_num
            pos_batch = torch.cat(pos, dim=0)
            target_ids = torch.cat(target_ids, dim=0)
        else:
            batch_num = batch_num_per_adj
            pos_batch = random_walk(
                adjs[args.prediction_edge_type][0],
                adjs[args.prediction_edge_type][1],
                batch,
                walk_length=1,
                coalesced=False,
            )[:, 1]

        neg_batch = torch.randint(
            0, node_num, torch.Size([batch_num]), dtype=torch.long
        )

        # forward
        optimizer.zero_grad()
        out = model(adjs)

        # computing loss
        if args.prediction_edge_type is None:
            node_in = []
            for i, batch in enumerate(batches):
                st_pos = adjs[i][0].index_select(0, batch)
                nod = out[st_pos, :]
                node_in.append(nod)
            node_in = torch.cat(node_in, 0)
        else:
            node_in = out[batch, :]

        if args.weight_model_enabled:
            node_in = weight_model(node_in, target_ids)
        node_pos = out[pos_batch, :]
        node_neg = out[neg_batch, :]
        pos_loss = F.logsigmoid((node_in * node_pos).sum(-1))
        neg_loss = F.logsigmoid(-(node_in * node_neg).sum(-1))
        loss = -pos_loss - neg_loss
        all_loss = loss.mean()
        ## metrics
        acc = torch.sum(pos_loss > neg_loss).to(torch.float32) / pos_loss.size(0)
        ##
        all_loss.backward()
        optimizer.step()
        return all_loss.item(), {"accuracy": acc.item()}

    best_loss = 10000
    for epoch in range(1, 100000):
        loss, metrics = train_graph_step()
        log = "Epoch: {:03d}, Train: {:.4f}"
        print(log.format(epoch, loss), metrics)
        if best_loss >= loss:
            best_loss = loss
            print("Saved model loss: {}, Epoch: {}".format(loss, epoch))

            model_path = "{}/graph.model.size_{}".format(
                parameters["SAVING_ROOT_PATH"], NUM_NODE_FEATURE
            )
            print("[SAVE]", model_path)
            torch.save(model.state_dict(), model_path)


def main(args):
    with open(args.config) as json_file:
        parameters = json.load(json_file)
    train_graph(parameters, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="Path to the JSON config file", default="config.json"
    )
    parser.add_argument(
        "--weight_model_enabled",
        help="Extra feature: use extra layer node encoding.",
        default=False,
    )
    parser.add_argument(
        "--prediction_edge_type",
        help="Extra feature: choose only one subgraph to encode, ignore the rest. None means all subgraphs are encoded.",
        default=None,
    )
    args = parser.parse_args()
    main(args)
