import os
import csv
import math
import re
from typing import Dict, Tuple, List

import pandas as pd
from pandas.core.frame import DataFrame


def make_nodes_undirected(graph: DataFrame) -> DataFrame:
    graph_prime = graph.rename(columns={"A": "B", "B": "A"})
    cat = pd.concat([graph, graph_prime], ignore_index=True)
    return cat.drop_duplicates(ignore_index=True)


def remove_self_connections(graph: DataFrame) -> DataFrame:
    graph = graph.query("A != B")
    return graph.reset_index(drop=True)


def clean_graph(graph: DataFrame) -> DataFrame:
    graph = graph.drop_duplicates(ignore_index=True)
    graph = make_nodes_undirected(graph)
    graph = remove_self_connections(graph)
    return graph


def convert_gene_ids(graph: DataFrame, ensambl: DataFrame) -> DataFrame:
    ensambl["HGNC ID"] = ensambl["HGNC ID"].str.split("HGNC:", 1).str[-1]
    dic = ensambl.reset_index(drop=True).to_dict("records")
    dic = {element["Ensembl gene ID"]: element["HGNC ID"] for element in dic}

    graph["a"] = graph["A"].map(dic)
    graph["b"] = graph["B"].map(dic)
    graph.loc[graph["a"].isnull(), "a"] = graph["A"]
    graph.loc[graph["b"].isnull(), "b"] = graph["B"]

    return graph


def create_labels_relations_dic(
    graph: DataFrame,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    def create_dic_nodes(graph_):
        uniq_a = graph_["a"].unique()
        uniq_b = graph_["b"].unique()
        unique_nodes = set()
        unique_nodes.update(uniq_a)
        unique_nodes.update(uniq_b)

        return {v: n for n, v in enumerate(unique_nodes)}

    def create_dic_relations(graph_):
        uniq_r = graph_["relationship"].unique()
        unique_relationship = set(uniq_r)

        return {v: n for n, v in enumerate(unique_relationship)}

    return create_dic_nodes(graph), create_dic_relations(graph)


def save_dics(dic: Dict[str, int], output_file: str) -> None:
    pd.DataFrame.from_dict(dic, orient="index").to_csv(
        output_file, sep="\t", header=False, quoting=csv.QUOTE_NONE
    )


class GraphGeneration(object):
    def __init__(self, graph_data_file: str) -> None:
        print("Loading Graph Data...")
        graph_columns = ["A", "relationship", "B"]
        with open(graph_data_file) as csv_file:
            reader = csv.reader(csv_file, delimiter="\t")
            self.graph_data = pd.DataFrame(list(reader), columns=graph_columns)

    def clean_graph(self) -> None:
        print("Cleaning graph data...")

        self.graph_data = clean_graph(self.graph_data)
        self.graph_data["A"] = (
            self.graph_data["A"].str.split("ENSEMBL:", 1).str[-1]
        )
        self.graph_data["B"] = (
            self.graph_data["B"].str.split("ENSEMBL:", 1).str[-1]
        )

    def save_graph(
        self,
        ensamble_to_hgnc: str,
        vertices_dic_location: str,
        relationships_dic_location: str,
        graph_location: str,
    ) -> None:
        print("Saving graph...")
        ensemble_to_hgnc = pd.read_csv(ensamble_to_hgnc, sep="\t")
        ensemble_to_hgnc = ensemble_to_hgnc.dropna()

        self.graph_data = convert_gene_ids(self.graph_data, ensemble_to_hgnc)
        (
            self.vertices_dic,
            self.relationships_dic,
        ) = create_labels_relations_dic(self.graph_data)
        self.graph_data["edge1"] = self.graph_data["a"].map(self.vertices_dic)
        self.graph_data["edge2"] = self.graph_data["b"].map(self.vertices_dic)
        self.graph_data["rel"] = self.graph_data["relationship"].map(
            self.relationships_dic
        )
        self.graph_data[["edge1", "rel", "edge2"]].to_csv(
            graph_location,
            sep="\t",
            index=False,
            header=False,
            quoting=csv.QUOTE_NONE,
        )
        save_dics(self.vertices_dic, vertices_dic_location)
        save_dics(self.relationships_dic, relationships_dic_location)

    def get_vertices_dic(self) -> Dict[str, int]:
        return self.vertices_dic
