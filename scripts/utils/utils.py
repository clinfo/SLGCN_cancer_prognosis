import csv
from typing import Dict, List

import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy import ndarray
from pandas.core.frame import DataFrame


def get_all_positions(value: str) -> List[str]:
    chromosome, positions = value.split(":")
    start_position, end_position = positions.split("-")

    values = []
    for current_position in range(int(start_position), int(end_position) + 1):
        values.append("{}:{}".format(chromosome, current_position))

    return values


def extract_genome_positions(mutation_data, separator=""):
    all_mutation_positions = []
    for genome_position in mutation_data["Mutation genome position"]:
        all_mutation_positions.extend(get_all_positions(genome_position))

    all_mutation_positions = list(sorted(set(all_mutation_positions)))
    mutation_features = {}
    encoded_mutation_position = {}

    mutation_data["Encoded Mutation Position"] = range(len(mutation_data))
    for cosmic_id, cell_line_mutations in tqdm(
        mutation_data.groupby(mutation_data.index)
    ):
        features = ["0" for _ in range(len(all_mutation_positions))]
        for _, sub_table in cell_line_mutations.iterrows():
            cell_line_mutation = sub_table["Mutation genome position"]
            mutations_per_gene = [
                "0" for _ in range(len(all_mutation_positions))
            ]
            for position in get_all_positions(cell_line_mutation):
                features[all_mutation_positions.index(position)] = "1"
                mutations_per_gene[
                    all_mutation_positions.index(position)
                ] = "1"
            encoded_mutation_position[
                sub_table["Encoded Mutation Position"]
            ] = separator.join(mutations_per_gene)
        mutation_features[cosmic_id] = separator.join(features)

    return mutation_features, encoded_mutation_position


def get_angles(pos: int, i: ndarray, d_model: int) -> ndarray:
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position: int, d_model: int) -> ndarray:
    angle_rads = get_angles(position, np.arange(d_model), d_model)
    # apply sin to even indices in the array; 2i
    angle_rads[0::2] = np.sin(angle_rads[0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[1::2] = np.cos(angle_rads[1::2])
    return angle_rads


def extract_genome_encoded_positions(mutation_data: DataFrame) -> DataFrame:
    mutation_data["Encoded Mutation Position"] = range(len(mutation_data))
    mutation_sample_ind = []
    total_features_vector = []

    # Iterate through the data
    for _, sub_table in tqdm(mutation_data.iterrows()):

        def get_pos_and_difference(position):
            start_pos, end_pos = position.split("-")
            return start_pos, int(end_pos) - int(start_pos)

        def split_base(position):
            conversion = []
            for i in range(3):
                ind = 3 * i
                ind_e = 3 * (i + 1)
                base_split = position[::-1][ind:ind_e]
                if base_split:
                    conversion.extend(
                        positional_encoding(int(base_split[::-1]), 4)
                    )
                else:
                    conversion.extend(positional_encoding(0, 4))
            return conversion

        mutation_sample_ind.append(sub_table["Encoded Mutation Position"])

        cell_line_mutation = sub_table["Mutation genome position"]
        chromosome, position = cell_line_mutation.split(":")
        raw_start_pos, diff = get_pos_and_difference(position)

        features_vector = split_base(raw_start_pos)

        features_vector.extend(positional_encoding(int(chromosome), 3))
        features_vector.extend(positional_encoding(diff, 3))

        total_features_vector.append(features_vector)
    columns = [
        "mutation_position_enc_{}".format(i)
        for i in range(len(features_vector))
    ]
    return pd.DataFrame(
        total_features_vector, index=mutation_sample_ind, columns=columns
    )


def extract_genome_positions(mutation_data, separator=""):
    all_mutation_positions = []
    for genome_position in mutation_data["Mutation genome position"]:
        all_mutation_positions.extend(get_all_positions(genome_position))

    all_mutation_positions = list(sorted(set(all_mutation_positions)))
    mutation_features = {}
    encoded_mutation_position = {}

    mutation_data["Encoded Mutation Position"] = range(len(mutation_data))
    for cosmic_id, cell_line_mutations in tqdm(
        mutation_data.groupby(mutation_data.index)
    ):
        features = ["0" for _ in range(len(all_mutation_positions))]
        for _, sub_table in cell_line_mutations.iterrows():
            cell_line_mutation = sub_table["Mutation genome position"]
            mutations_per_gene = [
                "0" for _ in range(len(all_mutation_positions))
            ]
            for position in get_all_positions(cell_line_mutation):
                features[all_mutation_positions.index(position)] = "1"
                mutations_per_gene[
                    all_mutation_positions.index(position)
                ] = "1"
            encoded_mutation_position[
                sub_table["Encoded Mutation Position"]
            ] = separator.join(mutations_per_gene)
        mutation_features[cosmic_id] = separator.join(features)

    return mutation_features, encoded_mutation_position


def extract_node_features(
    mutation_data: DataFrame, vertices_dic: Dict[str, int]
) -> DataFrame:
    node_features = mutation_data.drop(
        columns=[
            "Primary site",
            "cell_line_name",
            "cancer_type",
            "Mutation Description",
            "ID_tumour",
            "Mutation genome position",
            "Encoded Mutation Position",
            "Mutation CDS",
        ],
        errors="ignore",
    )
    # The following line keeps only nodes which are present in the Reactome graph
    node_features = node_features[
        node_features["HGNC ID"]
        .astype(int)
        .astype(str)
        .isin(vertices_dic.keys())
    ]
    node_features = node_features.dropna(subset=["HGNC ID"])
    node_features["NODE_ID"] = (
        node_features["HGNC ID"].astype(int).astype(str).map(vertices_dic)
    )
    node_features = node_features.dropna(subset=["NODE_ID"])
    node_features["NODE_ID"] = node_features["NODE_ID"].astype(int)
    node_features["HGNC ID"] = node_features["HGNC ID"].astype(int)

    return node_features.drop(columns=["HGNC ID"], errors="ignore")
