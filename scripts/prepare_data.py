import argparse
import csv
import json
import os
import re

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from .utils.drug_data_processor import DrugDataProcessor
from .utils.graph_generator import GraphGeneration
from .utils.mutations_processor import MutationsProcessor
from .utils.signature_analizer import MutationalSignals
from .utils.utils import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", help="Path to the JSON config file", default="config.json"
)
args = parser.parse_args()

with open(args.config) as json_file:
    parameters = json.load(json_file)

data_root = "data"

# Features to Include
CANCER_TYPE = parameters["CANCER_TYPE"]


# Folder to save the data (it will be created if it does not exist)
FOLDER = parameters["FOLDER"]

os.makedirs("{}/ready/{}".format(data_root, FOLDER), exist_ok=True)

GRAPH_DATA_FILE = "{}/{}".format(data_root, parameters["GRAPH_DATA_FILE"])

VERTICES_DIC = "{}/ready/{}/{}".format(
    data_root, FOLDER, parameters["VERTICES_DIC"]
)
RELATIONSHIPS_DIC = "{}/ready/{}/{}".format(
    data_root, FOLDER, parameters["RELATIONSHIPS_DIC"]
)
OUTPUT_GRAPH_FILE = "{}/ready/{}/{}".format(
    data_root, FOLDER, parameters["OUTPUT_GRAPH_FILE"]
)

# Generate the graph
graph_generator = GraphGeneration(GRAPH_DATA_FILE)
graph_generator.clean_graph()
graph_generator.save_graph(
    ENSEMBLE_TO_HGNC_DATA_FILE,
    VERTICES_DIC,
    RELATIONSHIPS_DIC,
    OUTPUT_GRAPH_FILE,
)

print("A graph was created.")
