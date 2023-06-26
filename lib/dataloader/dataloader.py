import csv

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from numpy import int64, ndarray
from pandas.core.frame import DataFrame
from torch import Tensor, device
from torch.utils.data.dataset import Subset
from typing import Dict, List, Tuple, Union


class FeaturesDataset(object):
    def __init__(
        self,
        node_features_path: str,
        sample_features_path: str,
        labels_path: str,
        device: device,
    ) -> None:
        self.regression_flag = False
        self.device = device

        self.statistics = {
            "mean": torch.tensor((0.0)).to(dtype=torch.float),
            "std": torch.tensor((1.0)).to(dtype=torch.float),
        }

        self.node_features_path = node_features_path
        self.sample_features_path = sample_features_path
        self.labels_path = labels_path

        print("Loading the features")
        self.node_features = pd.read_csv(
            node_features_path,
            header=None,
            index_col=0,
            sep="\t",
        )
        self.sample_features = pd.read_csv(
            sample_features_path,
            header=None,
            index_col=0,
            sep="\t",
        )
        self.labels = pd.read_csv(
            labels_path,
            header=None,
            index_col=0,
            sep="\t",
        )
        self.node_features = self.node_features.sort_index()
        self.sample_features = self.sample_features.sort_index()
        self.labels = self.labels.sort_index()

        self.get_label_statistics()

        self.stratification_labels = self.labels.loc[:, 1]
        self.cancer_type = self.sample_features.loc[:,:1]
        self.sample_features = self.sample_features.drop(columns=[1])
        self.pca_transform = None
        

    def __getitem__(
        self, idx: Union[int, ndarray]
    ) -> Dict[str, Union[Tuple[Tensor, Tensor], Tensor]]:
        if isinstance(idx, int) or isinstance(idx, np.int64):
            idx = [idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch = {
            "node_features": self.format_node_features(
                idx, self.node_features
            ),
            "sample_features": self.format_sample_features(
                idx, self.sample_features, self.statistics
            ),
            "labels": self.format_labels(
                idx, self.labels, self.regression_flag
            ),
            "cancer_type": self.format_labels(
                idx, self.cancer_type, self.regression_flag
            ), 
            
        }

        if self.pca_transform is not None:
            projection = batch["sample_features"][1].matmul(self.pca_transform)
            batch["sample_features"] = (
                batch["sample_features"][0],
                projection,
            )

        return batch

    def __len__(self) -> int64:
        return self.node_features.index.drop_duplicates()[-1] + 1

    def get_label_statistics(self) -> None:
        column = 1
        self.label_statistics = {
            "mean": self.labels.loc[:, column].mean(),
            "std": self.labels.loc[:, column].std(),
        }

    @staticmethod
    def format_node_features(
        idx: Union[List[int], ndarray], node_features: DataFrame
    ) -> Tuple[Tensor, Tensor]:
        sub_tab = node_features.loc[idx, 1]
        ids = torch.tensor(
            [sub_tab.index, sub_tab.values], dtype=torch.long
        ).permute(1, 0)
        features = torch.tensor(
            node_features.loc[idx, 2:].values, dtype=torch.float
        )
        return ids, features

    @staticmethod
    def format_sample_features(
        idx: Union[List[int], ndarray],
        sample_features: DataFrame,
        statistics: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor]:
        sub_table = sample_features.loc[idx, :]
        ids = torch.tensor(sub_table.index, dtype=torch.long)
        features = torch.tensor(sub_table.values, dtype=torch.float)
        features = (features - statistics["mean"]) / statistics["std"]
        features[features != features] = 0.0
        return ids, features

    def format_labels(
        self,
        idx: Union[List[int], ndarray],
        labels: DataFrame,
        regression_flag: bool,
    ) -> Tensor:
        if regression_flag:
            column = 1
            labels = torch.tensor(
                labels.loc[idx, column].values, dtype=torch.float
            )

            labels = (
                labels - self.label_statistics["mean"]
            ) / self.label_statistics["std"]
            labels = labels.unsqueeze(-1)
        else:
            column = 1
            labels = torch.tensor(
                labels.loc[idx, column].values, dtype=torch.long
            )
        return labels

    def get_data_constants(self, verbose: bool = True) -> Dict[str, int]:
        sample_node_idx, sample_node_feature = self[0]["node_features"]
        num_feature = sample_node_feature.shape[1]

        sample_idx, sample_feature = self[0]["sample_features"]
        num_sample_feature = sample_feature.shape[1]

        if self.regression_flag:
            num_label = 1
        else:
            #num_label = 2
            num_label = 1
        samples_label = self[0]["labels"]
        num_sample = samples_label.shape[0]

        if verbose:
            print("#node features:", num_feature)
            print("#features:", num_sample_feature)
            if self.regression_flag:
                print("Regression model")
            else:
                print("Classification model")

        return {
            "num_feature": num_feature,
            "num_sample_feature": num_sample_feature,
            "num_sample": num_sample,
            "num_label": num_label,
        }

    def apply_scaling_to_subset(
        self,
        subset_index: ndarray,
        num_pca_features: None,
        verbose: bool = True,
    ) -> None:
        # The subset functions should be used after the data is split in subsets
        subset = self.sample_features.loc[subset_index, :]
        self.statistics = {
            "mean": torch.tensor(np.array(subset).mean(0)).to(
                dtype=torch.float
            ),
            "std": torch.tensor(np.array(subset).std(0)).to(dtype=torch.float),
        }

    def apply_pca_to_subset(
        self, subset_index, num_pca_features, verbose=True
    ):
        # The subset functions should be used after the data is split in subsets
        subset = self.sample_features.loc[subset_index, :]
        if verbose:
            print("Performing PCA dimensionality reduction")
        normalized_df = (subset - subset.mean()) / subset.std()
        normalized_df = normalized_df.fillna(0)

        pca = PCA(n_components=num_pca_features)
        pca.fit(normalized_df)

        if verbose:
            explained_variance = np.sum(pca.explained_variance_ratio_)
            print(
                "Percentage of the variance explained by the chosen PC: {}".format(
                    explained_variance
                )
            )
        self.pca_transform = torch.tensor(pca.components_.T).to(
            dtype=torch.float
        )


class CrossValidationHandler(object):
    def __init__(
        self,
        dataset: FeaturesDataset,
        n_splits: int = 10,
        random_state: int = 1,
        shuffle: bool = True,
    ) -> None:
        self.kfold = StratifiedKFold(
            n_splits=n_splits, random_state=random_state, shuffle=shuffle
        )
        self.subsets = []
        sample_features = dataset.sample_features[
            dataset.sample_features.index < len(dataset)
        ]
        stratification_labels = dataset.stratification_labels[
            dataset.stratification_labels.index < len(dataset)
        ]
        for train_index, test_index in self.kfold.split(
            sample_features, stratification_labels
        ):
            train_subs = torch.utils.data.Subset(dataset, train_index)
            test_subs = torch.utils.data.Subset(dataset, test_index)

            self.subsets.append(
                {
                    "train": train_subs,
                    "test": test_subs,
                }
            )
        #np.savetxt('y5ngwct_idx.txt', test_index) 
    def __getitem__(self, idx: int) -> Dict[str, Subset]:
        return self.subsets[idx]

    def __len__(self):
        return len(self.subsets)
