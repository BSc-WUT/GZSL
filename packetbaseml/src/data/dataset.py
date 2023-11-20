import numpy as np
import pandas as pd
import os
import torch.utils.data as tdata

from src.vars import INTERIM_DATA_PATH, MERGED_DATA_FILENAME, COLUMNS_TO_DROP
from src.data.labels import labels, test_labels, train_labels
from src.data.parse_dataset import load_csv_to_dataframe, dataframe_to_dataloader


class DatasetIDS2018(tdata.Dataset):
    def __init__(
        self,
        csv_file_name: str = os.path.join(INTERIM_DATA_PATH, MERGED_DATA_FILENAME),
        transform=None,
    ):
        self.data = load_csv_to_dataframe(csv_file_name)
        self.transform = transform
        self.is_fixed = False

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple:
        data_point = self.data[idx]
        data_label = self.label[idx]
        if self.transform:
            data_point = self.transform(data_point)
        return data_point, data_label

    def filter_data_by_labels(self, labels: list) -> pd.DataFrame:
        """
        Returns pd.DataFrame which is filtered by labels from `labels` param.
        """
        mask = self.data["Label"].isin(labels)
        return pd.DataFrame(self.data[mask])

    def drop_columns(self, columns_to_drop: list) -> None:
        for column in columns_to_drop:
            if column in self.data.columns:
                del self.data[column]

    def fix_data_type_to_numeric(self) -> None:
        for column in self.data.columns:
            self.data[column] = pd.to_numeric(self.data[column], errors="coerce")

    def map_labels_to_idx(self) -> None:
        labels_idx: dict = {label: i for i, label in enumerate(labels)}
        self.data["Label"] = self.data["Label"].map(labels_idx)

    def fix_dataset(self) -> None:
        self.drop_columns(COLUMNS_TO_DROP)
        self.map_labels_to_idx()
        self.fix_data_type_to_numeric()
        self.is_fixed = True

    def load_dataloader(
        self, batch_size: int, data_loader_type: str
    ) -> tdata.DataLoader:
        if not self.is_fixed:
            self.fix_dataset()
        labels_idx: dict = {label: i for i, label in enumerate(labels)}
        l_labels: list = (
            train_labels[:-1] if data_loader_type == "train" else test_labels
        )  # Not including `Label` class
        l_labels_idx: list = [
            idx for label, idx in labels_idx.items() if label in l_labels
        ]
        dataframe: pd.DataFrame = self.filter_data_by_labels(l_labels_idx)
        dataloader: tdata.DataLoader = dataframe_to_dataloader(dataframe, batch_size)
        return dataloader
