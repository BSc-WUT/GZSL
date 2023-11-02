import pandas as pd
import os
import torch.utils.data as pdata

from src.vars import INTERIM_DATA_PATH, MERGED_DATA_FILENAME


class DatasetIDS2018(pdata.Dataset):
    def __init__(
        self,
        csv_file_name: str = os.path.join(INTERIM_DATA_PATH, MERGED_DATA_FILENAME),
        transform=None,
    ):
        self.data = pd.read_csv(csv_file_name)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_point = self.data[idx]
        data_label = self.label[idx]
        if self.transform:
            data_point = self.transform(data_point)
        return data_point, data_label

    def get_data_by_labels(self, labels: list) -> pd.DataFrame:
        mask = self.data["Label"].isin(labels)
        return pd.DataFrame(self.data[mask])


def parse_dataset(
    dataset: pd.DataFrame,
    columns_to_drop: list,
    labels_emb: dict,
    labels_column_name: str,
) -> pd.DataFrame:
    for column in columns_to_drop:
        if column in list(dataset.columns):
            del dataset[column]
    dataset[labels_column_name] = dataset[labels_column_name].map(labels_emb)
    for column in dataset.columns:
        dataset[column] = pd.to_numeric(dataset[column], errors="coerce")
    return dataset
