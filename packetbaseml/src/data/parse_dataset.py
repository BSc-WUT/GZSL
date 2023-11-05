import pandas as pd
import os
import torch.utils.data as tdata
import torch
from alive_progress import alive_bar

from src.vars import DATA_PATH, MERGED_DATA_FILENAME, INTERIM_DATA_PATH


def merge_csv_datasets() -> pd.DataFrame:
    data_file_paths: list = [
        os.path.join(DATA_PATH, file_name)
        for file_name in os.listdir(DATA_PATH)
        if file_name != MERGED_DATA_FILENAME
    ]
    with alive_bar(len(data_file_paths), title="Merging dataframes...") as bar:
        for data_file_path in data_file_paths:
            chunks: list = pd.read_csv(
                data_file_path, low_memory=False, chunksize=10**3
            )
            for chunk in chunks:
                chunk.to_csv(
                    os.path.join(INTERIM_DATA_PATH, MERGED_DATA_FILENAME),
                    mode="a",
                    index=False,
                )
            bar()


def save_dataframe_to_csv(dataframe: pd.DataFrame) -> None:
    dataframe.to_csv(os.path.join(INTERIM_DATA_PATH, MERGED_DATA_FILENAME), index=False)


def load_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    dataframe = pd.concat(
        [
            df
            for df in pd.read_csv(
                file_path,
                chunksize=10**3,
                low_memory=False,
            )
        ]
    )
    return dataframe


def load_pickle_dataset(pickle_path: str) -> pd.DataFrame:
    dataframe: pd.DataFrame = pd.read_pickle(pickle_path)
    return dataframe


def save_pickle_dataset(dataset: pd.DataFrame, pickle_path: str) -> None:
    dataset.to_pickle(pickle_path)


def dataframe_to_dataloader(
    dataframe: pd.DataFrame, batch_size: int
) -> tdata.TensorDataset:
    dataset: tdata.TensorDataset = tdata.TensorDataset(
        torch.from_numpy(dataframe.values).float(),
        torch.from_numpy(dataframe.values[:, -1].astype(float)).float(),
    )
    data_loader: tdata.DataLoader = tdata.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    return data_loader
