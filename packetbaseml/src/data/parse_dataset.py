import pandas as pd
import numpy as np
import os
import torch.utils.data as tdata
import torch
from alive_progress import alive_bar

from src.vars import DATA_PATH, MERGED_DATA_FILENAME, INTERIM_DATA_PATH, COLUMNS


def merge_csv_datasets(logger) -> pd.DataFrame:
    data_file_paths: list = [
        os.path.join(DATA_PATH, file_name)
        for file_name in os.listdir(DATA_PATH)
        if file_name != MERGED_DATA_FILENAME
    ]
    with alive_bar(len(data_file_paths), title="Merging dataframes...") as bar:
        logger.info("Started merging csv files")
        for data_file_path in data_file_paths:
            logger.info(f"Merging {data_file_path}...")
            chunks: list = pd.read_csv(
                data_file_path, low_memory=False, chunksize=10**3
            )
            for chunk in chunks:
                chunk_columns = chunk.columns
                if len(chunk_columns) != len(COLUMNS):
                    lacking_columns = [column for column in COLUMNS if column not in chunk_columns]
                    for column in lacking_columns:
                        chunk[column] = np.nan
                chunk.to_csv(
                    os.path.join(INTERIM_DATA_PATH, MERGED_DATA_FILENAME),
                    mode="a",
                    index=False,
                )
            logger.info(f"Csv file: {data_file_path} merged succesfully.")
            bar()


def save_dataframe_to_csv(dataframe: pd.DataFrame) -> None:
    dataframe.to_csv(os.path.join(INTERIM_DATA_PATH, MERGED_DATA_FILENAME), index=False)


def generate_dataframe(file_path: str, chunk_size: int = 10 ** 3, encoding: str = 'utf-8') -> pd.DataFrame:
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False, encoding=encoding):
        yield chunk


def load_csv_to_dataframe(file_path: str) -> pd.DataFrame:
    '''
    dataframe = pd.DataFrame()
    for chunk in generate_dataframe(file_path):
        dataframe = pd.concat([dataframe, chunk], ignore_index=True)
    '''
    dataframe = pd.concat(
        [
            df
            for df in pd.read_csv(
                file_path,
                chunksize=10**3,
                low_memory=False
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


if __name__ == '__main__':
    load_csv_to_dataframe('data/interim/merged.csv')