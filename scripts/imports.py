import os
from typing import Dict

import numpy as np
import pandas as pd

from config.global_configs import (
    DATA_FOLDER_PATH,
    DATASETS,
    TARGET,
    STORE_ID,
    DATE_COL
)


def import_csv(folder_path: str, filename: str, **kwargs):
    """
    Imports a csv file given its folder and name

    Args:
        folder_path (str): Name of the folder in which the csv file is
        filename (str): Name of the file, should end with a '.csv'
    """
    path_joined = os.path.join(folder_path, filename)
    return pd.read_csv(path_joined, **kwargs)


def import_data_folder(data_folder: str, datasets: Dict[str, str], **kwargs):
    """
    Import all the csv files from the data_folder.

    Args:
        data_folder (str): name of the folder containing the datasets
        datasets (Dict[str, str]): dictionnary containing the csv files along with their name
    """
    dict_return = dict()
    for name, path in datasets.items():
        dict_return[name] = import_csv(data_folder, path, **kwargs)

    return dict_return


def concat_train_test(df_train:str, df_test:str, target:str):
    """
    Concatenation of both the training and the testing sets.
    Differenciation will be made on the sales column - nan will be filled on target column for test set.

    Args:
        df_train (pd.DataFrame): dataframe with the training set
        df_test (pd.DataFrame): dataframe with the testing set
        target (str): name of the target column

    Returns:
        pd.DataFrame: concatenation of both dataframes
    """
    # Creation of a target column on the test set
    df_test_prep = df_test.copy()
    df_test_prep[target] = np.nan

    return pd.concat([df_train, df_test])


def merge_train_store(df_train: pd.DataFrame, df_store: pd.DataFrame, store_id: str):
    """
    Merge the training/testing sets with the store dataframe based on the store_id column

    Args:
        df_train (pd.DataFrame): dataframe containing the training set
        df_store (pd.DataFrame): dataframe containing store related information
        store_id (str): common column between the two dataframes
    """
    df_merge_store = df_train.merge(df_store, on=store_id, how='left')
    return df_merge_store


def merge_train_oil(df_train: pd.DataFrame, df_oil: pd.DataFrame, date_col: str):
    """
    Merge the training/testing sets with the oil dataframe based on the date column

    Args:
        df_train (pd.DataFrame): dataframe containing the train/test set concatenation
        df_oil (pd.DataFrame): dataframe containing the oil related information
        date_col (str): name of the column containing the date information in both dataframes
    """
    df_merge_oil = df_train.merge(df_oil, on=date_col, how='left')
    return df_merge_oil


def merge_train_transactions():
    pass


def merge_train_holidays():
    pass


def import_and_process(data_folder: str, dataset: str, target: str, store_id: str, date_col: str, **kwargs):
    # global compilation of previous functions

    # Import of all the dataframes
    dict_imports = import_data_folder(data_folder, dataset, **kwargs)

    # Concatenation of the training and testing datasets
    df_concat = concat_train_test(dict_imports['train'], dict_imports['test'], target)
    
    # Merging the concatenated dataframe with the store dataframe
    df_train_store = merge_train_store(df_concat, dict_imports['stores'], store_id)

    # Merging the concatenated dataframe with the oil dataframe
    df_train_oil = merge_train_store(df_train_store, dict_imports['oil'], date_col)
    return df_train_oil


if __name__ == "__main__":
    data_folder = DATA_FOLDER_PATH
    dataset = DATASETS
    target = TARGET
    store_id = STORE_ID
    date_col = DATE_COL

    df_final = import_and_process(data_folder, dataset, target, store_id, date_col)

    print(len(df_final))
    print(df_final.head(5))
    print(df_final.tail(5))

    print(df_final.isna().sum())
