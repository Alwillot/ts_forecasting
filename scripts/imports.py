from operator import index
import os
from typing import Dict
from xmlrpc.client import Boolean

import numpy as np
import pandas as pd


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


def merge_train_transactions(df_train: pd.DataFrame, df_transactions: pd.DataFrame, date_col: str, store_id:str):
    """
    Merge the training/testing sets with the transactions dataframe based on the date column and the store id.

    Args:
        df_train (pd.DataFrame): dataframe containing the train/test set concatenation
        df_transactions (pd.DataFrame): dataframe containing the transactions related information
        date_col (str): name of the column containing the date information in both dataframes
        store_id (str): common column between the two dataframes
    """
    df_merge_transactions = df_train.merge(df_transactions, on=[store_id, date_col], how='left')
    return df_merge_transactions


def merge_train_holidays(df_train: pd.DataFrame, df_holidays: pd.DataFrame, date_col: str):
    """
    Merge the training/testing sets with the holidays dataframe based on the date column

    Args:
        df_train (pd.DataFrame): dataframe containing the train/test set concatenation
        df_holidays (pd.DataFrame): dataframe containing the holidays related information
        date_col (str): name of the column containing the date information in both dataframes
    """
    # Duplicate issue with the name
    if "type" in df_holidays.columns:
        df_holidays = df_holidays.rename(columns={"type": "type_holidays"})

    df_merge_holidays = df_train.merge(df_holidays, on=date_col, how='left')
    return df_merge_holidays


def import_and_process(
    data_folder: str,
    dataset: Dict[str, str],
    target: str,
    store_id: str,
    date_col: str,
    **kwargs
):
    """
    Import all the csv files contained in the data folder and perform successive merges in order to
    create a final dataframe ready for feature engineering.

    Args:
        data_folder (str): Name of the folder in which the csv file is
        dataset (Dict[str, str]): dictionnary containing the csv files along with their name
        target (str): name of the target column in the training dataframe
        store_id (str): column containing the store identifier
        date_col (str): column containing the date information

    Returns:
        pd.DataFrame: dataframe containing the merged information from all individual dataframe
    """
    ## IMPORTS
    # Import of all the dataframes
    dict_imports = import_data_folder(data_folder, dataset, **kwargs)

    ## CONCATENATION
    # Concatenation of the training and testing datasets
    df_concat = concat_train_test(dict_imports['train'], dict_imports['test'], target)

    # Merging the concatenated dataframe with the store dataframe
    df_train_store = merge_train_store(df_concat, dict_imports['stores'], store_id)

    # Merging the concatenated dataframe with the oil dataframe
    df_train_oil = merge_train_oil(df_train_store, dict_imports['oil'], date_col)

    # Merging the concatenated dataframe with the holidays dataframe
    df_train_holidays = merge_train_holidays(df_train_oil, dict_imports['holidays'], date_col)

    # Merging the concatenated dataframe with the transactions dataframe
    df_train_complete = merge_train_transactions(df_train_holidays, dict_imports['transactions'], date_col, store_id)
    
    return df_train_complete


def export_dataframe(df:pd.DataFrame, data_folder:str, final_path:str):
    """
    Export the final dataframe as a csv file.

    Args:
        df (pd.DataFrame): dataframe that should be exported
        data_folder (str): name of the folder containing the data information
        final_path (str): path from the data folder to the 

    Returns:
        None
    """
    path_to_save = os.path.join(data_folder, final_path)
    df.to_csv(path_to_save, index=False)
    
    print("Export made successfully")
    return None


def prepare_and_save(
    data_folder: str,
    datasets_path: Dict[str, str],
    final_dataset_folder: str,
    final_dataset_name: str,
    target_col: str,
    store_id_col: str,
    date_col: str,
    save: bool=True,
    **kwargs
):
    """
    Handle the data importation, preparation, and exportation if not already existing.
    Else simply import the existing csv file.

    Args:
        data_folder (str): path to the data folder
        datasets_path (Dict[str): name of each csv file in the data folder
        final_dataset_folder (str): path from the root of the project to the final data folder
        final_dataset_name (str): name of the final dataframe (csv file)
        target_col (str): name of the column that we aim at predicting
        store_id_col (str): name of the column containing the stores ids in the dataframes
        date_col (str): name of the column containing the date information in the dataframes
        save (Boolean): determines whether to save the created dataframe if not existing (defaults to True)
    """
    # If the dataframe is not already existing, we create it
    if not os.path.exists(os.path.join(final_dataset_folder, final_dataset_name)):
        print("Creation of the main dataset")
        
        # Creation of the path to the final dataset folder if not existing
        if not os.path.exists(final_dataset_folder):
            os.makedirs(final_dataset_folder)
        
        df_final = import_and_process(data_folder, datasets_path, target_col, store_id_col, date_col, **kwargs)

        if save:
            export_dataframe(df_final, final_dataset_folder, final_dataset_name)
    
    # If the dataset already exists we can simply import it
    else:
        print("Dataset already exists, importing it")
        df_final = import_csv(final_dataset_folder, final_dataset_name)

    return df_final