import os

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



if __name__ == "__main__":
    print(import_csv("data/", "oil.csv").head(10))