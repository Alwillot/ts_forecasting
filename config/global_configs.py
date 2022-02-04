import os

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
DATA_FOLDER_PATH = os.path.join(REPO_DIR, 'data/')

DATASETS = {
    "train": "train.csv",
    "oil": "oil.csv",
    "stores": "stores.csv",
    "transactions": "transactions.csv",
    "holidays": "holidays_events.csv",
    "test": "test.csv",
}

TARGET = 'sales'
STORE_ID = 'store_nbr'
DATE_COL = 'date'