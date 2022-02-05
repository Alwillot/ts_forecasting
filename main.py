import os
from typing import Dict

import config.global_configs as configs
from scripts.imports import prepare_and_save

if __name__ == "__main__":
    ## CONFIGS IMPORT
    save_final_df = True

    # Path related configs
    data_folder_path = configs.DATA_FOLDER_PATH
    dataset_path = configs.DATASETS
    final_dataset_path = configs.FINAL_DATA_PATH
    final_dataset_filename = configs.FINAL_DATA_FILENAME
    
    # Columns related configs
    target = configs.TARGET
    store_id = configs.STORE_ID
    date_col = configs.DATE_COL


    ## DATA PREPARATION
    df_final = prepare_and_save(
        data_folder=data_folder_path,
        datasets_path=dataset_path,
        final_dataset_folder=final_dataset_path,
        final_dataset_name=final_dataset_filename,
        target_col=target,
        store_id_col=store_id,
        date_col=date_col,
        save=save_final_df
    )
    print("Import made successfully")
