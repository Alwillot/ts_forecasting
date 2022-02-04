import config.global_configs as configs
import scripts.imports as imports

if __name__ == "__main__":
    data_folder = configs.DATA_FOLDER_PATH
    dataset = configs.DATASETS

    dict_final = imports.import_data_folder(data_folder, dataset)

    print(len(dict_final))

    for name, df in dict_final.items():
        print(name)
        print(df.head(5))
        print()
        print()
