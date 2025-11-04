import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib
import pickle
import argparse
import os
from pathlib import Path

time_vec =[1, 10, 20, 50, 100, 200, 500, 1000, 2000,
           5000, 10000, 20000, 50000, 100000, 200000,
           500000, 1000000, 2000000, 5000000, 10000000]


task = '2'
metric = "bleu"

new_folder_path = Path('data/same_source')
# Create a single new folder
new_folder_path.mkdir(parents=True, exist_ok=True)
new_folder_path = Path('data/same_target')
# Create a single new folder
new_folder_path.mkdir(parents=True, exist_ok=True)

root_path = "data/same_source"
count_LCs = 1
for source_of_interest in ['en', 'jv', 'ms', 'tl', 'ta', 'id']:
    path = f"{root_path}/source_{source_of_interest}_task_{task}_m2m100_metric_{metric}.txt"
    path_pdata = f"{root_path}/source_{source_of_interest}_task_{task}_m2m100_metric_{metric}_pdata.txt"
    path_pdata_log = f"{root_path}/source_{source_of_interest}_task_{task}_m2m100_metric_{metric}_pdata_log.txt"
    path_pdata_log_int = f"{root_path}/source_{source_of_interest}_task_{task}_m2m100_metric_{metric}_pdata_log_int.txt"

    path_wo0 = f"{root_path}/source_{source_of_interest}_task_{task}_m2m100_metric_{metric}_wo0.txt"
    path_pdata_log_wo0 = f"{root_path}/source_{source_of_interest}_task_{task}_m2m100_metric_{metric}_pdata_log_wo0.txt"

    f = open(path_pdata, "w")
    f.close()
    f = open(path_pdata_log, "w")
    f.close()
    f = open(path_pdata_log_int, "w")
    f.close()
    model_ = 1

    for model in ["175M", "615M", "big"]:

        filename = f"task_{task}_m2m100_{model}_metric_{metric}.pickle"

        file = open(filename, 'rb')
        LCs = pickle.load(file)

        if os.path.isfile(path):
            f = open(path, "a")
            f.write(f"\n{model}, ")
            f.close()
        else:
            f = open(path, "w")
            f.write(f"{model}, ")
            f.close()

        if task == '2':
            lp = ['en-id', 'en-jv', 'en-ms', 'en-ta', 'en-tl', 'id-ms', 'jv-ms', 'id-en', 'id-jv',  'id-ta',
                  'id-tl', 'jv-id', 'jv-en', 'jv-ta', 'jv-tl', 'ms-id', 'ms-jv', 'ms-en', 'ms-ta', 'ms-tl',
                  'ta-ms', 'ta-en', 'ta-tl', 'tl-id', 'tl-jv', 'tl-ms', 'tl-en', 'tl-ta', 'ta-id', 'ta-jv']
            lang_dict = {'en': 0, 'jv': 1, 'ms': 2, 'tl': 3, 'ta': 4, 'id': 5}
            full_dict = {'en': 'English', 'jv': 'Javanese', 'ms': 'Malay', 'tl': 'Tagaloc', 'ta': 'Tamil', 'id': 'Indonesian'}


root_path = "data/same_target"
count_LCs = 1
for target_of_interest in ['en', 'jv', 'ms', 'tl', 'ta', 'id']:
    print(target_of_interest)
    print("**********")
    path = f"{root_path}/target_{target_of_interest}_task_{task}_m2m100_metric_{metric}.txt"
    path_wo0 = f"{root_path}/target_{target_of_interest}_task_{task}_m2m100_metric_{metric}_wo0.txt"

    path_pdata = f"{root_path}/target_{target_of_interest}_task_{task}_m2m100_metric_{metric}_pdata.txt"
    path_pdata_log_int = f"{root_path}/target_{target_of_interest}_task_{task}_m2m100_metric_{metric}_pdata_log_int.txt"
    path_pdata_log_int = f"{root_path}/target_{target_of_interest}_task_{task}_m2m100_metric_{metric}_pdata_log.txt"
    path_pdata_log_wo0 = f"{root_path}/target_{target_of_interest}_task_{task}_m2m100_metric_{metric}_pdata_wo0.txt"

    f = open(path_pdata, "w")
    f.close()
    f = open(path_pdata_log, "w")
    f.close()
    f = open(path_pdata_log_int, "w")
    f.close()
    model_ = 1
    for model in ["175M", "615M", "big"]:

        filename = f"task_{task}_m2m100_{model}_metric_{metric}.pickle"

        file = open(filename, 'rb')
        LCs = pickle.load(file)

        if os.path.isfile(path):
            f = open(path, "a")
            f.write(f"\n{model}, ")
            f.close()
        else:
            f = open(path, "w")
            f.write(f"{model}, ")
            f.close()

        if task == '2':
            lp = ['en-id', 'en-jv', 'en-ms', 'en-ta', 'en-tl', 'id-ms', 'jv-ms', 'id-en', 'id-jv',  'id-ta',
                  'id-tl', 'jv-id', 'jv-en', 'jv-ta', 'jv-tl', 'ms-id', 'ms-jv', 'ms-en', 'ms-ta', 'ms-tl',
                  'ta-ms', 'ta-en', 'ta-tl', 'tl-id', 'tl-jv', 'tl-ms', 'tl-en', 'tl-ta', 'ta-id', 'ta-jv']
            lang_dict = {'en': 0, 'jv': 1, 'ms': 2, 'tl': 3, 'ta': 4, 'id': 5}
            full_dict = {'en': 'English', 'jv': 'Javanese', 'ms': 'Malay', 'tl': 'Tagaloc', 'ta': 'Tamil', 'id': 'Indonesian'}

        count_LCs = 1
        counter = 1
        counter_log = 1
        counter_log_int = 1
        counter_log_wo0 = 1

        for key, value in LCs.items():
            target = key.split("-")[1]
            if target == target_of_interest:
                print(key)

                f = open(path, "a")
                data = ', '.join(map(str, value))
                f.write(f"{data},")
                f.close()

                f = open(path_wo0, "a")
                data = ', '.join(map(str, value[1:]))
                f.write(f"{data},")
                f.close()


                if model_ == 1:
                    f = open(path_pdata, "a")
                    for i in time_vec:
                        f.write(f"{counter}, {i} \n")
                    f.close()



                    f = open(path_pdata_log, "a")
                    for i in time_vec:
                        f.write(f"{counter_log}, {np.log(i)} \n")
                    f.close()



                    f = open(path_pdata_log_wo0, "a")
                    for i in time_vec[1:]:
                        f.write(f"{counter_log_wo0}, {np.log(i)} \n")
                    f.close()


                    f = open(path_pdata_log_int, "a")
                    for i in time_vec:
                        f.write(f"{counter_log_int}, {np.int(np.round(np.log(i)))} \n")
                    f.close()

                    counter = counter + 1
                    counter_log = counter_log + 1
                    counter_log_wo0 = counter_log_wo0 + 1
                    counter_log_int = counter_log_int + 1
        model_ = 2


import os

# Specify the folder containing the files
folder_path = 'data/same_source/'

# Iterate through all files in the specified folder
for filename in os.listdir(folder_path):
    # Check if the file has a .txt extension
    if filename.endswith('.txt'):
        # Create the new filename with .csv extension
        new_filename = filename[:-4] + '.csv'

        # Get the full path for the old and new files
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')

# Specify the folder containing the files
folder_path = 'data/same_target/'

# Iterate through all files in the specified folder
for filename in os.listdir(folder_path):
    # Check if the file has a .txt extension
    if filename.endswith('.txt'):
        # Create the new filename with .csv extension
        new_filename = filename[:-4] + '.csv'

        # Get the full path for the old and new files
        old_file = os.path.join(folder_path, filename)
        new_file = os.path.join(folder_path, new_filename)

        # Rename the file
        os.rename(old_file, new_file)
        print(f'Renamed: {filename} -> {new_filename}')

