import subprocess
import pickle
import numpy as np
from pathlib import Path
import shutil
import os
import pandas as pd

init_query_dict = {}
if os.path.exists("0_get_data/AL/Active_main"):
    shutil.rmtree("0_get_data/AL/Active_main")
path_folder = Path("0_get_data/AL/Active_main")
path_folder.mkdir(parents=True, exist_ok=True)

init_query_dict['d'] = [0, 1, 2, 3, 4]
init_query_dict['r'] = [0, 0, 0, 0, 0]
df = pd.DataFrame.from_dict(init_query_dict)
file_path = fr"0_get_data\AL\Active_main\query.txt"
df.to_csv(file_path, sep=',', index=False)
runs_num = 10 # Number of runs to average over

for queries in range(0, 25): #The number of queries

    subprocess.run(["python", "main.py", "--cfg", "Dataset_active.yaml", "--query_num", f"{queries}", "--runs_num",
                    f"{runs_num}"], check=True)

    path = fr"0_get_data/AL/Active_main/uncertainty_dicts"
    all_pred_means_dict = {}

    for run_i in range(runs_num):
        with open(f'{path}/uncertainty_dicts_{run_i}.pkl', 'rb') as f:
            uncertainty_dicts = pickle.load(f)

        #print(uncertainty_dicts)

        if run_i == 0:
            for key, values in uncertainty_dicts.items():
                all_pred_means_dict[key] = []

        for r_d, pred_means in uncertainty_dicts.items():
            d = r_d.split('-')[1]
            r = r_d.split('-')[0]
            all_pred_means_dict[r_d].append(pred_means)

    average_pred_var_dict = {}
    pred_var_largest = -1000
    for r_d, pred_means_vec in all_pred_means_dict.items():

        pred_var_avg = np.var(pred_means_vec, axis=0)
        pred_var_avg_mean = np.mean(pred_var_avg)
        average_pred_var_dict[r_d] = pred_var_avg_mean
        r = r_d.split('-')[0]
        d = r_d.split('-')[1]

        if pred_var_largest < pred_var_avg_mean:
            pred_var_largest = pred_var_avg_mean
            next_r = r
            next_d = d

    embd_dict = {0: 512, 1: 768, 2: 960, 3: 1024, 4: 1600}
    layer_dict = {0: 8, 1: 10, 2: 12, 3: 24, 4: 32, 5: 48}

    print(f"Most uncertaint LC: \\ "
          f"for output(embed) {next_d} ({embd_dict[int(next_d)]}) \\ "
          f"and replica (layer) {next_r} ({layer_dict[int(next_r)]}) \\ "
          f"and uncertainty {pred_var_largest}")

    file_path = fr"0_get_data\AL\Active_main\query.txt"

    with open(file_path, "a") as f:
        f.write(f"{next_d},{next_r}\n")


    #calc SL
    subprocess.run(["python", "active_fit_sl.py", "--query_num", f"{queries}", "--runs_num", f"{runs_num}", "--al", "main"],  check=True)
