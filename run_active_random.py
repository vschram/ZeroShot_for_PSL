import subprocess
import pickle
import os
import shutil
from pathlib import Path
import pandas as pd
import random

init_query_dict = {}
if os.path.exists("0_get_data/AL/Active_random/"):
    shutil.rmtree("0_get_data/AL/Active_random/")
path_folder = Path("0_get_data/AL/Active_random")
path_folder.mkdir(parents=True, exist_ok=True)

init_query_dict['d'] = [0, 1, 2, 3, 4]
init_query_dict['r'] = [0, 0, 0, 0, 0]
df = pd.DataFrame.from_dict(init_query_dict)
file_path = fr"0_get_data\AL\Active_random\query_random.txt"
df.to_csv(file_path, sep=',', index=False)
runs_num = 2  # 10

for queries in range(0, 3):


    subprocess.run(["python", "main.py", "--cfg", "Dataset_active_random.yaml", "--query_num", f"{queries}", "--runs_num",
                    f"{runs_num}"], check=True)


    path = fr"0_get_data/AL/Active_random/uncertainty_dicts"
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
    list_of_r_d = []
    for r_d, mnlpd_vec in all_pred_means_dict.items():
        list_of_r_d.append(r_d)

    random_string = random.choice(list_of_r_d)
    next_d = random_string.split('-')[1]
    next_r = random_string.split('-')[0]

    embd_dict = {0: 512, 1: 768, 2: 960, 3: 1024, 4: 1600}
    layer_dict = {0: 8, 1: 10, 2: 12, 3: 24, 4: 32, 5: 48}

    print(f"Most uncertaint LC: \\ "
          f"for output(embed) {next_d} ({embd_dict[int(next_d)]}) \\ "
          f"and replica (layer) {next_r} ({layer_dict[int(next_r)]}) \\ "
          f"this was picked randomly")

    embd_dict = {0: 512, 1: 768, 2: 960, 3: 1024, 4: 1600}
    layer_dict = {0: 8, 1: 10, 2: 12, 3: 24, 4: 32, 5: 48}

    print(f"Most uncertaint LC: for output(embed) {next_d} ({embd_dict[int(next_d)]}) and replica (layer) {next_r} ({layer_dict[int(next_r)]})")
    file_path = fr"0_get_data\AL\Active_random\query_random.txt"

    with open(file_path, "a") as f:
        f.write(f"{next_d},{next_r}\n")

    #calc SL
    subprocess.run(["python", "active_fit_sl.py", "--query_num", f"{queries}", "--runs_num", f"{runs_num}", "--al", "random"], check=True)
