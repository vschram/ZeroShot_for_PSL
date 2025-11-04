import subprocess
import pickle
import pandas as pd
import os
import shutil
from pathlib import Path


def main():

    init_query_dict = {}
    if os.path.exists("0_get_data/AL/Active_largest/"):
        shutil.rmtree("0_get_data/AL/Active_largest/")
    path_folder = Path("0_get_data/AL/Active_largest")
    path_folder.mkdir(parents=True, exist_ok=True)

    init_query_dict['d'] = [0, 1, 2, 3, 4]
    init_query_dict['r'] = [0, 0, 0, 0, 0]
    df = pd.DataFrame.from_dict(init_query_dict)
    file_path = fr"0_get_data\AL\Active_largest\query_largest.txt"
    df.to_csv(file_path, sep=',', index=False)
    runs_num = 2 # 10

    for queries in range(0, 3):

        subprocess.run(["python", "main.py", "--cfg", "Dataset_active_largest.yaml", "--query_num", f"{queries}", "--runs_num", f"{runs_num}"], check=True)

        path = fr"0_get_data/AL/Active_largest/uncertainty_dicts"
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

        dict_largest = {0: [4, 5], 1: [4, 4], 2: [4, 3], 3: [3, 5], 4: [2, 5], 5: [3, 4], 6: [4, 2], 7: [2, 4], 8: [1, 5], 9: [4, 1], 10: [3, 3], 11: [2, 3], 12: [1, 4], 13: [1, 3], 14: [3, 2], 15: [0, 5], 16: [2, 2], 17: [3, 1], 18: [2, 1], 19: [0, 4], 20: [1, 2], 21: [1, 1], 22: [0, 3], 23: [0, 2], 24: [0, 1]}

        next_d = dict_largest[queries][0]
        next_r = dict_largest[queries][1]

        embd_dict = {0: 512, 1: 768, 2: 960, 3: 1024, 4: 1600}
        layer_dict = {0: 8, 1: 10, 2: 12, 3: 24, 4: 32, 5: 48}

        print(f"Most uncertaint LC: \\ "
              f"for output(embed) {next_d} ({embd_dict[int(next_d)]}) \\ "
              f"and replica (layer) {next_r} ({layer_dict[int(next_r)]}) \\ "
              f"this is the largest key")

        embd_dict = {0: 512, 1: 768, 2: 960, 3: 1024, 4: 1600}
        layer_dict = {0: 8, 1: 10, 2: 12, 3: 24, 4: 32, 5: 48}

        print(f"Most uncertaint LC: for output(embed) {next_d} ({embd_dict[int(next_d)]}) and replica (layer) {next_r} ({layer_dict[int(next_r)]})")
        file_path = fr"0_get_data\AL\Active_largest\query_largest.txt"


        with open(file_path, "a") as f:
            f.write(f"{next_d},{next_r}\n")

        #calc SL
        subprocess.run(["python", "active_fit_sl.py", "--query_num", f"{queries}", "--runs_num", f"{runs_num}", "--al", "largest"], check=True)


def estimate_model_size_sim(d_model, num_layers, vocab_size=32e3):
    """You may narrow down the combinatorial space like this"""
    """ Parameters in the Chinchilla models. Unlike GPT they use relative positional embeddings. """

    #https://github.com/karpathy/nanoGPT/blob/master/scaling_laws.ipynb

    # token embeddings only
    # d_model is the embedding size
    embeddings = d_model * vocab_size
    # transformer blocks
    attention = 3*d_model**2 + 3*d_model  # weights and biases
    relative_pos = d_model**2 + 2*d_model  # relative keys, content bias, relative bias
    attproj = d_model**2 + d_model

    ffw_size = 4 * d_model
    ffw = d_model*ffw_size + ffw_size
    ffwproj = ffw_size*d_model + d_model

    layernorms = 2*2*d_model
    # dense
    ln_f = 2*d_model
    dense = d_model*vocab_size # note: no bias here
    # note: embeddings are not included in the param count!
    # note: Pierce code: embeddings are subtracted from params: chin_data_non[:,1] = chin_data_non[:,0]-chin_data_non[:,1] # could add this line to get non-embed params
    total_params = num_layers*(attention + relative_pos + attproj + ffw + ffwproj + layernorms) + ln_f + dense
    return total_params

if __name__ == '__main__':
    print('We begin to run')
    main()