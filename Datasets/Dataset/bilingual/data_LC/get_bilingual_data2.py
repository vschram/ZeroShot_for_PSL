import numpy as np
import glob
import os
import pandas as pd


LANGS = ['en', 'id', 'jv', 'ms', 'ta', 'tl']
ds = np.array([10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000,
               2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 200000000, 500000000, 1000000000])


# Define folder path
for model in ["mBart","Transformer"]: #, "mBart", "Transformer"

    for metric in ["chrf", "bleu"]:
        if model == "mBart":
            path_model = r"mBart50\mBart50_ood_tseed2"
            LANG_dict = {'en': 16, 'id': 16, 'jv': 13, 'ms': 15, 'ta': 13, 'tl': 16}
        elif model == "Transformer":
            path_model = r"Transformer\Transformer_ood_tseed1"
            LANG_dict = {'en': 16, 'id': 15, 'jv': 12, 'ms': 14, 'ta': 12, 'tl': 15}
        folder_path = path_model

        end_filename = f"tgt_over_src_{model}_{metric}"

        #output path
        path = r"data"

        # Get all .npy files in the folder

        count_src = -1
        idx_pref = -1
        for idx, tgt in enumerate(LANGS):
            print("NEW tgt", tgt, "idx:", idx, "idx_pref", idx_pref)

            for src in LANGS:
                count_src += 1

                npy_files = glob.glob(os.path.join(folder_path, f"{metric}*{src}_{tgt}*.npy"))
                # Load all .npy files into a dictionary
                data = {os.path.basename(f): np.load(f) for f in npy_files}

                if tgt == src:
                    array = np.zeros(LANG_dict[src], dtype=np.float32)
                    if idx == 0:
                        with open(f"{path}/{end_filename}_pdata.txt", 'w') as file:
                            values = np.round(np.log(ds[0:LANG_dict[src]]), 2)
                            for i in values:
                                file.write(f"{idx+1},{i}\n")
                    else:
                        with open(f"{path}/{end_filename}_pdata.txt", 'a') as file:
                            values = np.round(np.log(ds[0:LANG_dict[src]]), 2)
                            for i in values:
                                file.write(f"{idx+1},{i}\n")
                else:
                    for filename, array in data.items():
                        array = array + 0.01
                #     print(f"Loaded {filename} with shape {array.shape}")
                #     print(array)

                if count_src == len(LANGS)-1:

                    with open(f"{path}/{end_filename}.txt", 'a') as file:
                        values = array.tolist()
                        values = np.round(values[0:LANG_dict[src]], 2).astype(float)
                        result_string = ', '.join(map(str, values))
                        for idx_, i in enumerate(result_string.split(',')):
                            if idx_ == len(result_string.split(',')) - 1:
                                file.write(f"{i}\n ")
                            else:
                                file.write(f"{i}, ")
                        count_src = -1

                else:

                    if idx == 0 and count_src == 0:
                        with open(f"{path}/{end_filename}.txt", 'w') as file:
                            values = array.tolist()
                            values = np.round(values[0:LANG_dict[src]], 2).astype(float)
                            result_string = ', '.join(map(str, values))
                            for idx_, i in enumerate(result_string.split(',')):
                                if idx_ == 0:
                                    file.write(f"{tgt}, {i}, ")
                                else:
                                    file.write(f"{i}, ")
                            idx_pref = idx

                    elif idx > idx_pref:
                        with open(f"{path}/{end_filename}.txt", 'a') as file:
                            values = array.tolist()
                            values = np.round(values[0:LANG_dict[src]], 2).astype(float)
                            result_string = ', '.join(map(str, values))
                            for idx_, i in enumerate(result_string.split(',')):
                                if idx_ == 0:
                                    file.write(f"{tgt}, {i}, ")
                                else:
                                    file.write(f"{i}, ")

                            idx_pref = idx

                    else:
                        with open(f"{path}/{end_filename}.txt", 'a') as file:
                            values = array.tolist()
                            values = np.round(values[0:LANG_dict[src]], 2).astype(float)
                            result_string = ', '.join(map(str, values))
                            for i in result_string.split(','):
                                file.write(f"{i}, ")

                print("src, len", src, len(values))

        # Read TXT file (assuming it's tab- or comma-separated)
        df = pd.read_csv(f"{path}/{end_filename}.txt", delimiter=",", decimal='.', header=None)  # Change delimiter if needed
        # Save as CSV
        df.to_csv(f"{path}/{end_filename}.csv", index=False, header=None)
        df = pd.read_csv(f"{path}/{end_filename}_pdata.txt", delimiter=",", decimal='.', header=None) # Change delimiter if needed
        # Save as CSV
        df.to_csv(f"{path}/{end_filename}_pdata.csv", index=False, header=None)