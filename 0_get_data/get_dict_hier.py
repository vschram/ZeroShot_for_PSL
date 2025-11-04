import numpy as np
import os
import pickle
from itertools import combinations

folder = "data"


def find_common_key_groups(data, num):
    common_groups = {}

    keys = list(data.keys())  # Get all keys from the dictionary
    n = len(keys)

    # Iterate over all possible group sizes (from 2 to total number of keys)
    for group_size in range(2, n + 1):
        for group in combinations(keys, group_size):  # Generate all subsets of size `group_size`
            common_numbers = set(data[group[0]])  # Start with first key's numbers

            # Intersect with all other keys in the group
            for key in group[1:]:
                common_numbers &= set(data[key])

            # If at least 10 common numbers exist, save the group
            if len(common_numbers) >= num:
                common_groups[group] = common_numbers

    return common_groups




with open('dict_of_embd_layer.pkl', 'rb') as f:
    dict_of_embd_layer = pickle.load(f)
with open('dict_of_embd.pkl', 'rb') as f:
    dict_of_embd = pickle.load(f)
with open('dict_of_steps.pkl', 'rb') as f:
    dict_of_steps = pickle.load(f)
with open('dict_of_C.pkl', 'rb') as f:
    dict_of_C = pickle.load(f)


sorted_dict = {key: dict_of_embd[key] for key in sorted(dict_of_embd.keys(), key=int)}


sorted_keys = sorted(dict_of_embd.keys(), key=int)
point_of_interest = 500
start = 1
keys_of_interest = []
begin = 0
common_group = find_common_key_groups(sorted_dict, 6)

max_std = 0
max_len = 5

for i in common_group.keys():
    if len(i) == max_len:
        numbers = [int(s) for s in list(i)]
        std = np.std(numbers)
        print(numbers, std)
        keys_to_keep = i
        print(keys_to_keep)
keys_of_interest = [int(s) for s in list(keys_to_keep)]
common_layers = list(common_group[keys_to_keep])

import matplotlib.pyplot as plt

# Given values
# Remove rows and or columns if wanted for small dataset
# ------------------------------------
# common_layers.remove(10)
# common_layers.remove(32)
# numbers.remove(1024)
# keys_of_interest.remove(1024)
# numbers.remove(768)
# keys_of_interest.remove(768)
# ------------------------------------
n_embed_values = np.sort(numbers)
n_layer_values = np.sort(common_layers)

# Create subplots grid
fig, axes = plt.subplots(len(n_embed_values), len(n_layer_values), figsize=(15, 8), sharex=True, sharey=True)

for row, n_embed in enumerate(n_embed_values):
    for col, n_layer in enumerate(n_layer_values):
        ax = axes[row, col]  # Select subplot
        key = f"{n_embed}-{n_layer}"

        if key in dict_of_embd_layer and key in dict_of_steps:
            steps = dict_of_steps[key]
            val_loss = dict_of_embd_layer[key]
            print(key, val_loss)
            ax.plot(steps, val_loss, markersize=0.8, label=f"Loss")
            ax.set_title(f"n_embed={n_embed}, n_layer={n_layer}", fontsize=9)
            ax.grid(True, linestyle="--", alpha=0.6)


            # Annotate final loss value
            final_loss = val_loss[-1]
            ax.text(steps[-1], final_loss, f"{final_loss:.4f}",
                    fontsize=9, color='red', verticalalignment='bottom', horizontalalignment='right')

        if row == len(n_embed_values) - 1:
            ax.set_xlabel("Steps")

        if col == 0:
            ax.set_ylabel("Validation Loss")

# Adjust layout
plt.tight_layout()
#plt.suptitle("Validation Loss per (n_embed, n_layer)", fontsize=14, y=1.02)
plt.savefig('Dataset_large.png', format='png')
plt.savefig('Dataset_large.pdf', format='pdf')
plt.show()


# Create subplots grid
fig, ax = plt.subplots()

for row, n_embed in enumerate(n_embed_values):
    for col, n_layer in enumerate(n_layer_values):
        key = f"{n_embed}-{n_layer}"

        if key in dict_of_embd_layer and key in dict_of_C:
            C = dict_of_C[key]
            val_loss = dict_of_embd_layer[key]
            print(key, val_loss)
            C = np.array(C)
            val_loss = np.array(val_loss)

            # Choose 100 log-spaced indices
            num_points = 20
            log_indices = np.logspace(0, np.log10(len(C) - 1), num=num_points, dtype=int)

            # Remove duplicates (logspace might repeat indices)
            log_indices = np.unique(log_indices)

            # Sample data
            C_sampled = C[log_indices]
            val_loss_sampled = val_loss[log_indices]
            ax.plot(C_sampled, val_loss_sampled, color='blue', linewidth=0.8)
            # Annotate final loss value
            final_loss = val_loss[-1]
            # ax.text(C[-1], final_loss, f"{final_loss:.4f}",
            #         fontsize=9, color='red', verticalalignment='bottom', horizontalalignment='right')

ax.set_title(f"Validation Loss over Compute", fontsize=9)
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_xlabel("Compute")
ax.set_ylabel("Validation Loss")
ax.set_xscale('log')
ax.set_yscale('log')
# Adjust layout
plt.show()

common_layers = np.sort(common_layers)
keys_of_interest = np.sort(keys_of_interest)
print(common_layers)
begin = 0
for embed in keys_of_interest:
    for layer in common_layers:
        if begin == 0:
            print(f"{embed}-{layer}", set(dict_of_steps[f"{embed}-{layer}"]))
            common_steps = set(dict_of_steps[f"{embed}-{layer}"])
            begin = -1
        else:
            print(f"{embed}-{layer}", set(dict_of_steps[f"{embed}-{layer}"]))
            common_steps = common_steps & set(dict_of_steps[f"{embed}-{layer}"])
print(list(common_steps)[-1])
# Compute set intersection of their values
# common_values = set.intersection(*map(set, dict_of_embd.values()))

path = rf"{folder}"
if not os.path.exists(f"{path}"):
    os.makedirs(f"{path}")

first_embed = 1
begin = 0

log_start = np.ceil(np.log(1))
log_end = np.floor(np.log(list(common_steps)[-1]))
exponents = np.linspace(log_start, log_end, 100)
# for HMOGPLV
integer_exponents = np.unique(np.round(exponents).astype(int))
print(len(integer_exponents))

log_samples=[int(np.exp(i)) for i in integer_exponents]
# for bnn
# integer_exponents = np.unique(exponents)
# print(len(integer_exponents))
#
# log_samples=[(np.exp(i)) for i in integer_exponents]
# log_samples.append(list(common_steps)[-1]-1)
# breakpoint()

log_samples.append(list(common_steps)[-1]-1)

for n_embed in keys_of_interest:
    for n_layer in list(common_layers):
        if begin == 0:
            with open(f"{path}/layer_over_embd.txt", 'w') as file:
                # file.write(
                #     f"{n_embed}, {', '.join(map(str, dict_of_embd_layer[f'{n_embed}-{n_layer}'][1:list(common_steps)[-1]:point_of_interest]))}, ")
                # Extract corresponding values from the dictionary
                values = [dict_of_embd_layer[f'{n_embed}-{n_layer}'][i] for i in log_samples]
                # Join values into a string if needed
                result_string = ', '.join(map(str, values))
                file.write(f"{n_embed}, {result_string}, ")

                begin = -1
        elif begin == 2:
            with open(f"{path}/layer_over_embd.txt", 'a') as file:

                values = [dict_of_embd_layer[f'{n_embed}-{n_layer}'][i] for i in log_samples]

                # Join values into a string if needed
                result_string = ', '.join(map(str, values))

                file.write(f"{n_embed}, {result_string}, ")
                begin = -1
        else:
            with open(f"{path}/layer_over_embd.txt", 'a') as file:
                if n_layer != list(common_layers)[-1]:
                    print(len(dict_of_embd_layer[f'{n_embed}-{n_layer}']))
                    values = [dict_of_embd_layer[f'{n_embed}-{n_layer}'][i] for i in log_samples]

                    # Join values into a string if needed
                    result_string = ', '.join(map(str, values))
                    file.write(f"{result_string}, ")
                else:
                    values = [dict_of_embd_layer[f'{n_embed}-{n_layer}'][i] for i in log_samples]

                    # Join values into a string if needed
                    result_string = ', '.join(map(str, values))
                    file.write(f"{result_string}\n ")
                    begin = 2

try:
    begin = 0
    for idx, n_layer in enumerate(list(common_layers)):
        if idx == 0:
            for step in log_samples:
                if step == 1:
                    if begin == 0:
                        with open(f"{path}/layer_over_embd_pdata.txt", 'w') as file:
                            file.write(f"{idx+1}, {np.log(step)} \n")
                            begin = -1
                else:
                    with open(f"{path}/layer_over_embd_pdata.txt", 'a') as file:
                        file.write(f"{idx+1}, {np.log(step)} \n")
        else:
            for step in log_samples:
                if step == 1:
                    with open(f"{path}/layer_over_embd_pdata.txt", 'a') as file:
                        file.write(f"{idx+1}, {np.log(step)} \n")
                else:
                    with open(f"{path}/layer_over_embd_pdata.txt", 'a') as file:
                        file.write(f"{idx+1}, {np.log(step)} \n")

except RuntimeWarning:
    print(step)

# Read TXT and write to CSV
import pandas as pd
# Read the txt file (assuming space or tab-delimited)
df = pd.read_csv(f"{path}/layer_over_embd.txt", delimiter=',', header=None)
# Save to a csv file
df.to_csv(f"{path}/layer_over_embd.csv", index=False, header=False)
# Read the txt file (assuming space or tab-delimited)
df = pd.read_csv(f"{path}/layer_over_embd_pdata.txt", delimiter=',', header=None)
# Save to a csv file
df.to_csv(f"{path}/layer_over_embd_pdata.csv", index=False, header=False)
a=0

with open(f"{path}/MoreInfo.txt", 'a') as file:
    file.write(f"common layers: {common_layers}")
