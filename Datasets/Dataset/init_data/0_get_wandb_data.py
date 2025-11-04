import numpy as np
import matplotlib.pyplot as plt
import pickle
from itertools import combinations
from scipy import signal, stats
import matplotlib.ticker as ticker

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




with open('wandb_data/dict_of_embd_layer.pkl', 'rb') as f:
    dict_of_embd_layer = pickle.load(f)
with open('wandb_data/dict_of_embd.pkl', 'rb') as f:
    dict_of_embd = pickle.load(f)
with open('wandb_data/dict_of_steps.pkl', 'rb') as f:
    dict_of_steps = pickle.load(f)
with open('wandb_data/dict_of_C.pkl', 'rb') as f:
    dict_of_C = pickle.load(f)
with open('wandb_data/dict_for_SH.pkl', 'rb') as f:
    dict_for_SH = pickle.load(f)


sorted_dict = {key: dict_of_embd[key] for key in sorted(dict_of_embd.keys(), key=int)}

if get_data_online == 1:
    with open('wandb_data/dict_of_embd_layer.pkl', 'wb') as f:
        pickle.dump(dict_of_embd_layer, f)
    with open('wandb_data/dict_of_embd.pkl', 'wb') as f:
        pickle.dump(dict_of_embd, f)
    with open('wandb_data/dict_of_steps.pkl', 'wb') as f:
        pickle.dump(dict_of_steps, f)
    with open('wandb_data/dict_of_C.pkl', 'wb') as f:
        pickle.dump(dict_of_C, f)
    with open('wandb_data/dict_for_SH_large.pkl', 'wb') as f:
        pickle.dump(dict_for_SH, f)

sorted_keys = sorted(dict_of_embd.keys(), key=int)
point_of_interest = 1000
start = 1
keys_of_interest = []
begin = 0
common_group = find_common_key_groups(sorted_dict, 6)
max_std = 0
max_len = 5



# Create SL plot
fig_dims = (5, 5)
fig, ax = plt.subplots(1, 1, figsize=fig_dims)
fontsize = 18
fontsize_labels_n_ticks = fontsize-4
fontsize_legend = fontsize-4
info_arr = []
counter = 0
data_SL_plot = {}
min_loss = 100
max_loss = 0
max_val = 0
for key in dict_of_embd_layer.keys():
    if key in dict_of_C:
        C = dict_of_C[key]
        val_loss = dict_of_embd_layer[key]

        C = np.array(C)
        val_loss = np.array(val_loss)

        num = 2000
        for idx in range(num, len(val_loss)):
            if val_loss[idx] - val_loss[idx - num] > 2:
                val_loss = val_loss[:idx - num]
                C = C[:idx - num]
                break
        counter += 1
        yhat1 = signal.savgol_filter(val_loss.squeeze(), 11, 3)
        zs = stats.zscore(yhat1 - val_loss.squeeze())
        mask = np.abs(zs) < 3
        val_loss = yhat1.squeeze()[mask]
        C = C.squeeze()[mask]

        # Choose 100 log-spaced indices
        num_points = 20
        log_indices = np.logspace(0, np.log10(len(C) - 1), num=num_points, dtype=int)

        # Remove duplicates (logspace might repeat indices)
        log_indices = np.unique(log_indices)

        # Sample data

        C_sampled = C[log_indices]
        val_loss_sampled = val_loss[log_indices]
        if val_loss_sampled[-1] > max_loss:
            max_loss = val_loss_sampled[-1]
        if val_loss_sampled[-1] < min_loss:
            min_loss = val_loss_sampled[-1]
        if val_loss_sampled[0] > max_val:
            max_val = val_loss_sampled[0]
        info_arr.append([C_sampled, val_loss_sampled])
        ax.plot(C_sampled, val_loss_sampled, label='Learning Curves', color='green', linewidth=0.8)
        # Annotate final loss value
        final_loss = val_loss[-1]
        # ax.text(C[-1], final_loss, f"{final_loss:.4f}",
        #         fontsize=9, color='red', verticalalignment='bottom', horizontalalignment='right')

print(min_loss, max_loss, max_val)
ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(axis='both', which='major', labelsize=fontsize_labels_n_ticks)
#ax.tick_params(axis='both', which='minor', labelsize=fontsize_labels_n_ticks)
ax.grid(True, linestyle="--", alpha=0.6)
ax.set_xlabel("Compute", fontsize=fontsize)
ax.set_ylabel("Loss", fontsize=fontsize)



# Get a scaling law for your data
c_range_bounds = [np.log10(1e18), np.log10(1e20)]  # 5*10**19
C_range_T = np.logspace(c_range_bounds[0], c_range_bounds[1], 100)

best_loss = []

for C_i in C_range_T:  # 1) This is the C-range we are interested in (red scatter points)
    poss_loss1 = []
    for i in range(counter):  # 2) go through all N_i's
        # 4) C_\C_T_approx -> 3 get closest to the accurate loss calculation
        loss_diff = np.abs(info_arr[i][0] - C_i)
        # 5) Get the closest point
        loss_diff_min_idx = np.argmin(loss_diff)
        # 6) This is the min. loss when we have a certain N_i and want to use C_i for plotting
        poss_loss1.append([info_arr[i][1][loss_diff_min_idx]])
        # print(f"C_T_accurate: {info_arr[i, 0][loss_diff_min_idx]}, C_i: {C_i}, Loss: {info_arr[i, 1][loss_diff_min_idx]}")

    poss_loss1 = np.array(poss_loss1)
    # 7) find lowest loss when we consider all models (to find the compute efficient frontier)
    best_idx = np.argmin(poss_loss1[:, 0])
    best_loss.append(poss_loss1[best_idx])

    # print(f"C_i: {C_i} best_idx {best_idx} best loss {poss_loss1[best_idx]}")

best_loss_T = np.array(best_loss)
ax.scatter(C_range_T, best_loss_T[:, 0], label='Compute Efficient Frontier', color='black', s=2, zorder=400)

m_gt, b_gt = np.polyfit(np.log(C_range_T[:]), np.log(best_loss_T[:, 0]), 1)

slope_kaplan_new = m_gt
intercept_kaplan_new = b_gt
# print('C_T, Kaplan compute-loss form', m)
c_range_bounds_plot = [np.log10(1e15), np.log10(1e20)]  # 5*10**19
x_fit_gt = np.logspace(c_range_bounds_plot[0], c_range_bounds_plot[1], 1000)
y_fit_gt = np.exp(m_gt * np.log(x_fit_gt) + b_gt)

ax.plot(x_fit_gt, y_fit_gt, label=f'Scaling Law:\n $L_'+"{\log}"+r"^{SL}"+"(C)"+f'= {m_gt:.3f}'+"C_"+"{\log}"+f'+ {b_gt:.2f}$',
           color='blue', linewidth=1, zorder=8)

# ax.plot(x_fit_gt, y_fit_gt, label=f'Scaling Law\n $l^''{''log''}'f'(c) = {m_gt:.3f}c + {b_gt:.2f}$',
#            color='blue', linewidth=1, zorder=8)


#ax.xaxis.set_major_formatter(ScalarFormatter())
ax.get_xaxis().get_offset_text().set_fontsize(fontsize_labels_n_ticks)
handles, labels = ax.get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))  # Keep only the first occurrence

# Show legend with unique labels
ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right', fontsize=fontsize_legend)


ax.xaxis.set_minor_formatter(ticker.NullFormatter())  # Remove minor x-axis labels
ax.yaxis.set_minor_formatter(ticker.NullFormatter())
# Adjust layout
fig.tight_layout()
plt.savefig('wandb_data/SL_data_log.png')
plt.savefig('wandb_data/SL_data_log.pdf', dpi=300)
plt.show()
