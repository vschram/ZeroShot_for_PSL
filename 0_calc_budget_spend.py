import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import log10, floor
import pickle
from collections import Counter
from itertools import product
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})
path = r"Results\Dataset\Play_HMOGPLV_a_few_data_point" \
       r"\All-plot" \
       r"\HMOGPLV" \
       r"\HMOGPLVM" \
       r"\Embed"
fig_dims = (8, 6)
fontsize = 30
fig_sl, ax_sl = plt.subplots(1, 1, figsize=fig_dims)


def main():
    xtick_positions = np.linspace(0, 25, num=6, dtype=int)
    plt.figure(figsize=(8, 6))
    n_embed_vec = ['512', '768', '960', '1024', '1600']  #
    n_layer_vec = ['8', '10', '12', '24', '32', '48']  #
    combinations = [f"{embed}-{layer}" for embed, layer in product(n_embed_vec, n_layer_vec)]

    path_ = fr"0_get_data/AL"
    dict_r = {0: 8, 1: 10, 2: 12, 3: 24, 4: 32, 5: 48}
    dict_d = {0: 512, 1: 768, 2: 960, 3: 1024, 4: 1600}
    train = ['512-8', '768-8', '960-8', '1024-8', '1600-8']
    with open(r'0_get_data/dict_of_C.pkl', 'rb') as f:
        dict_of_C = pickle.load(f)
    # Choose 100 log-spaced indices
    num_points = 11
    log_indices = np.logspace(0, np.log10(19042 - 1), num=num_points, dtype=int)
    # Remove duplicates (logspace might repeat indices)
    log_indices = np.unique(log_indices)

    runs = np.arange(0, 26)  # Runs 1 to 25

    cost_dict = {}
    for scenario in ['al', 'random', 'largest', 'smallest']:

        if scenario == 'al':
            path = fr"{path_}/Active_main/query.txt"
        else:
            path = fr"{path_}/Active_{scenario}/query_{scenario}.txt"

        data = pd.read_csv(path, sep=",")

        queried_keys = []
        for i in range(5, len(data)):
            r_ = int(data.loc[i]['r'])
            d_ = int(data.loc[i]['d'])
            num_layer = dict_r[r_]
            num_embed = dict_d[d_]
            queried_keys.append(f'{num_embed}-{num_layer}')

        Compute_cost_train = 0
        for i in train:
            C_sampled = [dict_of_C[i][indx] for indx in log_indices]
            Compute_cost_train += C_sampled[-1]/10**15
        accum_cost = [Compute_cost_train]
        print(f'{scenario}', accum_cost)

        Compute_cost = 0
        print(f'{scenario}', queried_keys)
        cost_dict[f'{scenario}'] = queried_keys
        for i in queried_keys:
            C_sampled = [dict_of_C[i][indx] for indx in log_indices]
            Compute_cost += C_sampled[-1]/10**15

            accum_cost.append(C_sampled[-1]/10**15+accum_cost[-1])
        print(f'{scenario} acC', accum_cost[-1])
        print(f'{scenario}', Compute_cost)

        if scenario == 'al':
            name = 'Active Learning'
        elif scenario == 'random':
            name = 'Random Order'
        elif scenario == 'largest':
            name = 'Largest First'
        elif scenario == 'smallest':
            name = 'Smallest First'

        plt.plot(runs, accum_cost, label=f'{name}')

    same_elements = Counter(cost_dict['al']) == Counter(cost_dict[f'random'])
    print("Same strings regardless of order:", same_elements)
    # Convert to sets
    set1 = set(cost_dict['al'])
    set2 = set(cost_dict['random'])


    # Find differences
    only_in_list1 = set1 - set2
    only_in_list2 = set2 - set1

    # Print results
    if not only_in_list1 and not only_in_list2:
        print("Both lists have the same strings (ignoring order and duplicates).")
    else:
        if only_in_list1:
            print("Strings only in list1:", only_in_list1)
        if only_in_list2:
            print("Strings only in list2:", only_in_list2)

    plt.yscale('log')
    plt.xlabel('Queries', fontsize=fontsize)
    plt.ylabel('Compute Cost (PetaFLOPs)', fontsize=fontsize)
    #plt.title('AOI Mean and Std over 25 Runs', fontsize=16)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.xticks(xtick_positions)
    from matplotlib.transforms import Bbox
    bbox = Bbox.from_extents(0.125, 0.11, 0.9, 0.88)
    plt.gca().set_position(bbox)
    plt.tick_params(axis='both', which='major', labelsize=fontsize-2)
    plt.grid(axis='y', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.legend(fontsize=fontsize-5, loc='lower right')
    #plt.savefig(f'Budget_spend.png')
    #plt.savefig(f'Budget_spend.pdf')
    plt.show()


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


def flops_to_petaflops_scientific(compute_cost_flops):
    """
    Convert compute cost from FLOPs to PetaFLOPs and express in scientific notation.

    Parameters:
        compute_cost_flops (int or float): Compute cost in FLOPs.

    Returns:
        tuple: (coefficient, exponent) representing coefficient × 10^exponent PetaFLOPs
    """
    # Conversion to PetaFLOPs
    petaflops = compute_cost_flops / 1e15

    # Handle zero case
    if petaflops == 0:
        return (0, 0)

    # Calculate exponent and coefficient for scientific notation
    exponent = floor(log10(petaflops))
    coefficient = petaflops / (10 ** exponent)

    return (coefficient, exponent)


if __name__ == '__main__':
    print('We begin to run')
    main()
