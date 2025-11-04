import numpy as np
import matplotlib.pyplot as plt
import pickle
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})

def main():

    for flag_plot in ['AL_random', 'AL_smallest', 'AL_largest', 'AL_main']:
        path = fr"0_get_data/AL/Active_main/m_b_dicts"
        path_random = fr"0_get_data/AL\Active_random\m_b_dicts"
        path_largest = fr"0_get_data/AL\Active_largest\m_b_dicts"
        path_smallest = fr"0_get_data/AL\Active_smallest\m_b_dicts"

        fontsize = 30
        aoi_means = []
        aoi_stds = []
        aoi_means_random = []
        aoi_stds_random = []
        aoi_means_largest = []
        aoi_stds_largest = []
        aoi_means_smallest = []
        aoi_stds_smallest = []
        xtick_positions = np.linspace(0, 25, num=6, dtype=int)

        plt.figure(figsize=(8, 6))
        # Loop over 25 runs
        for run in range(25):

            with open(f'{path}/average_m_d_dict_{run}.pkl', 'rb') as chin_obj:
                params_list = pickle.load(chin_obj)

                params_list = params_list[f"{run}"]

            with open(f'{path_random}/average_m_d_dict_{run}.pkl', 'rb') as chin_obj:
                params_list_random = pickle.load(chin_obj)

                params_list_random = params_list_random[f"{run}"]

            with open(f'{path_largest}/average_m_d_dict_{run}.pkl', 'rb') as chin_obj:
                params_list_largest = pickle.load(chin_obj)

                params_list_largest = params_list_largest[f"{run}"]

            with open(f'{path_smallest}/average_m_d_dict_{run}.pkl', 'rb') as chin_obj:
                params_list_smallest = pickle.load(chin_obj)

                params_list_smallest = params_list_smallest[f"{run}"]

            mean_m, std_m, mean_b, \
            std_b, mean_aoi, std_aoi = params_list[0], params_list[1], params_list[2], \
                                       params_list[3], params_list[4], params_list[5],

            aoi_means.append(mean_aoi)
            aoi_stds.append(std_aoi)

            mean_m_random, std_m_random, mean_b_random, \
            std_b_random, mean_aoi_random, std_aoi_random = params_list_random[0], params_list_random[1], params_list_random[2], \
                                       params_list_random[3], params_list_random[4], params_list_random[5],

            aoi_means_random.append(mean_aoi_random)
            aoi_stds_random.append(std_aoi_random)

            mean_m_largest, std_m_largest, mean_b_largest, \
            std_b_largest, mean_aoi_largest, std_aoi_largest = params_list_largest[0], params_list_largest[1], params_list_largest[2], \
                                       params_list_largest[3], params_list_largest[4], params_list_largest[5],

            aoi_means_largest.append(mean_aoi_largest)
            aoi_stds_largest.append(std_aoi_largest)

            mean_m_smallest, std_m_smallest, mean_b_smallest, \
            std_b_smallest, mean_aoi_smallest, std_aoi_smallest = params_list_smallest[0], params_list_smallest[1], params_list_smallest[2], \
                                       params_list_smallest[3], params_list_smallest[4], params_list_smallest[5],

            aoi_means_smallest.append(mean_aoi_smallest)
            aoi_stds_smallest.append(std_aoi_smallest)

        aoi_means_smallest.append(0)
        aoi_stds_smallest.append(0)
        aoi_means.append(0)
        aoi_stds.append(0)
        aoi_means_random.append(0)
        aoi_stds_random.append(0)
        aoi_means_largest.append(0)
        aoi_stds_largest.append(0)

        pos1 = plt.gca().get_position()
        print(pos1)
        # Convert to numpy arrays for easier plotting
        aoi_means = np.array(aoi_means)
        aoi_stds = np.array(aoi_stds)

        aoi_means_random = np.array(aoi_means_random)
        aoi_stds_random = np.array(aoi_stds_random)

        aoi_means_largest = np.array(aoi_means_largest)
        aoi_stds_largest = np.array(aoi_stds_largest)

        aoi_means_smallest = np.array(aoi_means_smallest)
        aoi_stds_smallest = np.array(aoi_stds_smallest)
        # Plot AOI mean and std over runs
        runs = np.arange(0, 26)  # Runs 1 to 25

        if flag_plot == 'AL_random':
            plt.errorbar(runs, aoi_means, zorder=10, yerr=aoi_stds, fmt='-o', capsize=5, color='tab:blue',label='Active Learning')
        else:
            plt.errorbar(runs, aoi_means, zorder=10, yerr=aoi_stds, fmt='-o', capsize=5, color='tab:blue',
                         )

        if flag_plot == 'AL_random':
            plt.errorbar(runs, aoi_means_random, yerr=aoi_stds_random, fmt='-o', color='tab:orange', capsize=5, label='Random Order')
        elif flag_plot == 'AL_largest':
            plt.errorbar(runs, aoi_means_largest, yerr=aoi_stds_largest, fmt='-o', color='tab:green', capsize=5, label='Largest First')
        elif flag_plot == 'AL_smallest':
            plt.errorbar(runs, aoi_means_smallest, yerr=aoi_stds_smallest, color='tab:red', fmt='-o', capsize=5, label='Smallest First')

        plt.xlabel('Queries', fontsize=fontsize)
        plt.ylabel('Area between Curves', fontsize=fontsize)
        #plt.title('AOI Mean and Std over 25 Runs', fontsize=16)
        plt.grid(True, linestyle='--', linewidth=0.5)
        plt.xticks(xtick_positions)
        plt.ylim([-0.3, 1])
        plt.tick_params(axis='both', which='major', labelsize=fontsize-2)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        plt.legend(fontsize=fontsize-2, loc='upper right')
        plt.tight_layout()
        #plt.savefig(f'AOI_over_runs_{flag_plot}.png', dpi=300)
        #plt.savefig(f'AOI_over_runs_{flag_plot}.pdf')
        plt.show()


if __name__ == '__main__':
    print('We begin to run')
    main()