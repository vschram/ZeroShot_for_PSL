import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import argparse
import pickle
from itertools import product
import matplotlib.ticker as ticker
from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union, polygonize
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})



def arg_parse():
    """
    Parsing arguments
    This function is used for pass the parameter from the yaml file.
    """
    parser = argparse.ArgumentParser(description="Configure files")
    parser.add_argument("--query_num", type=str, required=True, help="current number of query")
    parser.add_argument("--runs_num", type=str, required=True, help="current number of runs")
    parser.add_argument("--al", type=str, required=True, help="current experiment")
    args = parser.parse_args()
    return args


def main():
    # ---- setup configures ----
    args = arg_parse()
    query_num = args.query_num
    runs_num = int(args.runs_num)
    exp = args.al
    fig_dims = (9, 9)
    fontsize = 25
    fontsize_labels_n_ticks = 20
    fontsize_legend = 23
    fig_sl, ax_sl = plt.subplots(1, 1, figsize=fig_dims)

    plt.rc('legend', fontsize=fontsize_legend)

    case = 'Embed'
    scenario = case
    name_fig = "init"
    path = rf"Results\HMOGPLV\Embed_active_{exp}_{query_num}"
    n_embed_vec = ['512', '768', '960', '1024', '1600']  #
    n_layer_vec = ['8', '10', '12', '24', '32', '48']  #
    combinations = [f"{embed}-{layer}" for embed, layer in product(n_embed_vec, n_layer_vec)]

    # GT data train/test:
    with open(r'0_get_data/dict_of_C.pkl',
              'rb') as f:
        dict_of_C = pickle.load(f)
    with open(r'0_get_data/dict_of_embd_layer.pkl',
              'rb') as f:
        dict_of_y = pickle.load(f)


    info_arr = []

    # Choose 100 log-spaced indices
    num_points = 11
    log_indices = np.logspace(0, np.log10(19042 - 1), num=num_points, dtype=int)

    # Remove duplicates (logspace might repeat indices)
    log_indices = np.unique(log_indices)

    for i in combinations:
        C_sampled = [dict_of_C[i][indx] for indx in log_indices]
        val_loss_sampled = [dict_of_y[i][indx] for indx in log_indices]

        #ax_sl.plot(C_sampled, val_loss_sampled, 'mediumseagreen', linewidth=1, label="Training Data")
        info_arr.append([C_sampled, val_loss_sampled])

    c_range_bounds = [np.log10(1e18), np.log10(1e20)]  # 5*10**19
    C_range_T = np.logspace(c_range_bounds[0], c_range_bounds[1], 100)

    best_loss = []

    for C_i in C_range_T:  # 1) This is the C-range we are interested in (red scatter points)
        poss_loss1 = []
        for i in range(len(combinations)):  # 2) go through all N_i's
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
    #ax_sl.scatter(C_range_T, best_loss_T[:, 0], label='Compute Efficient Frontier', color='red', s=2, zorder=400)

    m_gt, b_gt = np.polyfit(np.log(C_range_T[:]), np.log(best_loss_T[:, 0]), 1)

    slope_kaplan_new = m_gt
    intercept_kaplan_new = b_gt
    # print('C_T, Kaplan compute-loss form', m)
    c_range_bounds_plot = [np.log10(1e15), np.log10(1e20)]  # 5*10**19
    x_fit_gt = np.logspace(c_range_bounds_plot[0], c_range_bounds_plot[1], 1000)
    y_fit_gt = np.exp(m_gt * np.log(x_fit_gt) + b_gt)
    y_fit_gt_orig = m_gt * np.log(x_fit_gt) + b_gt


    m_vec, b_vec = [], []
    aoi_vec = []
    info_arr = []
    for runs in np.arange(0, runs_num):
        filename = f"/{case}/Embed_{runs}/"
        fig_sl2, ax_sl2 = plt.subplots(1, 1, figsize=fig_dims)

        ax_sl2.plot(x_fit_gt, y_fit_gt, label=f'GT Scaling Law:\n $log (L) = {m_gt:.3f} log(C) + {b_gt:.2f}$',
                    color='blue', linestyle='-', linewidth=2, zorder=8)

        info_arr = []
        info_arr_gt = []

        for i in combinations:
            l = i.split('-')[1]
            e = i.split('-')[0]

            full_name = f"{path}/{filename}Test_pred_active_{exp}_{e}_{l}.npy"  # _{n_embed}_{n_layer}
            # Load the first matching file
            full_name_gt = f"{path}/{filename}Test_GT_active_{exp}_{e}_{l}.npy"  # _{n_embed}_{n_layer}


            try:

                data = np.load(full_name, allow_pickle=True)
                data_gt = np.load(full_name_gt, allow_pickle=True)

                C, y = data[0], data[1]
                C_gt, y_gt = data_gt[0], data_gt[1]

                info_arr.append([C, y])
                ax_sl2.plot(C, y,  '--', linewidth=1, color='red', label="Test Prediction")
                info_arr_gt.append([C_gt, y_gt])
                # ax_sl2.plot(C_gt, y_gt, color='blue', label="Test Ground Truth")

            except:
                C_sampled = [dict_of_C[i][indx] for indx in log_indices]
                val_loss_sampled = [dict_of_y[i][indx] for indx in log_indices]

                ax_sl2.plot(C_sampled, val_loss_sampled, 'mediumseagreen', linewidth=1, label="Training Data")
                info_arr.append([C_sampled, val_loss_sampled])

        best_loss = []

        for C_i in C_range_T:  # 1) This is the C-range we are interested in (red scatter points)
            poss_loss1 = []
            for i in range(len(combinations)):  # 2) go through all N_i's
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
        ax_sl2.scatter(C_range_T, best_loss_T[:, 0], label='Compute efficient frontier', color='black', s=4, zorder=400)

        m, b = np.polyfit(np.log(C_range_T[:]), np.log(best_loss_T[:, 0]), 1)
        m_vec.append(m)
        b_vec.append(b)
        slope_kaplan_new = m
        intercept_kaplan_new = b
        # print('C_T, Kaplan compute-loss form', m)
        c_range_bounds_plot = [np.log10(1e15), np.log10(1e20)]  # 5*10**19
        x_fit = np.logspace(c_range_bounds_plot[0], c_range_bounds_plot[1], 1000)
        y_fit = np.exp(m * np.log(x_fit) + b)

        ax_sl2.plot(x_fit, y_fit, label=f'Predicted SL:\n $log (L) = {m:.3f} log(C) + {b:.2f}$',
                    color='blue', linestyle='--', linewidth=2, zorder=8)

        # ax_sl2.set_title(f"Validation Loss over Compute", fontsize=9)
        ax_sl2.grid(which='major', linestyle='--', linewidth='0.5', color='black')

        ax_sl2.set_xlabel("C", fontsize=fontsize)
        ax_sl2.set_ylabel("Loss", fontsize=fontsize)
        ax_sl2.set_xscale('log')
        ax_sl2.set_yscale('log')

        handles, labels = ax_sl2.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))  # Keep only the first occurrence

        # Show legend with unique labels
        ax_sl2.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')
        # {path}/{filename}/",
        # ax_sl2.set_ylim([2, 1 * 10 ** 3])
        # ax_sl2.set_xlim([1e4, 1e20])
        ax_sl2.set_ylim([2.5, 1 * 10 ** 1])
        ax_sl2.set_xlim([1e15, 1e20])
        ax_sl2.xaxis.set_minor_formatter(ticker.NullFormatter())  # Remove minor x-axis labels
        ax_sl2.yaxis.set_minor_formatter(ticker.NullFormatter())  # Remove minor y-axis labels
        ax_sl2.tick_params(axis='both', which='major', labelsize=fontsize)
        fig_sl2.tight_layout()

        path_to_slfig = fr'0_get_data\AL\Active_{exp}\SL_fit'
        path_folder = Path(path_to_slfig)
        path_folder.mkdir(parents=True, exist_ok=True)

        fig_sl2.savefig(os.path.join(path_to_slfig, f'{query_num}_{scenario}_{runs}_{name_fig}.png'), format='png',
                        bbox_inches='tight')
        fig_sl2.savefig(os.path.join(path_to_slfig, f'{query_num}_{scenario}_{runs}_{name_fig}.pdf'), format='pdf',
                        bbox_inches='tight')

        # calc AOI
        xy_gt = np.c_[np.log(x_fit), y_fit_gt_orig]
        y_new = m * np.log(x_fit) + b
        xy_new = np.c_[np.log(x_fit), y_new]

        try:
            aoi = calc_area_of_interest(xy_gt, xy_new)
        except:
            aoi = 0

        aoi_vec.append(aoi)

    average_m_d_dict = {}
    mean_b = np.mean(b_vec)
    std_b = np.std(b_vec)
    mean_m = np.mean(m_vec)
    std_m = np.std(m_vec)

    mean_aoi = np.mean(aoi_vec)
    std_aoi = np.std(aoi_vec)

    average_m_d_dict[query_num] = [mean_m, std_m, mean_b, std_b, mean_aoi, std_aoi]

    if exp == "main":
        path_ = r'0_get_data\AL\Active_main\m_b_dicts'
    elif exp == "random":
        path_ = r'0_get_data\AL\Active_random\m_b_dicts'
    elif exp == "largest":
        path_ = r'0_get_data\AL\Active_largest\m_b_dicts'
    elif exp == "smallest":
        path_ = r'0_get_data\AL\Active_smallest\m_b_dicts'

    path_folder = Path(path_)
    path_folder.mkdir(parents=True, exist_ok=True)

    with open(f'{path_}/average_m_d_dict_{query_num}.pkl', 'wb') as f:
        pickle.dump(average_m_d_dict, f)

    print(f"b: {np.round(mean_b, 4)}+-{np.round(std_b, 4)}")
    print(f"m: {np.round(mean_m, 4)}+-{np.round(std_m, 4)}")
    print(f"aoi: {np.round(mean_aoi, 4)}+-{np.round(std_aoi, 4)}")


def calc_area_of_interest(xy, xy_new):
    polygon_points = []  # creates a empty list where we will append the points to create the polygon

    for xyvalue in xy:
        polygon_points.append([xyvalue[0], xyvalue[1]])  # append all xy points for curve 1

    for xyvalue in xy_new[::-1]:
        polygon_points.append([xyvalue[0], xyvalue[
            1]])  # append all xy points for curve 2 in the reverse order (from last point to first point)

    for xyvalue in xy[0:1]:
        polygon_points.append(
            [xyvalue[0], xyvalue[1]])  # append the first point in curve 1 again, to it "closes" the polygon

    polygon = Polygon(polygon_points)
    area = polygon.area

    x, y = polygon.exterior.xy
    # original data
    ls = LineString(np.c_[x, y])
    # closed, non-simple
    lr = LineString(ls.coords[:] + ls.coords[0:1])
    lr.is_simple  # False
    mls = unary_union(lr)
    mls.geom_type  # MultiLineString'

    Area_cal = []

    for polygon in polygonize(mls):
        Area_cal.append(polygon.area)
        Area_poly = (np.asarray(Area_cal).sum())
    # print(Area_poly)
    aoi = Area_poly

    return aoi

if __name__ == '__main__':
    print('We begin to run')
    main()