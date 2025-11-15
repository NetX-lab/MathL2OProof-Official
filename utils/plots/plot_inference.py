# This script is used to plot the inference performances to demonstrate effectiveness of the CoordMathDNN for QP.
# Baselines: GD, Adam, LISTA

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import colors as mcolors
import math

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

markers = [
    'o',  # Circle
    's',  # Square
    '^',  # Triangle up
    'v',  # Triangle down
    'p',  # Pentagon
    'P',  # Plus (filled)
    '*',  # Star
    'h',  # Hexagon
    'H',  # Hexagon2
    'X',  # X (filled)
    'D',  # Diamond
    'd',  # Thin diamond
    '|',  # Vline
    '_',  # Hline
    '4'
]

colors = [
    "#000000",  # Black
    "#4472C4",
    "#ED7D31",
    "#A5A5A5",
    "#70AD47",
    "#FFC000",
    "#5B9BD5",
    "#C00000",
    "#7030A0",
    "#00B0F0",
    "#8F3F97",
    "#5F9E6E",
    "#FF6F61",
    "#B55A30",
    "#4BACC6",
    "#F79646",
    "#7F7F7F",
    "#B4C7E7",
    "#92D050",
    "#D99694",
    "#C0504D"
]


def plot_opt_processes_with_tune(files_1: list, files_2: list, legends_1: list, legends_2: list,
                                 name="compare_fixed", data_files="qp-rand", traj_len=1000,
                                 y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7]):
    """
    Plot the evolution of objective function values during optimization.

    Parameters:
        files_1: list of paths to results for deterministic methods
        files_2: list of lists, where each inner list contains paths to multiple runs for a stochastic method
        legends_1: list of legend names for deterministic methods
        legends_2: list of legend names for stochastic methods
        name: str, filename for the saved plot
        data_files: str or list, name(s) of data files to load
        traj_len: int, length of trajectory to plot
        y_ticks: list, y-axis tick values
    """
    plt.rcParams['axes.linewidth'] = 0.5
    myfont = {'size': 36, 'family': 'Helvetica'}
    myfont_legend = {'size': 26, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    # Convert data_files to list if it's a single string
    if isinstance(data_files, str):
        data_files = [data_files] * (len(files_1) + len(files_2))

    fg, (ax) = plt.subplots(1, 1, sharex=True, figsize=(12, 5))
    fg.subplots_adjust(hspace=0.05, right=0.7, top=0.95, bottom=0.15, left=0.1)

    x_ticks = [int(i) for i in range(0, traj_len+1, int(traj_len/5))]

    # Plot files_1 (deterministic methods)
    for i in range(len(legends_1)):
        y = np.loadtxt(f"{files_1[i]}/{data_files[i]}")
        y = y[:traj_len]
        c = colors[i]
        ax.plot(y, label=legends_1[i], linewidth=1,
                alpha=0.8, color=c, linestyle='dashed', marker=markers[0],
                markerfacecolor=c, markersize=6, markevery=0.1)

    # Function to plot distribution from a list of files
    def plot_distribution(ax, data_file, file_paths, legend, color, max_steps=traj_len):
        """
        Plot the mean with min/max bounds from multiple runs

        Parameters:
            ax: matplotlib axes
            data_file: name of the data file to load
            file_paths: list of file paths containing multiple runs
            legend: legend name for this distribution
            color: color for plotting
            max_steps: maximum number of steps to plot
        """
        if file_paths:
            # Load data from each file path
            ys = []
            for path in file_paths:
                try:
                    data = np.loadtxt(f"{path}/{data_file}")
                    ys.append(data[:max_steps])
                except Exception as e:
                    print(f"Error loading {path}/{data_file}: {e}")

            if not ys:
                return

            ys = np.array(ys)
            steps = np.arange(min(max_steps, ys.shape[1]))
            means = np.nanmean(ys, axis=0)[:len(steps)]
            mins = np.nanmin(ys, axis=0)[:len(steps)]
            maxs = np.nanmax(ys, axis=0)[:len(steps)]

            ax.plot(steps, means, label=legend, linewidth=1,
                    alpha=0.8, color=color, linestyle='-')

            ax.fill_between(
                steps,
                mins,  # Use min values instead of mean-std
                maxs,  # Use max values instead of mean+std
                alpha=0.2,
                color=color
            )

    # Plot files_2 (stochastic methods with distributions)
    for j in range(len(legends_2)):
        idx = len(legends_1) + j
        plot_distribution(ax, data_files[idx],
                          files_2[j], legends_2[j], colors[idx])

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    lines, labels = ax.get_legend_handles_labels()
    num_items = len(lines)
    ncol = math.ceil(num_items / 6)

    fg.legend(
        lines,
        labels,
        ncol=ncol,
        loc='upper right',
        bbox_to_anchor=(0.95, 0.9),
        prop=myfont_legend,
        columnspacing=0.8,
        labelspacing=0.4,
        borderaxespad=0.2,
        frameon=False
    )
    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    ax.set_yscale('log')  # Use log scale for y-axis
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$F(x_t)$', fontdict=myfont)
    plt.xlabel('Optimization Step $t$', fontdict=myfont)
    plt.xticks(x_ticks, fontsize=myfont['size'] // 3 * 2, rotation=45)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 3 * 2)

    # # Add zoomed-in inset for the first zoom_in_steps steps
    # zoom_in_steps = 300
    # ax_inset = inset_axes(ax, width="40%", height="40%",
    #                       loc='lower left', borderpad=8)

    # # Plot files_1 in inset
    # for i in range(len(legends_1)):
    #     y = np.loadtxt(f"{files_1[i]}/{data_files[i]}")
    #     y = y[:zoom_in_steps]  # Only the first zoom_in_steps steps
    #     c = colors[i]
    #     ax_inset.plot(y, label=legends_1[i], linewidth=1,
    #                   alpha=0.8, color=c, linestyle='dashed', marker=markers[0],
    #                   markerfacecolor=c, markersize=4, markevery=0.1)

    # # Plot files_2 distributions in inset
    # for j in range(len(legends_2)):
    #     idx = len(legends_1) + j
    #     plot_distribution(ax_inset, data_files[idx], files_2[j],
    #                       legends_2[j], colors[idx], max_steps=zoom_in_steps)

    # ax_inset.set_xlim(0, zoom_in_steps)
    # ax_inset.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    # ax_inset.set_yscale('log')  # Use log scale for inset y-axis
    # ax_inset.set_xticks(range(0, zoom_in_steps+1, int(zoom_in_steps/5)))
    # ax_inset.set_yticks(y_ticks)
    # ax_inset.tick_params(axis='both', which='major',
    #                      labelsize=myfont['size'] // 3 * 2)
    # ax_inset.grid(which='major', axis='both', linestyle='dashed')

    plt.tight_layout()
    plt.savefig(f"plots/{name}.pdf")


def plot_opt_processes(files_1: list, files_2: list, legends_1: list, legends_2: list, name="compare_fixed",
                       data_files=["qp-rand"], traj_len=1000,
                       y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7], myfontsize_legend=20, mybbox_to_anchor=(0.95, 0.9), colors=colors):
    plt.rcParams['axes.linewidth'] = 0.5
    myfont = {'size': 36, 'family': 'Helvetica'}
    myfont_legend = {'size': myfontsize_legend, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    fg, (ax) = plt.subplots(1, 1, sharex=True, figsize=(12, 5))
    fg.subplots_adjust(hspace=0.05, right=0.7, top=0.95, bottom=0.15, left=0.1)

    # color_map = ['b', 'tab:orange', 'g', 'r', 'm', 'tab:brown', 'gold', 'forestgreen',
    #              'tab:pink', 'c', 'y', 'tab:olive', 'C0', 'slategray', 'indigo', 'crimson', 'fuchsia']
    # colors = iter(color_map)

    x_ticks = [int(i) for i in range(0, traj_len+1, int(traj_len/5))]

    for i in range(len(legends_1)):
        y = np.loadtxt("results/" + files_1[i] + "/" + data_files[i])
        y = y[:traj_len]
        # sns.lineplot(data=y, palette=palette[i])
        # mean_loss = np.mean(y)
        c = colors[i]
        #  + '\t' + '{:.2e}'.format(mean_loss)
        ax.plot(y, label=legends_1[i], linewidth=1,
                alpha=0.8, color=c, linestyle='dashed', marker=markers[0],
                markerfacecolor=c, markersize=6, markevery=0.1)

    # if files_2 is not None:
    #     # colors = iter(color_map)
    #     for i in range(len(legends_2)):
    #         y = np.loadtxt("results/" + files_2[i] + "/" + data_files[i])
    #         y = y[:traj_len]
    #         mean_loss = np.mean(y)
    #         c = colors[i]
    #         #  + ' ' + '{:.2e}'.format(mean_loss)
    #         plt.plot(y, label=legends_2[i], linewidth=1,
    #                  alpha=0.8, color=c, linestyle='-', marker=markers1[1],
    #                  markerfacecolor=c, markersize=7, markevery=0.1)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    lines, labels = ax.get_legend_handles_labels()
    num_items = len(lines)
    ncol = math.ceil(num_items / 6)

    fg.legend(
        lines,
        labels,
        ncol=ncol,
        loc='upper right',
        bbox_to_anchor=mybbox_to_anchor,
        prop=myfont_legend,
        columnspacing=0.8,
        labelspacing=0.4,
        borderaxespad=0.2,
        frameon=False
    )
    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    # if not is_ood:
    #     ax.set_ylim(10**(-8), 10**1)
    # else:
    #     ax.set_ylim(10**(-8), 10**4)
    ax.set_yscale('log')
    ax.set_xscale('linear')
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$F(x_t)$', fontdict=myfont)
    plt.xlabel('Optimization Step $t$', fontdict=myfont)
    plt.xticks(x_ticks, fontsize=myfont['size'] // 3 * 2, rotation=45)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


# NOTE 1: Inference Comparison 3225
files_1 = ['inference/GD3225',
           #    'AGD',
           'inference/Adam3225',
           #    'AdamHD',
           #    'L2O-DM', 'L2O-RNNprop',
           'inference/SGDlr0.001-T100-rand100orth-3225-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0']

legends_1 = ['GD',
             #  'AGD',
             'Adam',
             #  'AdamHD',
             #  'L2O-DM', 'L2O-RNNprop',
             'Our L2O']

name = "results5_inference_comparison_T100_e100_3225"

# plot_opt_processes(files_1, None, legends_1, None, name, traj_len=5000,
#                    y_ticks=[5e2, 1e0, 1e-4, 1e-8, 1e-12])


# 2: Inference Comparison 512400
files_1 = ['inference/GD512400',
           #    'AGD',

           #    'AdamHD',
           #    'L2O-DM', 'L2O-RNNprop',
           'inference/SGDlr0.0000001-T100-rand500orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0',
           'inference/Adam/Adam512400',]

legends_1 = ['GD',
             #  'AGD',

             #  'AdamHD',
             #  'L2O-DM', 'L2O-RNNprop',
             'Our L2O',
             'Adam',]

data_files = ["qp-rand", "qp-rand", "qp-rand"]

name = "results5_inference_comparison_T100_e500_512400"

# plot_opt_processes(files_1, None, legends_1, None, name, data_files=data_files, traj_len=5000,
#                    y_ticks=[5e2, 1e0, 1e-4, 1e-8, 1e-12])

'''
NOTE Paper figure 8
'''
# 2.2: Inference Comparison 512400, Tune Adam

files_1 = ['results/inference/GD512400',

           #    'inference/SGDlr0.0000001-T100-rand500orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0'
           ]

legends_1 = ['GD',
             #  'Our L2O, LR $10^{-3}$',
             #  'Our L2O, LR $10^{-4}$',
             #  'Our L2O, LR $10^{-5}$',
             #  'Our L2O, LR $10^{-6}$',
             #  'Our L2O, LR $10^{-7}$',
             #  'Our L2O'
             ]


def list_direct_subdirectories(dir_path):

    subdirectories = []
    for item in os.listdir(dir_path):
        full_path = os.path.join(dir_path, item)
        if os.path.isdir(full_path):
            subdirectories.append("results/inference/Adam2/" + item)
    return subdirectories


files_2 = ['results/training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr0.001--optimizer-training-steps100--unroll-length100',
           'results/training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr0.0001--optimizer-training-steps100--unroll-length100',
           'results/training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr1e-05--optimizer-training-steps100--unroll-length100',
           'results/training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr1e-06--optimizer-training-steps100--unroll-length100',
           'results/training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr1e-07--optimizer-training-steps100--unroll-length100',]
files_2 = [files_2] + [list_direct_subdirectories('results/inference/Adam2')]

legends_2 = ['Our L2O', 'Adam']

# data_files = ["qp-rand", "qp-rand", "qp-rand"]

data_files = ["qp-rand", "qp-rand", "qp-rand"]

name = "results5_inference_comparison_T100_e500_512400_adam_tune"

plot_opt_processes_with_tune(files_1, files_2, legends_1, legends_2, name, data_files=data_files, traj_len=3000,
                             y_ticks=[1e3, 1e1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11])


# NOTE 3: Inference Comparison 512400 for LISTA trianing Fixed M inference with Fixed M training
files_1 = ['training/FixedM/GD512400-FixedM',
           'training/FixedM/LISTA-CPSS-Shared-FixM-lr0.001',
           'training/FixedM/LISTA-CPSS-NotShared-FixM-lr0.001',
           'training/FixedM/SGDlr0.001-T100-rand5000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0-FixedM']

legends_1 = ['GD',
             'LISTA-CPSS-Shared-FixM-lr0.001',
             'LISTA-CPSS-NotShared-FixM-lr0.001',
             'Our L2O']

name = "results6_inference_comparison_T100_e500_512400_FixedMwithFixedM"

# plot_opt_processes(files_1, None, legends_1, None, name, traj_len=5000,
#                    y_ticks=[5e2, 1e0, 1e-4, 1e-8, 1e-12], data_file="qp-rand-FixedM")


# 4: Inference Comparison 512400 for LISTA trianing Random M inference with Fixed M training
files_1 = ['training/FixedM/GD512400',
           'training/FixedM/LISTA-CPSS-Shared-FixM-lr0.001',
           'training/FixedM/LISTA-CPSS-NotShared-FixM-lr0.001',
           'training/FixedM/SGDlr0.001-T100-rand5000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0-FixedM']

legends_1 = ['GD',
             'LISTA-CPSS-Shared-FixM-lr0.001',
             'LISTA-CPSS-NotShared-FixM-lr0.001',
             'Our L2O']

name = "results6_inference_comparison_T100_e500_512400_RandomMwithFixedM"

# plot_opt_processes(files_1, None, legends_1, None, name, traj_len=5000,
#                    y_ticks=[5e2, 1e0, 1e-4, 1e-8, 1e-12])


'''
NOTE: Paper Figure 6b
'''
# Inference of LISTA-CPSS with Fixed M
files_1 = ['training/FixedM/LISTA-CPSS-WOnly-NotShared-lr0.0005-MInitW-UnrollTrain']

legends_1 = ['LISTA-CPSS']

name = "results6_train_inference_lista_cpss_T100_512400"

# plot_opt_processes(files_1, None, legends_1, None, name, traj_len=100,
#                    y_ticks=[1e27, 1e23, 1e19, 1e15, 1e11, 1e7, 1e3],
#                    myfontsize_legend=36, mybbox_to_anchor=(0.95, 0.8), colors=["#4472C4"])
