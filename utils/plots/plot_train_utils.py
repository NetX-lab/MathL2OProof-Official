import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib import colors as mcolors
import math

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

# colors = ['b', 'tab:orange', 'g', 'y', 'm', 'tab:brown',
#           'tab:pink', 'c', 'r', 'tab:olive', 'C0']

# colors = ['b', 'tab:orange', 'g', 'r', 'm', 'tab:brown', 'gold', 'forestgreen',
#           'tab:pink', 'c', 'y', 'tab:olive', 'C0', 'slategray', 'indigo', 'crimson', 'fuchsia']
# colors = [
#     "#000000",  # Black
#     "#E69F00",  # Orange
#     "#56B4E9",  # Sky blue
#     "#009E73",  # Bluish green
#     "#F0E442",  # Yellow
#     "#D55E00",  # Vermillion
#     "#CC79A7",   # Reddish purple
#     "#a6761d",  # 棕褐
#     "#0072B2",  # Blue
#     "red",
#     'tab:olive', 'C0', 'slategray', 'indigo', 'crimson', 'fuchsia'
# ]

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


def plot_training_last_obj(x_tickes, files_1: list, legends_1: list, name="compare_fixed",
                           _loc='upper right',
                           y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7],
                           x_ticks=[1e0, 1e1, 1e2, 1e3, 1.6*1e3], t='100', gd=True):
    '''
    Plot the Last Objective of Math-L2O with training
    '''
    plt.rcParams['axes.linewidth'] = 0.5
    myfont_y = {'size': 36, 'family': 'Helvetica'}
    myfont_x = {'size': 36, 'family': 'Helvetica'}
    myfont_legend = {'size': 26, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    fg, (ax) = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
    fg.subplots_adjust(hspace=0.05, right=0.7, top=0.95, bottom=0.15, left=0.1)

    gd_baseline = [np.loadtxt("results/" + files_1[0])[-1]] * len(files_1)
    c = colors[0]
    ax.plot(x_tickes, gd_baseline, label=legends_1[0], linewidth=2,
            alpha=0.8, color=c, linestyle='dashed')

    L2O_last = []

    for i in range(len(files_1)):
        L2O_last.append(np.loadtxt("results/" + files_1[i])[-1])

    c = colors[1]
    ax.plot(x_tickes, L2O_last, label=legends_1[1], linewidth=2,
            alpha=0.8, color=c, linestyle='solid', marker=markers[1],
            markerfacecolor=c, markersize=6, markevery=0.01)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    lines, labels = ax.get_legend_handles_labels()
    fg.legend(lines, labels, loc='center right',
              prop=myfont_legend,
              bbox_to_anchor=(0.95, 0.7), borderaxespad=0.)

    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    ax.set_yscale('log')
    ax.set_xscale('linear')
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$F(X_{' + t + '}^k)$', fontdict=myfont_y)
    plt.xlabel('Training Iteration $k$', fontdict=myfont_x)
    plt.xticks(x_ticks, fontsize=myfont_y['size'] // 3 * 2, rotation=45)
    plt.yticks(y_ticks, fontsize=myfont_y['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def plot_training_last_obj_gd_vs_others(train_iters, gd_file, files_list: list, legends_1: list, name="compare_lista_our_fixed",
                                        _loc='upper right',
                                        y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7],
                                        x_ticks=[1e0, 1e1, 1e2, 1e3, 1.6*1e3], myfontsize_legend=36):
    '''
    Plot the Last Objective in training, use GD as baseline.
    '''
    plt.rcParams['axes.linewidth'] = 0.5
    myfont_y = {'size': 36, 'family': 'Helvetica'}
    myfont_x = {'size': 36, 'family': 'Helvetica'}
    myfont_legend = {'size': myfontsize_legend, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    fg, (ax) = plt.subplots(1, 1, sharex=True, figsize=(12, 5))
    fg.subplots_adjust(hspace=0.05, right=0.7, top=0.95, bottom=0.15, left=0.1)

    gd_baseline = [np.loadtxt(gd_file)[-1]] * len(train_iters)
    c = colors[0]
    ax.plot(train_iters, gd_baseline, label=legends_1[0], linewidth=2,
            alpha=0.8, color=c, linestyle='dashed')

    for i in range(len(files_list)):
        last_data = [np.loadtxt(files_list[i][j])[-1]
                     for j in range(len(files_list[i]))]
        c = colors[i+1]
        ax.plot(train_iters, last_data, label=legends_1[i+1], linewidth=2,
                alpha=0.8, color=c, linestyle='solid', marker=markers[i+1],
                markerfacecolor=c, markersize=6, markevery=0.05)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    lines, labels = ax.get_legend_handles_labels()
    fg.legend(lines, labels, loc='center right',
              prop=myfont_legend,
              bbox_to_anchor=(0.95, 0.7), borderaxespad=0.)

    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    ax.set_yscale('log')
    ax.set_xscale('linear')
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$F(X_{100}^k)$', fontdict=myfont_y)
    plt.xlabel('Training Iteration $k$', fontdict=myfont_x)
    plt.xticks(x_ticks, fontsize=myfont_y['size'] // 3 * 2, rotation=45)
    plt.yticks(y_ticks, fontsize=myfont_y['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def plot_training_last_obj_baseline_vs_ours(train_iter, unroll_len, gd_file, files_list: list, legends_1: list, name="compare_lista_our_fixed",
                                            y_ticks=[1e1, 1e-2,
                                                     1e-4, 1e-6, 1e-7],
                                            x_ticks=list(range(5, 105, 5))):
    '''
    Plot the Last Objective in training, Ours v.s. Math-L2O
    '''
    plt.rcParams['axes.linewidth'] = 0.5
    myfont_y = {'size': 36, 'family': 'Helvetica'}
    myfont_x = {'size': 36, 'family': 'Helvetica'}
    myfont_legend = {'size': 20, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    fg, (ax) = plt.subplots(1, 1, sharex=True, figsize=(10, 4))
    fg.subplots_adjust(hspace=0.05, right=0.7, top=0.95, bottom=0.15, left=0.1)

    gd_traj = np.loadtxt("results/" + gd_file)
    gd_baseline = [gd_traj[i] for i in unroll_len]
    c = colors[0]
    ax.plot(unroll_len, gd_baseline, label=legends_1[0], linewidth=2,
            alpha=0.8, color=c, linestyle='dashed')

    def read_data_if_exist(file_list):
        last_data_by_unroll_len = []
        for f in file_list:
            # print(f)
            if os.path.exists(f):
                last_data_by_unroll_len.append(np.loadtxt(f)[-1])
        return last_data_by_unroll_len

    for i in range(len(files_list)):
        last_data_by_unroll_len = read_data_if_exist(files_list[i])
        if len(last_data_by_unroll_len) == 0:
            continue
        c = colors[i+1]
        if len(last_data_by_unroll_len) == 1:
            ax.scatter(unroll_len[:len(last_data_by_unroll_len)], last_data_by_unroll_len,
                       label=legends_1[i+1], color=c, marker=markers[i+1], s=100)
        else:
            ax.plot(unroll_len[:len(last_data_by_unroll_len)], last_data_by_unroll_len, label=legends_1[i+1], linewidth=2,
                    alpha=0.8, color=c, linestyle='solid', marker=markers[i+1], markerfacecolor=c, markersize=6, markevery=0.05)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    lines, labels = ax.get_legend_handles_labels()
    num_items = len(lines)
    ncol = math.ceil(num_items / 3)

    fg.legend(
        lines,
        labels,
        ncol=ncol,
        loc='upper right',
        bbox_to_anchor=(0.98, 0.98),
        prop=myfont_legend,
        columnspacing=0.8,
        labelspacing=0.4,
        borderaxespad=0.2,
        frameon=False
    )
    # fg.legend(lines, labels, loc='upper right',
    #           prop=myfont_legend,
    #           bbox_to_anchor=(0.95, 0.9), borderaxespad=0.)

    ax.set_ylim(0.98*y_ticks[-1], 1.02*y_ticks[0])
    ax.set_yscale('log')
    ax.set_xscale('linear')
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    # ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$F(X_T^{' + str(train_iter) + '})$', fontdict=myfont_y)
    plt.xlabel('Optimization Step $T$', fontdict=myfont_x)
    plt.xticks(x_ticks, fontsize=myfont_y['size'] // 3 * 2, rotation=45)
    plt.yticks(y_ticks, fontsize=myfont_y['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def plot_training_GD_process(files_1: list, legends_1: list, name="compare_fixed",
                             _loc='upper right',
                             y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7],
                             x_ticks=[1e0, 1e1, 1e2, 1e3, 1.6*1e3], t='100'):
    '''
    Plot the GD trajectory of Math-L2O with training
    '''
    plt.rcParams['axes.linewidth'] = 0.5
    myfont_y = {'size': 36, 'family': 'Helvetica'}
    myfont_x = {'size': 36, 'family': 'Helvetica'}
    myfont_legend = {'size': 16, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    # palette = sns.color_palette()
    # print(palette)
    # fg = plt.figure(figsize=(10, 5), right=0.7, top=0.95, bottom=0.15, left=0.1)
    # fg.subplots_adjust(hspace=0.05)
    fg, (ax) = plt.subplots(1, 1, sharex=True, figsize=(12, 5))
    fg.subplots_adjust(hspace=0.05, right=0.7, top=0.95, bottom=0.15, left=0.1)
    # ax = plt.gca()

    # if not is_ood else 200
    #  if not is_ood else [1e2, 1e-2, 1e-4, 1e-6, 1e-7]

    # if not is_ood else [1e5, 1e4, 1e3, 1e2]
    # data_file = "losses-rand" if not is_ood else "losses-rand-OOD"
    for i in range(len(legends_1)):
        y = np.loadtxt("results/" + files_1[i])
        c = colors[i]
        ax.plot(y, label=legends_1[i], linewidth=2,
                alpha=0.8, color=c, linestyle='dashed', marker=markers[i],
                markerfacecolor=c, markersize=3, markevery=0.1)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    # plt.legend(loc=_loc,
    #               bbox_to_anchor=(0.68, 0.7),
    #            prop=myfont_legend,
    #               ncols=4,
    #            frameon=1, framealpha=0.5)

    lines, labels = ax.get_legend_handles_labels()
    fg.legend(lines, labels, loc='center left',
              prop=myfont_legend,
              ncols=2,
              bbox_to_anchor=(0.245, 0.79), borderaxespad=0.)

    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    # if not is_ood:
    #     ax.set_ylim(10**(-8), 10**1)
    # else:
    #     ax.set_ylim(10**(-8), 10**4)
    ax.set_yscale('log')
    ax.set_xscale('linear')
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$F(X_{' + t + '})$', fontdict=myfont_y)
    plt.xlabel('GD Iteration $t$', fontdict=myfont_x)
    plt.xticks(x_ticks, fontsize=myfont_y['size'] // 3 * 2)
    plt.yticks(y_ticks, fontsize=myfont_y['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def plot_opt_process(files_1: list, legends_1: list, 
                             iters: list,
                             name="compare_fixed",
                             _loc='upper right',
                             y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7],
                             x_ticks=[1e0, 1e1, 1e2, 1e3, 1.6*1e3], t='100'):
    '''
    Plot the GD trajectory of Math-L2O with training
    '''
    plt.rcParams['axes.linewidth'] = 0.5
    myfont_y = {'size': 40, 'family': 'Helvetica'}
    myfont_x = {'size': 40, 'family': 'Helvetica'}
    myfont_legend = {'size': 30, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    # palette = sns.color_palette()
    # print(palette)
    # fg = plt.figure(figsize=(10, 5), right=0.7, top=0.95, bottom=0.15, left=0.1)
    # fg.subplots_adjust(hspace=0.05)
    fg, (ax) = plt.subplots(1, 1, sharex=True, figsize=(12, 5))
    fg.subplots_adjust(hspace=0.05, right=0.7, top=0.95, bottom=0.15, left=0.1)
    # ax = plt.gca()

    # if not is_ood else 200
    #  if not is_ood else [1e2, 1e-2, 1e-4, 1e-6, 1e-7]

    # if not is_ood else [1e5, 1e4, 1e3, 1e2]
    # data_file = "losses-rand" if not is_ood else "losses-rand-OOD"
    for i in range(len(legends_1)):
        y = np.loadtxt("results/" + files_1[i])[:iters[i]]
        c = colors[i]
        ax.plot(y, label=legends_1[i], linewidth=2,
                alpha=0.8, color=c, linestyle='dashed', marker=markers[i],
                markerfacecolor=c, markersize=3, markevery=0.1)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    # plt.legend(loc=_loc,
    #               bbox_to_anchor=(0.68, 0.7),
    #            prop=myfont_legend,
    #               ncols=4,
    #            frameon=1, framealpha=0.5)

    lines, labels = ax.get_legend_handles_labels()
    fg.legend(lines, labels, loc='center left',
              prop=myfont_legend,
              ncols=1,
              bbox_to_anchor=(0.245, 0.7), borderaxespad=0.)

    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    # if not is_ood:
    #     ax.set_ylim(10**(-8), 10**1)
    # else:
    #     ax.set_ylim(10**(-8), 10**4)
    ax.set_yscale('log')
    ax.set_xscale('linear')
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('$F(X_{t})$', fontdict=myfont_y)
    plt.xlabel('Optimization Step $t$', fontdict=myfont_x)
    plt.xticks(x_ticks, fontsize=myfont_y['size'] // 3 * 2)
    plt.yticks(y_ticks, fontsize=myfont_y['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def plot_training_gain(iters, ratios, name="ratio",
                       _loc='upper right',
                       y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7],
                       x_ticks=[1e0, 1e1, 1e2, 1e3, 1.6*1e3]):
    plt.rcParams['axes.linewidth'] = 0.5
    myfont = {'size': 36, 'family': 'Helvetica'}
    myfont_legend = {'size': 16, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    fg, (ax) = plt.subplots(1, 1, sharex=True, figsize=(12, 5))
    fg.subplots_adjust(hspace=0.05, right=0.7, top=0.95, bottom=0.15, left=0.1)

    markers = ['o']
    colors = ["#0072B2"]

    c = colors[0]
    ax.plot(iters, ratios, linewidth=2,
            alpha=0.8, color=c, linestyle='dashed', marker=markers[0],
            markerfacecolor=c, markersize=6, markevery=0.1)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    lines, labels = ax.get_legend_handles_labels()
    # fg.legend(lines, labels, loc='center left',
    #           bbox_to_anchor=(0.68, 0.7), borderaxespad=0.)

    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])
    # ax.set_yscale('log')
    # ax.set_xscale('linear')
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('Ratio $\\times 100 \%$', fontdict=myfont)
    plt.xlabel('Training Iteration $k$', fontdict=myfont)
    plt.xticks(x_ticks, fontsize=myfont['size'] // 3 * 2, rotation=45)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def plot_training_gain_multi_prob(iters: list, ratios: list, legends_1: list, name="compare_fixed",
                                  y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7],
                                  x_ticks=[1e0, 1e1, 1e2, 1e3, 1.6*1e3]):
    plt.rcParams['axes.linewidth'] = 0.5
    myfont = {'size': 36, 'family': 'Helvetica'}
    myfont_legend = {'size': 20, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"

    fg, (ax) = plt.subplots(1, 1, sharex=True, figsize=(12, 5))
    fg.subplots_adjust(hspace=0.05, right=0.7, top=0.95, bottom=0.15, left=0.1)

    # if not is_ood else 200
    #  if not is_ood else [1e2, 1e-2, 1e-4, 1e-6, 1e-7]

    # if not is_ood else [1e5, 1e4, 1e3, 1e2]
    # data_file = "losses-rand" if not is_ood else "losses-rand-OOD"
    for i in range(len(legends_1)):
        y = ratios[i]
        c = colors[i]
        ax.plot(iters, y, label=legends_1[i], linewidth=2,
                alpha=0.8, color=c, linestyle='dashed', marker=markers[i],
                markerfacecolor=c, markersize=6, markevery=0.05)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    lines, labels = ax.get_legend_handles_labels()
    fg.legend(lines, labels, loc='upper right',
              prop=myfont_legend,
              bbox_to_anchor=(0.95, 0.77), borderaxespad=0.)

    ax.set_ylim(0.94*y_ticks[-1], 1.1*y_ticks[0])

    # ax.set_yscale('log')
    # ax.set_xscale('log')

    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('Ratio ($\\times 100 \%$)', fontdict=myfont)
    plt.xlabel('Training Iteration $k$', fontdict=myfont)
    plt.xticks(x_ticks, fontsize=myfont['size'] // 3 * 2, rotation=45)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")
