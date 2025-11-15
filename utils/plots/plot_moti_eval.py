# This script is used to plot evaluation of the violation of LISTA on inference.
# Baselines: LISTA, LISTA-CPSS

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_nmse_processes(files: list, legends: list, name="compare_fixed",
                        data_file="lasso-rand-nmse",
                        y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7], plot_len=1000):

    plt.rcParams['axes.linewidth'] = 1.0
    myfont = {'size': 26, 'family': 'Helvetica'}
    myfont_legend = {'size': 20, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"
    # palette = sns.color_palette()
    # print(palette)
    fg = plt.figure(figsize=(20, 5))
    ax = plt.gca()

    markers1 = ['*', '.', '1', '2']
    # markers2 = ['1', '2']
    # 3:
    colors = ['b', 'tab:orange', 'g', 'y', 'm', 'tab:brown',
              'tab:pink', 'c', 'r', 'tab:olive', 'C0']

    # color_map = ['b', 'tab:orange', 'g', 'r', 'm', 'tab:brown', 'gold', 'forestgreen',
    #              'tab:pink', 'c', 'y', 'tab:olive', 'C0', 'slategray', 'indigo', 'crimson', 'fuchsia']
    # colors = iter(color_map)

    # if not is_ood else 200
    #  if not is_ood else [1e2, 1e-2, 1e-4, 1e-6, 1e-7]
    # x_ticks = [1e0, 1e1, 1e2, 1e3]
    x_ticks = list(range(plot_len))
    # if not is_ood else [1e5, 1e4, 1e3, 1e2]
    # data_file = "losses-rand" if not is_ood else "losses-rand-OOD"
    for i in range(len(legends)):
        y = np.loadtxt("results/" + files[i] + "/" + data_file)
        y = y[:plot_len]
        # sns.lineplot(data=y, palette=palette[i])
        c = colors[i]
        #  + '\t' + '{:.2e}'.format(mean_loss)
        plt.plot(y, label=legends[i], linewidth=2.0,
                 alpha=0.8, color=c, linestyle='dashed', marker=markers1[i],
                 markerfacecolor=c, markersize=6, markevery=0.1)

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    plt.legend(loc='upper right',
               #    bbox_to_anchor=(0, 1.05),
               prop=myfont_legend,
               #    ncols=4,
               frameon=1, framealpha=0.5)
    ax.set_ylim(0.9*y_ticks[-1], 1.1*y_ticks[0])
    # if not is_ood:
    #     ax.set_ylim(10**(-8), 10**1)
    # else:
    #     ax.set_ylim(10**(-8), 10**4)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('Violation Percentage', fontdict=myfont)
    plt.xlabel('Iteration $k$', fontdict=myfont)
    plt.xticks(x_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 3 * 2)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


def plot_violation_processes(files: list, legends: list, name="compare_fixed",
                             data_file=["signXk_XStar"],
                             y_ticks=[1e1, 1e-2, 1e-4, 1e-6, 1e-7], plot_len=1000):

    plt.rcParams['axes.linewidth'] = 1.0
    myfont = {'size': 30, 'family': 'Helvetica'}
    myfont_legend = {'size': 20, 'family': 'Helvetica'}
    plt.rcParams["axes.edgecolor"] = "black"
    # palette = sns.color_palette()
    # print(palette)
    fg = plt.figure(figsize=(10, 4))
    ax = plt.gca()

    markers1 = ['*', '.', '1', '2']
    # markers2 = ['1', '2']
    # 3:
    colors = ['b', 'tab:orange', 'g', 'y', 'm', 'tab:brown',
              'tab:pink', 'c', 'r', 'tab:olive', 'C0']

    # color_map = ['b', 'tab:orange', 'g', 'r', 'm', 'tab:brown', 'gold', 'forestgreen',
    #              'tab:pink', 'c', 'y', 'tab:olive', 'C0', 'slategray', 'indigo', 'crimson', 'fuchsia']
    # colors = iter(color_map)

    # if not is_ood else 200
    #  if not is_ood else [1e2, 1e-2, 1e-4, 1e-6, 1e-7]
    # x_ticks = [1e0, 1e1, 1e2, 1e3]
    x_ticks = list(range(plot_len))
    # if not is_ood else [1e5, 1e4, 1e3, 1e2]
    # data_file = "losses-rand" if not is_ood else "losses-rand-OOD"
    i = 0
    for j in range(len(files)):
        for s in range(len(data_file[j])):
            y = np.loadtxt("results/" + files[j] + "/" + data_file[j][s])
            y = y[:plot_len]
            # sns.lineplot(data=y, palette=palette[i])
            c = colors[i]
            #  + '\t' + '{:.2e}'.format(mean_loss)
            plt.plot(y, label=legends[i], linewidth=4.0,
                    alpha=0.8, color=c, linestyle='dashed', marker=markers1[i],
                    markerfacecolor=c, markersize=20, markevery=0.1)
            i += 1

    sns.set_style("whitegrid", {"axes.edgecolor": "black"})

    plt.legend(loc='upper right',
                  bbox_to_anchor=(1, 0.9),
               prop=myfont_legend,
               #    ncols=4,
               frameon=1, framealpha=0.5)
    ax.set_ylim(0.9*y_ticks[-1], 1.1*y_ticks[0])
    # if not is_ood:
    #     ax.set_ylim(10**(-8), 10**1)
    # else:
    #     ax.set_ylim(10**(-8), 10**4)
    # ax.set_yscale('log')
    # ax.set_xscale('log')
    plt.grid(which='major', axis='both', linestyle='dashed')
    plt.ylabel('Violation Ratio %', fontdict=myfont)
    plt.xlabel('Optimization Step $T$', fontdict=myfont)
    plt.xticks(x_ticks, fontsize=myfont['size'] // 4 * 3)
    plt.yticks(y_ticks, fontsize=myfont['size'] // 4 * 3)
    plt.tight_layout()
    plt.savefig("plots/" + name + ".pdf")


# # NOTE 0: Figure 0, NMSE
# files_1 = ['training/LASSO-LISTA', 'training/LASSO-LISTACPSS',
#            'training/LASSO-LISTA-WnotShared', 'training/LASSO-LISTACPSS-WnotShared']

# legends_1 = ['LASSO-LISTA', 'LASSO-LISTACPSS',
#              'LASSO-LISTA-WnotShared', 'LASSO-LISTACPSS-WnotShared']

# name = "training/figure0_ind_lasso_nmse"

# plot_nmse_processes(files_1, legends_1, name,
#                     y_ticks=[10, -20, -40, -60, -80, -100, -120], plot_len=16)


# # NOTE 1: Figure 1, LASSO Violation
# files_1 = 'training/LASSO-LISTA'

# legends_1 = ['$sign(x^k) \\neq sign(x^*)$', '$y \\neq Ax^*$', '$y \\neq Ax^k$']

# name = "training/figure1_ind_lasso_violation_lista"

# data_file = ["signXk_XStar", 'bXk', 'bXStar']
# plot_violation_processes(files_1, legends_1, name,
#                          data_file, y_ticks=[100, 80, 60, 40, 20, 0], plot_len=16)


# # NOTE 2: Figure 2, LASSO-CPSS Violation
# files_1 = 'training/LASSO-LISTACPSS'


# legends_1 = ['$sign(x^k) \\neq sign(x^*)$', '$y \\neq Ax^*$', '$y \\neq Ax^k$']

# name = "training/figure2_ind_lasso_violation_lista_cpss"

# data_file = ["signXk_XStar", 'bXk', 'bXStar']
# plot_violation_processes(files_1, legends_1, name,
#                          data_file, y_ticks=[100, 80, 60, 40, 20, 0], plot_len=16)


# # # NOTE 3: Figure 3, LASSO-CPSS Violation
# files_1 = 'training/LASSO-LISTACPSS'

# legends_1 = ['$W_i^{\\top}A_i \\neq 1$', '$W_i^{\\top}A_j > 1, j\\neq i$']

# name = "training/figure3_ind_lasso_Wviolation_lista_cpss"

# data_file = ["WiAi", 'WiAj']
# plot_violation_processes(files_1, legends_1, name,
#                          data_file, y_ticks=[100, 80, 60, 40, 20, 0], plot_len=16)


# # NOTE 4: Figure 4, LASSO Violation W not shared
# files_1 = 'training/LASSO-LISTA-WnotShared'

# legends_1 = ['$sign(x^k) \\neq sign(x^*)$', '$y \\neq Ax^*$', '$y \\neq Ax^k$']

# name = "training/figure4_ind_lasso_violation_lista_WnotShared"

# data_file = ["signXk_XStar", 'bXk', 'bXStar']
# plot_violation_processes(files_1, legends_1, name,
#                          data_file, y_ticks=[100, 80, 60, 40, 20, 0], plot_len=16)


# # NOTE 5: Figure 5, LASSO-CPSS Violation W not shared
# files_1 = 'training/LASSO-LISTACPSS-WnotShared'


# legends_1 = ['$sign(x^k) \\neq sign(x^*)$', '$y \\neq Ax^*$', '$y \\neq Ax^k$']

# name = "training/figure5_ind_lasso_violation_lista_cpss_WnotShared"

# data_file = ["signXk_XStar", 'bXk', 'bXStar']
# plot_violation_processes(files_1, legends_1, name,
#                          data_file, y_ticks=[100, 80, 60, 40, 20, 0], plot_len=16)


# # NOTE 6: Figure 6, LASSO-CPSS Violation W not shared
# files_1 = 'training/LASSO-LISTACPSS-WnotShared'

# legends_1 = ['$W_i^{\\top}A_i \\neq 1$', '$W_i^{\\top}A_j > 1, j\\neq i$']

# name = "training/figure6_ind_lasso_Wviolation_lista_cpss_WnotShared"

# data_file = ["WiAi", 'WiAj']
# plot_violation_processes(files_1, legends_1, name,
#                          data_file, y_ticks=[100, 80, 60, 40, 20, 0], plot_len=16)


# Paper
    
'''
NOTE: Paper figure 3a
'''
# Figure 1 Violation of Condition of Theorem by Training
files_1 = ['training/LASSO-LISTA', 'training/LASSO-LISTACPSS']

legends_1 = ['LISTA $sign(x^t) \\neq sign(x^*)$',
             'LISTA-CPSS $sign(x^t) \\neq sign(x^*)$',
             'LISTA-CPSS $W_i^{\\top}M_i \\neq 1$', 'LISTA-CPSS $W_i^{\\top}M_j > 1, j\\neq i$']

name_1 = "/figure1_ind_lasso_WShared"

data_file = [["signXk_XStar"], ["signXk_XStar", "WiAi", 'WiAj']]

plot_violation_processes(files_1, legends_1, name_1,
                         data_file, y_ticks=[100, 80, 60, 40, 20, 0], plot_len=16)


'''
NOTE: Paper figure 3b
'''
files_2 = ['training/LASSO-LISTA-WnotShared',
           'training/LASSO-LISTACPSS-WnotShared']

legends_2 = ['LISTA $sign(x^t) \\neq sign(x^*)$',
             'LISTA-CPSS $sign(x^t) \\neq sign(x^*)$',
             'LISTA-CPSS $W_i^{\\top}M_i \\neq 1$', 'LISTA-CPSS $W_i^{\\top}M_j > 1, j\\neq i$']

name_2 = "figure2_ind_lasso_WnotShared"

data_file = [["signXk_XStar"], ["signXk_XStar", "WiAi", 'WiAj']]

plot_violation_processes(files_2, legends_2, name_2,
                         data_file, y_ticks=[100, 80, 60, 40, 20, 0], plot_len=16)