import numpy as np
from plot_train_utils import *

FREQUENCY = 100
NUM_FILES = 50  # TODO Change this Jan 25 2025
iters = [i*FREQUENCY for i in range(0, NUM_FILES)]

FREQUENCY_x_ticks = 500
x_ticks = [i*FREQUENCY_x_ticks for i in range(0, 10)]


def cal_ratio_last(file_GD_baseline, files_l2o):
    obj_gd = np.loadtxt("results/" + file_GD_baseline)[-1]
    ratios = []
    for f in files_l2o:
        obj_l2o = np.loadtxt("results/" + f)[-1]
        ratios.append((obj_gd - obj_l2o) / obj_gd)
    return ratios


def cal_ratio_mean(file_GD_baseline, files_l2o):
    obj_gd = np.loadtxt("results/" + file_GD_baseline)
    ratios = []
    for f in files_l2o:
        obj_l2o = np.loadtxt("results/" + f)
        ratios.append(np.mean((obj_gd - obj_l2o) / obj_gd))
    return ratios


def plot_gain_ratio_last(file_GD_baseline, files_l2o, name_1, x_ticks):
    ratios = cal_ratio_last(file_GD_baseline, files_l2o)

    plot_training_gain(iters, ratios, name_1, y_ticks=[
                       0.6, 0.4, 0.2, 0], x_ticks=x_ticks)


def plot_gain_ratio_mean(file_GD_baseline, files_l2o, name_1, x_ticks):
    ratios = cal_ratio_mean(file_GD_baseline, files_l2o)

    plot_training_gain(iters, ratios, name_1, y_ticks=[
                       0.6, 0.4, 0.2, 0], x_ticks=x_ticks)


# PROB = '512400'
# PROB = '256200'
# PROB = '128100'
# PROB = '6450'
PROB = '3225'

# Single problem, mean of 100 GD iteration ratio among training
model_dir = 'SGDlr0.0000001-T100-rand100orth-' + PROB + \
    '-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0/obj_train_iter_'
file_GD_baseline = model_dir + '0'
files_l2o = [model_dir + str(i*FREQUENCY) for i in range(0, NUM_FILES)]


name_1 = "results_figure2_"+PROB+"_training_gain_ratio_last"
# plot_gain_ratio_last(file_GD_baseline, files_l2o, name_1, x_ticks)

name_2 = "results_figure2_"+PROB+"_training_gain_ratio_mean"
# plot_gain_ratio_mean(file_GD_baseline, files_l2o, name_2, x_ticks)


# Ablation on 32 20


# NUM_FILES = 50

def plot_gain_ratio_last_multi_e(problems, legends, name, p_size='3220', lr='0.0000001', y_ticks=[0.6, 0.4, 0.2, 0]):
    FREQUENCY = 100
    NUM_FILES = 50  # TODO Change this Jan 25 2025
    iters = [i*FREQUENCY for i in range(0, NUM_FILES)]

    FREQUENCY_x_ticks = 500
    x_ticks = [i*FREQUENCY_x_ticks for i in range(0, 10)]

    all_ratios = []
    for p in problems:
        model_dir = 'ablation_fullsparse_sumloss_T' + t + '/SGDlr' + lr + '-T' + t + '-rand'+p+'orth-' + p_size + \
            '-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0/obj_train_iter_'
        file_GD_baseline = model_dir + '0'
        files_l2o = [model_dir + str(i*FREQUENCY) for i in range(0, NUM_FILES)]

        ratios_one_prob = cal_ratio_last(file_GD_baseline, files_l2o)
        all_ratios.append(ratios_one_prob)

    plot_training_gain_multi_prob(
        iters, all_ratios, legends, name, y_ticks=y_ticks, x_ticks=x_ticks)


def plot_gain_ratio_mean_multi_e(problems, legends, name, p_size='3220', lr='0.0000001', y_ticks=[0.6, 0.4, 0.2, 0]):
    FREQUENCY = 100
    NUM_FILES = 50  # TODO Change this Jan 25 2025
    iters = [i*FREQUENCY for i in range(0, NUM_FILES)]

    FREQUENCY_x_ticks = 500
    x_ticks = [i*FREQUENCY_x_ticks for i in range(0, 10)]

    all_ratios = []
    for p in problems:
        model_dir = 'ablation_fullsparse_sumloss_T' + t + '/SGDlr' + lr + '-T' + t + '-rand'+p+'orth-' + p_size + \
            '-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0/obj_train_iter_'
        file_GD_baseline = model_dir + '0'
        files_l2o = [model_dir + str(i*FREQUENCY) for i in range(0, NUM_FILES)]

        ratios_one_prob = cal_ratio_mean(file_GD_baseline, files_l2o)
        all_ratios.append(ratios_one_prob)

    plot_training_gain_multi_prob(
        iters, all_ratios, legends, name, y_ticks=y_ticks, x_ticks=x_ticks)


legends = ['$x: 32$, $y: 20$',
           '$x: 64$, $y: 50$',
           '$x: 128$, $y: 110$',
           '$x: 256$, $y: 200$',
           '$x: 512$, $y: 400$']

# problems = ['3220', '6450', '128100', '256200', '512400']


def plot_gain_ratio_last_multi_lr(lrs, legends, name, e='100', p='3220', t='10', y_ticks=[0.5, 0.3, 0.1, -0.1, -1.3]):

    all_ratios = []
    for l in lrs:
        model_dir = 'ablation_fullsparse_sumloss_T' + t + '/SGDlr' + l + '-T' + t + '-rand' + e + 'orth-' + p + \
            '-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0/obj_train_iter_'
        file_GD_baseline = model_dir + '0'
        files_l2o = [model_dir + str(i*FREQUENCY) for i in range(0, NUM_FILES)]

        ratios_one_prob = cal_ratio_last(file_GD_baseline, files_l2o)
        all_ratios.append(ratios_one_prob)

    plot_training_gain_multi_prob(
        iters, all_ratios, legends, name, y_ticks=y_ticks, x_ticks=x_ticks)


def plot_gain_ratio_mean_multi_lr(lrs, legends, name, e='100', p='3220', t='10', y_ticks=[0.5, 0.3, 0.1, -0.1, -0.8]):

    all_ratios = []
    for l in lrs:
        model_dir = 'ablation_fullsparse_sumloss_T' + t + '/SGDlr' + l + '-T' + t + '-rand' + e + 'orth-' + p + \
            '-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0/obj_train_iter_'
        file_GD_baseline = model_dir + '0'
        files_l2o = [model_dir + str(i*FREQUENCY) for i in range(0, NUM_FILES)]

        ratios_one_prob = cal_ratio_mean(file_GD_baseline, files_l2o)
        all_ratios.append(ratios_one_prob)

    plot_training_gain_multi_prob(
        iters, all_ratios, legends, name, y_ticks=y_ticks, x_ticks=x_ticks)


# ******************** Plot Alation of LR ***********************
legends = ['Learning Rate: 0.001',
           'Learning Rate: 0.0001',
           'Learning Rate: 0.00001',
           'Learning Rate: 0.000001',
           'Learning Rate: 0.0000001']

lrs = ['0.001', '0.0001', '0.00001', '0.000001', '0.0000001']

e = '100' # '1', '5', '25', '50', '100', '200', '300', '500', '5000'

t = '100' # '5', '20', '30', '50', '100'

# y_ticks_last = [0.1, -0.1, -1.0]
# y_ticks_mean = [0.1, -0.1, -1.0]

# y_ticks_last = [0.5, 0.1, -0.1, -1.0]
# y_ticks_mean = [0.5, 0.1, -0.1, -1.0]

# t = '10'
# y_ticks_last = [0.5, 0.3, 0.1, -0.1, -1.3]
# y_ticks_mean = [0.5, 0.3, 0.1, -0.1, -0.8]

y_ticks_last = [0.5, 0.3, 0.1, -0.1]
y_ticks_mean = [0.5, 0.3, 0.1, -0.1]

# y_ticks_last = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
# y_ticks_mean = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]

# t = '20'
# y_ticks_last = [0.5, 0.3, 0.1, -0.1]
# # y_ticks_mean = [0.1, -0.1, -0.8]
# y_ticks_mean = [0.5, 0.3, 0.1, -0.1]

# t = '30'
# y_ticks_last = [0.5, 0.3, 0.1, -0.1]
# y_ticks_mean = [0.1, -0.1, -0.8]

# t = '50'
# y_ticks_last = [0.5, 0.3, 0.1, -0.1]
# y_ticks_mean = [0.3, 0.1, -0.1]

# t = '100'
# y_ticks_last = [0.6, 0.4, 0.2, 0]
# y_ticks_mean = [0.6, 0.4, 0.2, 0]

# '3225', '6450', '128100', '256200', '512400'
problems = ['3225']

# for p in problems:
#     name_3 = "results_figure4_ablation_lr_multi_training_gain_ratio_last_fullsparse_sum_loss_T" + \
#         t + "_e" + e + "_" + p
#     name_4 = "mean_results/results_figure4_ablation_lr_multi_training_gain_ratio_mean_fullsparse_sum_loss_T" + \
#         t + "_e" + e + "_" + p
#     plot_gain_ratio_last_multi_lr(
#         lrs, legends, name_3, e=e, p=p, t=t, y_ticks=y_ticks_last)
#     plot_gain_ratio_mean_multi_lr(
#         lrs, legends, name_4, e=e, p=p, t=t, y_ticks=y_ticks_mean)


# 32 25 scale
# name_3 = "results_figure5_ablation_lr_multi_training_gain_ratio_last_fullsparse_3225"
# name_4 = "mean_results/results_figure5_ablation_lr_multi_training_gain_ratio_mean_fullsparse_3225"
# plot_gain_ratio_last_multi_lr(lrs, legends, name_3, p='3225')
# plot_gain_ratio_mean_multi_lr(lrs, legends, name_4, p='3225')


# ******************** Plot Alation of e ***********************
legends = [
    '$e: 1$',
    '$e: 5$',
    '$e: 25$',
    '$e: 50$',
    '$e: 100$',
    # '$e: 500$',
    # '$e: 1000$',
    # '$e: 2000$',
    # '$e: 5000$',
    # '$e: 10000$',
    # '$e: 20000$'
    ]
es = ['1', '5', '25', '50', '100']
# es = ['100', '500', '1000', '2000', '5000', '10000', '20000']

lr = '0.0000001'

t = '100'

# y_ticks_last = [0.5, 0.3, 0.1, -1.0]
# y_ticks_mean = [0.3, 0.1, -0.5]


# t = '5' # '10' '20' '30' '50' '100'
y_ticks_last = [0.5, 0.3, 0.1, -0.1]
y_ticks_mean = [0.5, 0.3, 0.1, -0.1]

# name_2 = "results_figure3_ablation_multi_training_gain_ratio"

# '3225', '6450', '128100', '256200', '512400'
problems = ['3225', ]

# for p in problems:
#     name_3 = "results_figure3_ablation_e_multi_training_gain_ratio_last_fullsparse_sum_loss_T" + \
#         t + "_lr" + lr + "_" + p
#     name_4 = "mean_results/results_figure3_ablation_e_multi_training_gain_ratio_mean_fullsparse_sum_loss_T" + \
#         t + "_lr" + lr + "_" + p
#     plot_gain_ratio_last_multi_e(
#         es, legends, name_3, p_size=p, lr=lr, y_ticks=y_ticks_last)
#     plot_gain_ratio_mean_multi_e(
#         es, legends, name_4, p_size=p, lr=lr, y_ticks=y_ticks_mean)

