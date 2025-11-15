# This script is used to plot training performance of our proposed model.
# None.


from plot_train_utils import *


# 0117 2025, QP with deterministic initialization

FREQUENCY = 100

PLOT_FILES = 50  # TODO change this Jan 25 2025


def plot_last_mean(PROBX, PROBY, y_ticks_last, y_ticks_traj, t='100', e='100', lr='0.0000001'):
    # TODO: Select final running with best LR.

    model_dir = 'ablation_fullsparse_sumloss_T' + t + '/SGDlr' + lr + '-T' + t + '-rand' + e + 'orth-'+PROBX + PROBY + \
        '-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0/obj_train_iter_'

    iters = [i*FREQUENCY for i in range(PLOT_FILES)]
    x_ticks = [i*500 for i in range(0, 10)]
    # x_ticks = [format(i*FREQUENCY, "e") for i in range(PLOT_FILES)]

    name_1 = "results_figure1_"+PROBX + PROBY+"_last_lr" + lr + "e" + e
    files_1 = [model_dir + str(i*FREQUENCY) for i in range(PLOT_FILES)]
    legends_1 = ['Gradient Descent', 'Math-L2O']
    plot_training_last_obj(iters, files_1, legends_1, name_1,
                           y_ticks=y_ticks_last,
                           x_ticks=x_ticks, t=t)

    # files_2 = [model_dir + str(i*5000) for i in range(10)]
    # legends_2 = ['$x:$ '+PROBX + ' $y$: '+PROBY+' Training Step-' +
    #              str(i*5000) for i in range(10)]
    # name_2 = "results_figure1_"+PROBX + PROBY+"_traj_lr" + lr +  "e" + e
    # plot_training_GD_process(files_2, legends_2, name_2,
    #                          y_ticks=y_ticks_traj,
    #                          x_ticks=[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])


'''
NOTE: Paper figure 4a
'''
# For x 512 y 400 problem
PROBX = '512'
PROBY = '400'
t = '100'
e = '100'
lr = '0.0000001'
y_ticks_last = [1e0, 9*1e-1, 8*1e-1,
                7*1e-1, 6*1e-1, 5*1e-1, 4*1e-1]
y_ticks_traj = [5*1e2, 1*1e2, 1*1e1, 1*1e0, 1*1e-1, 1*1e-2]

plot_last_mean(PROBX, PROBY, y_ticks_last, y_ticks_traj, t, e, lr)

# # # For x 256 y 200 problem
# PROBX = '256'
# PROBY = '200'
# y_ticks_last = [5*1e-2, 4*1e-2, 3*1e-2, 2*1e-2]
# y_ticks_traj =[5*1e2, 1*1e2, 1*1e1, 1*1e0, 1*1e-1, 1*1e-2]
# plot_last_mean(PROBX, PROBY, y_ticks_last, y_ticks_traj)

# # # For x 128 y 110 problem
# PROBX = '128'
# PROBY = '100'
# y_ticks_last = [2*1e-2, 1*1e-2, 0.8*1e-2]
# y_ticks_traj =[2*1e2, 1*1e0, 1*1e-1, 1*1e-2, 0.8*1e-2]
# plot_last_mean(PROBX, PROBY, y_ticks_last, y_ticks_traj)

# # # For x 64 y 50 problem
# PROBX = '64'
# PROBY = '50'
# y_ticks_last = [1.5*1e-2, 0.8*1e-2, 0.6*1e-2]
# y_ticks_traj =[2*1e2, 1*1e0, 1*1e-1, 1*1e-2, 5*1e-3]
# plot_last_mean(PROBX, PROBY, y_ticks_last, y_ticks_traj)

# # For x 32 y 20 problem
# PROBX = '32'
# PROBY = '20'
# y_ticks_last = [0.8*1e-3, 0.7*1e-3, 0.6*1e-3, 0.5*1e-3, 0.4*1e-3, 0.3*1e-3]
# y_ticks_traj =[2*1e1, 1*1e0, 1*1e-1, 1*1e-3, 2*1e-4]
# plot_last_mean(PROBX, PROBY, y_ticks_last, y_ticks_traj)
