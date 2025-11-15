# This script is used to plot the training performance comparison with LISTA.
# Baselines: GD, Adam, LISTA

from plot_train_utils import *
import math

FREQUENCY = 100

PLOT_FILES = 50  # TODO change this Jan 25 2025


def sci_to_latex(x: float, precision: int = 1) -> str:
    """
    Convert a float to LaTeX scientific-notation string, e.g. 2e-6 -> '2*10^{-6}'.

    Parameters
    ----------
    x : float
        The number to convert.
    precision : int, optional
        Significant-figure precision to keep in the mantissa (default 1).

    Returns
    -------
    str
        LaTeX-ready string.
    """
    if x == 0:
        return "0"

    # Format in scientific notation, e.g. '2.0e-06'
    sci = f"{x:.{precision}e}"
    mantissa, exp = sci.split("e")

    # Clean up mantissa like '2.0' -> '2'
    mantissa = mantissa.rstrip("0").rstrip(".")
    exp = int(exp)          # remove leading zeros and '+'

    # Omit the coefficient if it is exactly 1
    if mantissa == "1":
        return fr"10^{{{exp}}}"
    else:
        return fr"{mantissa}*10^{{{exp}}}"


def plot_last_mean_lista(y_ticks_last, lr='0.001'):

    iters = [i*FREQUENCY for i in range(PLOT_FILES)]
    x_ticks = [i*500 for i in range(0, 10)]
    # x_ticks = [format(i*FREQUENCY, "e") for i in range(PLOT_FILES)]

    gd_file = f'results/training/FixedM/SGDlr{lr}-T100-rand5000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0-FixedM/obj_train_iter_0'

    # model = 'LISTA-CPSS-Shared'
    lista_model_sharedW_randW = 'LISTA-CPSS-Shared-FixM-RandInitW'
    lista_model_sharedW = 'LISTA-CPSS-Shared-FixM'
    # model = 'LISTA-CPSS-NotShared'
    lista_model_nosharedW_randW = 'LISTA-CPSS-NotShared-FixM-RandInitW'
    lista_model_nosharedW = 'LISTA-CPSS-NotShared-FixM'

    our_model = f'SGDlr{lr}-T100-rand5000orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0-FixedM'

    files_lista_model_sharedW_randW = ['results/training/FixedM/' + lista_model_sharedW_randW + '-lr' + lr +
                                       '/obj_train_iter_' + str(i*FREQUENCY) for i in range(PLOT_FILES)]

    files_lista_model_sharedW = ['results/training/FixedM/' + lista_model_sharedW + '-lr' + lr +
                                 '/obj_train_iter_' + str(i*FREQUENCY) for i in range(PLOT_FILES)]

    files_lista_model_nosharedW_randW = ['results/training/FixedM/' + lista_model_nosharedW_randW + '-lr' + lr +
                                         '/obj_train_iter_' + str(i*FREQUENCY) for i in range(PLOT_FILES)]

    files_lista_model_nosharedW = ['results/training/FixedM/' + lista_model_nosharedW + '-lr' + lr +
                                   '/obj_train_iter_' + str(i*FREQUENCY) for i in range(PLOT_FILES)]

    files_our_model = ['results/training/FixedM/' + our_model + '/obj_train_iter_' +
                       str(i*FREQUENCY) for i in range(PLOT_FILES)]

    files_list = [files_our_model, files_lista_model_sharedW_randW, files_lista_model_sharedW,
                  files_lista_model_nosharedW_randW, files_lista_model_nosharedW]

    legends_1 = ['Gradient Descent', "Our L2O", lista_model_sharedW_randW,
                 lista_model_sharedW, lista_model_nosharedW_randW, lista_model_nosharedW]

    name_1 = f"results6_train_T100_512400_lr" + lr

    plot_training_last_obj_gd_vs_others(
        iters, gd_file, files_list, legends_1, name_1, y_ticks=y_ticks_last, x_ticks=x_ticks)


def plot_last_mean_baseline(y_ticks_last, step='400', lrs=[0.001, 0.0001, 1e-05, 1e-06, 1e-07], base_path='Math-L2O-PA', base_model_name='QP-Math-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0', name_1="figure3"):

    # x_ticks = [format(i*FREQUENCY, "e") for i in range(PLOT_FILES)]

    gd_file = f'training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr0.001--optimizer-training-steps100--unroll-length100/obj_train_iter_0'

    unroll_len = list(range(5, 105, 5))
    files_list = [
        [f'results/training/{base_path}/{base_model_name}-lr{lr}--optimizer-training-steps{ul}--unroll-length{ul}/obj_train_iter_{step}' for ul in unroll_len] for lr in lrs]

    legends_1 = ['Gradient Descent'] + ['LR:$' + str(lr) + '$' for lr in lrs]

    name_1 = name_1 + f"_train_512400_{base_path}_k{step}"

    plot_training_last_obj_baseline_vs_ours(step,
                                            unroll_len, gd_file, files_list, legends_1, name_1, y_ticks=y_ticks_last, x_ticks=unroll_len)


def plot_last_mean_lista_fixedM(y_ticks_last, gd_path="results/training/FixedM/SGDlr0.001-T100-rand100orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0-UP", lista_path='results/training/FixedM/LISTA-CPSS-WOnly-NotShared-lr0.0005-MInitW-UnrollTrain', myfontsize_legend=30):
    '''
    Plot training performance of orignial but unsupervised version of LISTA-CPSS.
    '''
    PLOT_FILES = 50
    FREQUENCY = 400
    iters = [i*FREQUENCY for i in range(0, PLOT_FILES)]
    x_ticks = [i*FREQUENCY*5 for i in range(0, PLOT_FILES//5)]
    # x_ticks = [format(i*FREQUENCY, "e") for i in range(PLOT_FILES)]

    name_1 = "results6_train_listacpss_fixedM"
    gd_file = f"{gd_path}/obj_train_iter_0"
    files_list = [
        [lista_path + f"/obj_train_iter_{i}" for i in range(PLOT_FILES)]]
    legends_1 = ['Gradient Descent', 'LISTA-CPSS']
    plot_training_last_obj_gd_vs_others(
        iters, gd_file, files_list, legends_1, name_1, y_ticks=y_ticks_last, x_ticks=x_ticks, myfontsize_legend=myfontsize_legend)


def plot_last_mean_lista_original(y_ticks_last, step='400', lista_name='LISTA-CPSS', lista_path='results/training/MultipleM/LISTA-CPSS-WOnly-NotShared-lr0.001-ZeroW-UnrollTrain', our_name='Our', our_path='results/training/MultipleM/SGDlr0.001-T100-rand100orth-512400-QP-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0-UP', name_1="figure3"):
    '''
    Plot training performance of orignial training scheme, but unsupervised version of LISTA-CPSS.
    '''
    PLOT_FILES = 13
    FREQUENCY = 400

    gd_path = f'{our_path}/obj_train_iter_0'

    iters = [i*FREQUENCY for i in range(PLOT_FILES)]
    x_ticks = [i*FREQUENCY for i in range(0, PLOT_FILES)]
    # x_ticks = [format(i*FREQUENCY, "e") for i in range(PLOT_FILES)]

    name_1 = name_1 + f"_train_512400_{lista_name}_vs_{our_name}_k{step}"
    lista_files = [f'{lista_path}/obj_train_iter_' +
                   str(i) for i in range(PLOT_FILES)]
    our_files = [f'{our_path}/obj_train_iter_' +
                 str(i*FREQUENCY) for i in range(PLOT_FILES)]

    files_list = [lista_files, our_files]

    legends_1 = ['Gradient Descent', "LISTA-CPSS", "Our L2O"]
    plot_training_last_obj_gd_vs_others(
        iters, gd_path, files_list, legends_1, name_1, y_ticks=y_ticks_last, x_ticks=x_ticks)


def plot_last_mean_cnn_mnist_GD_vs_Ours(y_ticks, gd_path="inference/cnn/SGDlr0.000001-T10000-rand100orth-MINST200AdamStepS01-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0", our_path='inference/cnn/SGDlr0.000001-T100-rand100orth-MINST200SGDStepS01-L2O-PShallow-SgleLoss-DNN-DetachState-sigmoid-ZeroX0', myfontsize_legend=30):
    '''
    Plot training performance of orignial but unsupervised version of LISTA-CPSS.
    '''
    
    PLOT_FILES = 3
    FREQUENCY = 10
    iters = [100 for i in range(0, PLOT_FILES)]
    x_ticks = [i*FREQUENCY for i in range(0, 100//FREQUENCY)]
    EPOCHS = 100
    # x_ticks = [format(i*FREQUENCY, "e") for i in range(PLOT_FILES)]

    name_1 = "results8_train_cnn_mnist_GD_vs_Ours"
    gd_file = f"{gd_path}/obj_train_iter_0"
    files_list = [gd_file] + [our_path + f"/obj_train_iter_{i*EPOCHS}" for i in range(1, PLOT_FILES)]

    legends_1 = ['Gradient Descent', 'Our L2O, Epochs: 100', 'Our L2O, Epochs: 200']
    plot_opt_process(files_list, legends_1, iters, name_1, y_ticks=y_ticks, x_ticks=x_ticks)
        

if __name__ == "__main__":
    '''
    Case 0: Motivation from Math-L2O
    '''
    '''
    NOTE: Paper figure 2a 
    '''
    # **Figure: Math-L2O Grid Search Learning Rate
    # y_ticks_last = [2e2, 1e1, 1e0, 5e-1]
    # plot_last_mean_baseline(y_ticks_last,
    #                         lrs=[1e-05, 2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 7e-05, 8e-05, 9e-05, 1e-04],
    #                         base_path='Math-L2O-P',
    #                         base_model_name='QP-Math-L2O-P-SgleLoss-DetachState-Sigmoid-ZeroX0',
    #                         name_1="figure3_moti")

    '''
    NOTE: Paper figure 2b 
    '''
    # **Figure: LISTACPSS Grid Searching Learning Rate
    # y_ticks_last = [2e2, 1e1, 1e0, 5e-1]
    # plot_last_mean_baseline(y_ticks_last,
    #                         lrs=[1e-05, 2e-05, 3e-05, 4e-05, 5e-05, 6e-05, 7e-05, 8e-05, 9e-05, 1e-04],
    #                         base_path='LISTACPSS',
    #                         base_model_name='LISTACPSSWOnly-ZeroX0',
    #                         name_1="figure4_moti")

    '''
    Case 1: Comparison with LISTA
    '''
    # # For x 512 y 400 problem
    # t = '100'
    # # y_ticks_last = [3e3, 1e3, 1e1, 1e0]
    # # y_ticks_traj = [5*1e2, 1*1e2, 1*1e1, 1*1e0, 1*1e-1, 1*1e-2]

    # lr = '0.001'
    # y_ticks_last = [1e4, 1e-1, 1e-2, 0.5e-2]

    # # lr = '0.0000001'
    # # y_ticks_last = [1e4, 1e0, 1e-1]

    # plot_last_mean_lista(y_ticks_last, lr)

    # **Figure: LISTACPSS General Comparison
    # y_ticks_last = [1e4, 1e3, 1e2, 1e1, 1e0, 1e-1]
    # plot_last_mean_baseline(y_ticks_last, base_path='LISTACPSS', base_model_name='LISTACPSSWOnly-ZeroX0', name_1="figure4")

    '''
    NOTE: Paper figure 7
    '''
    # **Figure: LISTACPSS Comparison with exp aligned with LISTA-CPSS paper settings
    # y_ticks_last = [1e6, 1e5, 1e4, 1e3, 1e2, 1e1]
    # plot_last_mean_lista_original(y_ticks_last, name_1="results6")

    # **Figure: LISTACPSS Fixed M
    # y_ticks_last = [1e4, 1e2, 1e0, 1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    # plot_last_mean_lista_fixedM(y_ticks_last)
    '''
    Case 2: Comparison with Math-L2O
    '''
    # **Figure: Math-L2O-P General Comparison
    # y_ticks_last = [2e2, 1e1, 1e0, 5e-1]
    # plot_last_mean_baseline(y_ticks_last, base_path='Math-L2O-P', base_model_name='QP-Math-L2O-P-SgleLoss-DetachState-Sigmoid-ZeroX0')

    # Figure: Math-L2O-PA Gneral Comparison
    # y_ticks_last = [2e2, 1e1, 1e0, 3e-1]
    # plot_last_mean_baseline(y_ticks_last)

    '''
    NOTE: Paper figure 4b
    '''
    # **Figure: Our Results, robust to learning rate.
    # y_ticks_last = [2e2, 1e1, 1e0, 3e-1]
    # plot_last_mean_baseline(y_ticks_last, base_path='Our', base_model_name='QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0')

    '''
    Case 3: Plot Math-L2O, LISTA, Ours in one figure
    '''
    y_ticks=[1e1, 1e-1, 1e-2, 1e-3, 1e-4]
    plot_last_mean_cnn_mnist_GD_vs_Ours(y_ticks)
