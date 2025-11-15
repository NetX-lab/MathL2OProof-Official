"""
File:    utils.py
Created: December 2, 2019
Revised: December 2, 2019
Authors: Howard Heaton, Xiaohan Chen
Purpose: Definition of different utility functions used in other files, e.g.,
         logging handler.
"""

import logging
import torch
import torch.nn.functional as F
import os
import numpy as np
import numpy as np



class TrainingDebugger:

    def __init__(self, opts) -> None:
        if opts.optimizer == 'GOMathL2O':
            self.R, self.Q, self.H, self.A, self.B, self.C, = [], [], [], [], [], []
            self.b1, self.b2, self.b3 = [], [], []
        elif opts.optimizer == 'CoordMathLSTM':
            self.P, self.A2 = [], []

    def add(self, opts, optimizer):
        if opts.optimizer == 'GOMathL2O':
            if opts.r_use:
                self.R.append(
                    optimizer.R.mean().detach().cpu().item())
            if opts.q_use:
                self.Q.append(
                    optimizer.Q.mean().detach().cpu().item())
            if opts.h_use:
                self.H.append(
                    optimizer.H.mean().detach().cpu().item())
            if opts.a_use:
                self.A.append(
                    optimizer.A.mean().detach().cpu().item())
            if opts.b_use:
                self.B.append(
                    optimizer.B.mean().detach().cpu().item())
            if opts.c_use:
                self.C.append(
                    optimizer.C.mean().detach().cpu().item())
            if opts.b3_use:
                self.b1.append(
                    torch.linalg.norm(
                        optimizer.b1, dim=(-2, -1), ord='fro', keepdim=False
                    ).mean().detach().cpu().item())
            if opts.b4_use:
                self.b2.append(
                    torch.linalg.norm(
                        optimizer.b2, dim=(-2, -1), ord='fro', keepdim=False
                    ).mean().detach().cpu().item())
            if opts.b5_use:
                self.b3.append(
                    torch.linalg.norm(
                        optimizer.b3, dim=(-2, -1), ord='fro', keepdim=False
                    ).mean().detach().cpu().item())
        elif opts.optimizer == 'CoordMathLSTM':
            self.P.append(optimizer.P.mean().detach().cpu().item())
            self.A2.append(optimizer.A.mean().detach().cpu().item())

    def log(self, opts, writer, validation_losses, validation_grad, validation_grad_dirc, eval_step):
        for e in range(0, opts.val_length, 10):
            writer.add_scalar(
                "eval/Obj-Step"+str(e+1), validation_losses[e], eval_step)
            writer.add_scalar(
                "eval/Grad-Step"+str(e+1), validation_grad[e], eval_step)
            writer.add_scalar(
                "eval/Grad-Target-Step"+str(e+1), validation_grad_dirc[e], eval_step)
            if opts.optimizer == 'GOMathL2O':
                if opts.r_use:
                    writer.add_scalar(
                        "R/Step"+str(e+1), self.R[e], eval_step)
                if opts.q_use:
                    writer.add_scalar(
                        "Q/Step"+str(e+1), self.Q[e], eval_step)
                if opts.h_use:
                    writer.add_scalar(
                        "H/Step"+str(e+1), self.H[e], eval_step)
                if opts.a_use:
                    writer.add_scalar(
                        "A/Step"+str(e+1), self.A[e], eval_step)
                if opts.b_use:
                    writer.add_scalar(
                        "B/Step"+str(e+1), self.B[e], eval_step)
                if opts.c_use:
                    writer.add_scalar(
                        "C/Step"+str(e+1), self.C[e], eval_step)
                if opts.b3_use:
                    writer.add_scalar(
                        "b1/Step"+str(e+1), self.b1[e], eval_step)
                if opts.b4_use:
                    writer.add_scalar(
                        "b2/Step"+str(e+1), self.b2[e], eval_step)
                if opts.b5_use:
                    writer.add_scalar(
                        "b3/Step"+str(e+1), self.b3[e], eval_step)
            elif opts.optimizer == 'CoordMathLSTM':
                writer.add_scalar(
                    "P/Step"+str(e+1), self.P[e], eval_step)
                writer.add_scalar(
                    "A2/Step"+str(e+1), self.A2[e], eval_step)


class TestingDebugger:
    def __init__(self, opts) -> None:
        if opts.optimizer == 'GOMathL2O':
            self.R, self.Q, self.H, self.A, self.B, self.C, = [0.0] * (opts.test_length), [0.0] * (opts.test_length), [0.0] * (
                opts.test_length), [0.0] * (opts.test_length), [0.0] * (opts.test_length), [0.0] * (opts.test_length)
            # Input of LSTM
            self.lstm_input, self.lstm_input_c, self.lstm_input_d = [
                0.0] * (opts.test_length), [0.0] * (opts.test_length), [0.0] * (opts.test_length)
        elif opts.optimizer == 'CoordMathLSTM':
            self.P, self.A2 = [0.0] * \
                (opts.test_length), [0.0] * (opts.test_length)
        elif opts.optimizer == 'CoordMathDNN':
            self.P = [0.0] * (opts.test_length)

    def add(self, opts, optimizer, j):
        if opts.optimizer == 'GOMathL2O':
            if opts.r_use:
                self.R[j] += optimizer.R.mean().detach().cpu().item()
            if opts.q_use:
                self.Q[j] += optimizer.Q.mean().detach().cpu().item()
            if opts.h_use:
                self.H[j] += optimizer.H.mean().detach().cpu().item()
            if opts.a_use:
                self.A[j] += optimizer.A.mean().detach().cpu().item()
            if opts.b_use:
                self.B[j] += optimizer.B.mean().detach().cpu().item()
            if opts.c_use:
                self.C[j] += optimizer.C.mean().detach().cpu().item()
            self.lstm_input[j] += torch.linalg.norm(
                optimizer.lstm_input, dim=(-2, -1), ord='fro', keepdim=False).mean().detach().cpu().item()
            self.lstm_input_c[j] += torch.linalg.norm(
                optimizer.lstm_input_c, dim=(-2, -1), ord='fro', keepdim=False).mean().detach().cpu().item()
            self.lstm_input_d[j] += torch.linalg.norm(
                optimizer.lstm_input_d, dim=(-2, -1), ord='fro', keepdim=False).mean().detach().cpu().item()
        elif opts.optimizer == 'CoordMathLSTM':
            self.P[j] += optimizer.P.mean().detach().cpu().item()
            self.A2[j] += optimizer.A.mean().detach().cpu().item()
        elif opts.optimizer == 'CoordMathDNN':
            self.P[j] += optimizer.P.mean().detach().cpu().item()

    def log(self, opts, num_test_batches):
        if opts.optimizer == 'GOMathL2O':
            if opts.r_use:
                self.R = [r / num_test_batches for r in self.R]
                np.savetxt(os.path.join(opts.save_dir, 'R_S' +
                                        str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.R))
            if opts.q_use:
                self.Q = [q / num_test_batches for q in self.Q]
                np.savetxt(os.path.join(opts.save_dir, 'Q_S' +
                                        str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.Q))
            if opts.h_use:
                self.H = [h / num_test_batches for h in self.H]
                np.savetxt(os.path.join(opts.save_dir, 'H_S' +
                                        str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.H))
            if opts.a_use:
                self.A = [a / num_test_batches for a in self.A]
                np.savetxt(os.path.join(opts.save_dir, 'A_S' +
                                        str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.A))
            if opts.b_use:
                self.B = [b / num_test_batches for b in self.B]
                np.savetxt(os.path.join(opts.save_dir, 'B_S' +
                                        str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.B))
            if opts.c_use:
                self.C = [c / num_test_batches for c in self.C]
                np.savetxt(os.path.join(opts.save_dir, 'C_S' +
                                        str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.C))
            self.lstm_input = [l1 / num_test_batches for l1 in self.lstm_input]
            np.savetxt(os.path.join(opts.save_dir, 'lstm_input_S' +
                                    str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.lstm_input))
            self.lstm_input_c = [
                l2 / num_test_batches for l2 in self.lstm_input_c]
            np.savetxt(os.path.join(opts.save_dir, 'lstm_input_c_S' +
                                    str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.lstm_input_c))
            self.lstm_input_d = [
                l3 / num_test_batches for l3 in self.lstm_input_d]
            np.savetxt(os.path.join(opts.save_dir, 'lstm_input_d_S' +
                                    str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.lstm_input_d))
        elif opts.optimizer == 'CoordMathLSTM':
            self.P = [p / num_test_batches for p in self.P]
            np.savetxt(os.path.join(opts.save_dir, 'P_S' +
                                    str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.P))
            self.A2 = [a2 / num_test_batches for a2 in self.A2]
            np.savetxt(os.path.join(opts.save_dir, 'A_S' +
                                    str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.A2))
        elif opts.optimizer == 'CoordMathDNN':
            self.P = [p / num_test_batches for p in self.P]
            np.savetxt(os.path.join(opts.save_dir, 'P_S' +
                                    str(opts.ood_s) + 'T'+str(opts.ood_t)), np.array(self.P))
            

def setup_logger(log_file):
    if log_file is not None:
        logging.basicConfig(filename=log_file, level=logging.INFO)
        lgr = logging.getLogger()
        lgr.addHandler(logging.StreamHandler())
        lgr = lgr.info
    else:
        lgr = print

    return lgr


class FileLogger(logging.Logger):
    '''
    From https://stackoverflow.com/a/76268417
    '''

    def __init__(self, name, filename, mode='a', level=logging.INFO, fformatter=None, log_to_console=False, sformatter=None):
        super().__init__(name, level)

        # Create a custom file handler
        self.file_handler = logging.FileHandler(filename=filename, mode=mode)

        # Set the formatter for the file handler
        if fformatter is not None:
            self.file_handler.setFormatter(fformatter)

        # Add the file handler to the logger
        self.addHandler(self.file_handler)

        if log_to_console:
            # Create a console handler
            self.console_handler = logging.StreamHandler()  # Prints to the console

            # Set the formatter for the console handler
            if not sformatter:
                sformatter = fformatter
            self.console_handler.setFormatter(sformatter)

            # Add the console handler to the logger
            self.addHandler(self.console_handler)

    def fdebug(self, msg, pre_msg=''):
        if pre_msg:
            print(pre_msg)
        self.debug(msg)

    def finfo(self, msg):
        self.info(msg)

    def fwarn(self, msg):
        self.warning(msg)

    def ferror(self, msg):
        self.error(msg)

    def fcritical(self, msg):
        self.critical(msg)


def setup_loss_logger(log_file):
    s_log_level = logging.CRITICAL
    file_log_level = logging.WARN
    log_format = "[%(asctime)s.%(msecs)03d] %(message)s"

    logging.basicConfig(format=log_format,
                        level=s_log_level, datefmt="%H:%M:%S")

    # Create an instance of the custom logger
    # formatter = logging.Formatter(log_format, "%H:%M:%S")
    fLogger = FileLogger('train_loss', log_file, mode='a',
                         level=file_log_level)

    return fLogger.ferror


def rand_WY(opts):
    def generate_WY(_batch_size, iterations, seed):
        Ws, Ys = [], []
        for i in range(iterations):
            # torch.manual_seed(seed + i)
            _W = torch.randn(_batch_size, opts.output_dim,
                             opts.input_dim, dtype=torch.float32)
            _W = _W / torch.sum(_W**2, dim=1, keepdim=True).sqrt()
            # torch.manual_seed(seed + i)
            _X_gt = torch.randn(_batch_size, opts.input_dim,
                                dtype=torch.float32)
            # torch.manual_seed(seed + i)
            non_zero_idx = torch.multinomial(
                torch.ones_like(_X_gt), num_samples=opts.sparsity, replacement=False
            )
            X_gt = torch.zeros_like(_X_gt).scatter(
                dim=1, index=non_zero_idx, src=_X_gt
            ).unsqueeze(-1)
            _Y = torch.bmm(_W, X_gt)
            Ws.append(_W)
            Ys.append(_Y)
        return torch.concat(Ws, dim=0), torch.concat(Ys, dim=0)

    W, Y = generate_WY(opts.train_batch_size,
                       opts.global_training_steps, opts.seed + 77)

    eval_W, eval_Y = generate_WY(opts.val_size, 1, opts.seed + 650)
    return W, Y, eval_W, eval_Y


def shrink_free(x, theta):
    """
    Soft Shrinkage function without the constraint that the thresholds must be
    greater than zero.
    From https://github.com/VITA-Group/LISTA-CPSS/blob/master/utils/tf.py
    """
    return x.sign() * F.relu(x.abs() - theta)


def shrink_ss(x, theta, q):
    """
    Special shrink that does not apply soft shrinkage to entries of top q%
    magnitudes.

    :inputs_: TODO
    :thres_: TODO
    :q: TODO
    :returns: TODO

    From https://github.com/VITA-Group/LISTA-CPSS/blob/master/utils/tf.py
    """
    abs_ = torch.abs(x)
    thres_ = torch.quantile(abs_, 1.0-q, dim=0, keepdim=True)

    """
    Entries that are greater than thresholds and in the top q% simultnaneously
    will be selected into the support, and thus will not be sent to the
    shrinkage function.
    """
    """Stop gradient at index_, considering it as constant."""
    index_ = torch.logical_and(
        abs_ > theta, abs_ > thres_).type(torch.float32).detach()
    cindex_ = 1.0 - index_  # complementary index

    return (index_ * x + shrink_free(cindex_*x, theta))


def script_genrater(file_name, config_dict, identical_config_dict, exepy_file, config_file, static_scripts, device, init_savedir):
    init_script = 'python ' + exepy_file + ' --config ' + config_file + \
        ' \\\n    ' + static_scripts + ' \\\n    ' + device + ' \\\n'
    # init_savedir = 'LASSO-L2O-PA-SgleLoss-DNN-DetachState'

    keys = list(config_dict.keys())

    def recursion(k_index, k_len, keys, config_dict, identical_config_dict, script_list, name_list):
        if k_index < k_len:
            current_script_list = []
            current_name_list = []
            for i, ps in enumerate(script_list):
                for cs in config_dict[keys[k_index]]:
                    current_script = ps + '    ' + \
                        keys[k_index] + ' ' + str(cs)
                    current_name = name_list[i]+keys[k_index] + str(cs)
                    if keys[k_index] in identical_config_dict:
                        current_script = current_script + '    ' + \
                            identical_config_dict[keys[k_index]
                                                  ] + ' ' + str(cs)
                        current_name = current_name + \
                            identical_config_dict[keys[k_index]] + str(cs)
                    current_script = current_script + ' \\\n'
                    current_script_list.append(current_script)
                    current_name_list.append(current_name)
            k_index += 1
            return recursion(k_index, k_len, keys, config_dict, identical_config_dict, current_script_list, current_name_list)
        else:
            return script_list, name_list

    script_list, name_list = recursion(0, len(keys), keys, config_dict, identical_config_dict, [
                                       init_script], [init_savedir])

    with open(file_name, 'a') as f:
        for i, s in enumerate(script_list):
            f.write(s + '    --save-dir ' + name_list[i] + ' \n\n')



def generate_adam_tune_script():
    # Test Adam Tune Beta1 and Beta2
    config_dict = {'--momentum1': [round(i, 1) for i in np.arange(0.1, 1.0, 0.2, dtype=np.float32)],
                   '--momentum2': [round(i, 3) for i in np.arange(0.95, 1.0, 0.005, dtype=np.float32)]}

    identical_config_dict = {}
    
    script_genrater('scripts/3_qp_baseline_adam_tune2.sh',
                    config_dict, identical_config_dict, 'main_train_anlys.py',
                    './configs/2_qp_testing.yaml',
                    '--optimizer Adam --optimizee-dir ./optimizees/matdata/qp-rand-512400 --input-dim 512 --output-dim 400 --test-length 5000 --load-mat --loss-save-path qp-rand',
                    '--device \"cuda:1\"',
                    'inference/Adam2/Adam')
    
def generate_math_l2o_train_script(lr=0.0001):
    # Test Adam Tune Beta1 and Beta2
    config_dict = {'--optimizer-training-steps': [i for i in range(5, 105, 5)]}

    identical_config_dict = {'--optimizer-training-steps': '--unroll-length'}
    
    # Train
    script_genrater(f'scripts/math_l2o_train/0_qp_math_l2o_p_unroll_length_moti_lr{lr}.sh',
                    config_dict, identical_config_dict, 'main_train_anlys.py',
                    './configs/1_qp_training_train_anlys.yaml',
                    f'--meta-optimizer SGD --init-lr {lr} --optimizer CoordMathLSTM --p-use --p-norm \"sigmoid\" --input-dim 512 --output-dim 400 --lstm-hidden-size 20 --global-training-steps 500 --train-batch-size 10 --val-size 10',
                    '--device \"cuda:1\"',
                    f'training/Math-L2O-P/QP-Math-L2O-P-SgleLoss-DetachState-Sigmoid-ZeroX0-lr{lr}')
    
    # script_genrater('scripts/0_qp_math_l2o_p_DNN_unroll_length_moti.sh',
    #                 config_dict, identical_config_dict, 'main_train_anlys.py',
    #                 './configs/1_qp_training_train_anlys.yaml',
    #                 '--p-use --optimizer CoordMathDNN --p-norm \"sigmoid\" --input-dim 512 --output-dim 400 ',
    #                 '--device \"cuda:1\"',
    #                 'training/Math-L2O/QP-Math-L2O-P-DNN-SgleLoss-DetachState-Sigmoid-RandX0')
    

def generate_lista_train_script(lr=0.0001):
    # Test Adam Tune Beta1 and Beta2
    config_dict = {'--optimizer-training-steps': [i for i in range(5, 105, 5)]}

    identical_config_dict = {'--optimizer-training-steps': '--unroll-length'}
    
    # Train
    script_genrater(f'scripts/lista_train/0_qp_lista_unroll_length_lr{lr}.sh',
                    config_dict, identical_config_dict, 'main_train_anlys.py',
                    './configs/1_qp_training_train_anlys.yaml',
                    f'--meta-optimizer SGD --init-lr {lr} --optimizer LISTACPSSWOnly --input-dim 512 --output-dim 400 --rho 0.2 --lamb 0.4 --global-training-steps 500 --train-batch-size 10 --val-size 10',
                    '--device \"cuda:0\"',
                    f'training/LISTACPSS/LISTACPSSWOnly-ZeroX0-lr{lr}')


def generate_our_train_script(lr=0.0001):
    # Test Adam Tune Beta1 and Beta2
    config_dict = {'--optimizer-training-steps': [i for i in range(5, 105, 5)]}

    identical_config_dict = {'--optimizer-training-steps': '--unroll-length'}
    
    # Train
    script_genrater(f'scripts/our_train/0_qp_l2o_p_unroll_length_moti_lr{lr}.sh',
                    config_dict, identical_config_dict, 'main_train_anlys.py',
                    './configs/1_qp_training_train_anlys.yaml',
                    f'--meta-optimizer SGD --init-lr {lr} --optimizer CoordMathDNN --p-use --p-norm \"sigmoid\" --input-dim 512 --output-dim 400 --lstm-hidden-size 5120 --global-training-steps 500 --train-batch-size 10 --val-size 10',
                    '--device \"cuda:1\"',
                    f'training/Our/QP-Our-L2O-PA-SgleLoss-DetachState-Sigmoid-ZeroX0-lr{lr}')
    
    
if __name__ == "__main__":
    # config_dict = {'--init-lr': [0.01, 0.001, 0.0001],
    #                '--optimizer-training-steps': [20, 50, 100]}

    # identical_config_dict = {'--optimizer-training-steps': '--unroll-length'}

    # # Generate Gaussian scripts
    # # Train
    # script_genrater('scripts/train_anlys/1_lasso_l2o_pa_activaton_iGaussian_randx0.sh',
    #                 config_dict, 'main_train_anlys.py',
    #                 './configs/1_lasso_training_train_anlys.yaml',
    #                 '--p-use --a-use --optimizer CoordMathDNN --p-norm \"Gaussian\" --a-norm \"Gaussian\"',
    #                 '--device \"cuda:0\"',
    #                 'train_anlys/activation/LASSO-L2O-PA-SgleLoss-DNN-DetachState-Gaussian-RandX0')
    # # Test
    # script_genrater('scripts/train_anlys/1_lasso_l2o_pa_activation_Gaussian_randx0.sh',
    #                 config_dict, 'main_train_anlys.py',
    #                 './configs/2_lasso_testing.yaml',
    #                 '--optimizee-dir ./optimizees/matdata/lasso-rand --load-mat --load-sol --p-use --a-use --optimizer CoordMathDNN --p-norm \"Gaussian\" --a-norm \"Gaussian\"',
    #                 '--device \"cuda:0\"',
    #                 'train_anlys/activation/LASSO-L2O-PA-SgleLoss-DNN-DetachState-Gaussian-RandX0')

    # # Generate Sigmoid scripts
    # # Train
    # script_genrater('scripts/train_anlys/1_lasso_l2o_pa_activation_sigmoid_randx0.sh',
    #                 config_dict, 'main_train_anlys.py',
    #                 './configs/1_lasso_training_train_anlys.yaml',
    #                 '--p-use --a-use --optimizer CoordMathDNN --p-norm \"sigmoid\" --a-norm \"sigmoid\"',
    #                 '--device \"cuda:1\"',
    #                 'train_anlys/activation/LASSO-L2O-PA-SgleLoss-DNN-DetachState-sigmoid-RandX0')
    # # Test
    # script_genrater('scripts/train_anlys/1_lasso_l2o_pa_activation_sigmoid_randx0.sh',
    #                 config_dict, 'main_train_anlys.py',
    #                 './configs/2_lasso_testing.yaml',
    #                 '--optimizee-dir ./optimizees/matdata/lasso-rand --load-mat --load-sol --p-use --a-use --optimizer CoordMathDNN --p-norm \"sigmoid\" --a-norm \"sigmoid\"',
    #                 '--device \"cuda:1\"',
    #                 'train_anlys/activation/LASSO-L2O-PA-SgleLoss-DNN-DetachState-sigmoid-RandX0')

    # QP problem
    # Generate Gaussian scripts
    # Train
    # script_genrater('scripts/train_anlys/1_qp_l2o_pa_activation_Gaussian_randx0.sh',
    #                 config_dict, identical_config_dict, 'main_train_anlys.py',
    #                 './configs/1_qp_training_train_anlys.yaml',
    #                 '--p-use --a-use --optimizer CoordMathDNN --p-norm \"Gaussian\" --a-norm \"Gaussian\"',
    #                 '--device \"cuda:0\"',
    #                 'train_anlys/QP/QP-L2O-PA-SgleLoss-DNN-DetachState-Gaussian-RandX0')
    # # Test
    # script_genrater('scripts/train_anlys/1_qp_l2o_pa_activation_Gaussian_randx0.sh',
    #                 config_dict, identical_config_dict, 'main_train_anlys.py',
    #                 './configs/2_qp_testing.yaml',
    #                 '--optimizee-dir ./optimizees/matdata/lasso-rand --load-mat --load-sol --p-use --a-use --optimizer CoordMathDNN --p-norm \"Gaussian\" --a-norm \"Gaussian\"',
    #                 '--device \"cuda:0\"',
    #                 'train_anlys/QP/QP-L2O-PA-SgleLoss-DNN-DetachState-Gaussian-RandX0')

    # # Generate Sigmoid scripts
    # # Train
    # script_genrater('scripts/train_anlys/1_qp_l2o_pa_activation_sigmoid_randx0.sh',
    #                 config_dict, identical_config_dict, 'main_train_anlys.py',
    #                 './configs/1_qp_training_train_anlys.yaml',
    #                 '--p-use --a-use --optimizer CoordMathDNN --p-norm \"sigmoid\" --a-norm \"sigmoid\"',
    #                 '--device \"cuda:1\"',
    #                 'train_anlys/QP/QP-L2O-PA-SgleLoss-DNN-DetachState-sigmoid-RandX0')
    # # Test
    # script_genrater('scripts/train_anlys/1_qp_l2o_pa_activation_sigmoid_randx0.sh',
    #                 config_dict, identical_config_dict, 'main_train_anlys.py',
    #                 './configs/2_qp_testing.yaml',
    #                 '--optimizee-dir ./optimizees/matdata/lasso-rand --load-mat --load-sol --p-use --a-use --optimizer CoordMathDNN --p-norm \"sigmoid\" --a-norm \"sigmoid\"',
    #                 '--device \"cuda:1\"',
    #                 'train_anlys/QP/QP-L2O-PA-SgleLoss-DNN-DetachState-sigmoid-RandX0')

    
    # generate_adam_tune_script()
    
    # Generate ``Math-L2O`` training ablation scripts, Test Math-L2O Training Performance
    # generate_math_l2o_train_script(0.001)
    # generate_math_l2o_train_script(0.0001)
    # generate_math_l2o_train_script(0.00001)
    # generate_math_l2o_train_script(0.000001)
    # generate_math_l2o_train_script(0.0000001)
    
    # Generate ``Math-L2O`` training ablation scripts, Test Math-L2O NaN on QP
    # for i in np.arange(0.00001, 0.0001, 0.00001):
    #     i = round(i, 5)
    #     generate_math_l2o_train_script(i)
    
    
    # Generate ``LISTA`` training ablation scripts:
    # generate_lista_train_script(0.001)
    # generate_lista_train_script(0.0001)
    # generate_lista_train_script(0.00001)
    # generate_lista_train_script(0.000001)
    # generate_lista_train_script(0.0000001)
    
    # Generate ``LISTA`` training ablation scripts, Test Math-L2O NaN on QP
    for i in np.arange(0.00001, 0.0001, 0.00001):
        i = round(i, 5)
        generate_lista_train_script(i)

    # Generate ``Our`` training ablation scripts:
    # generate_our_train_script(0.001)
    # generate_our_train_script(0.0001)
    # generate_our_train_script(0.00001)
    # generate_our_train_script(0.000001)
    # generate_our_train_script(0.0000001)