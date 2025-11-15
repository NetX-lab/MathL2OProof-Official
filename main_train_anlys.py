import os
import numpy as np
from config_parser import *
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import random
import utils
import optimizers

import time

from optimizees import OPTIMIZEE_DICT

from utils.util_train_loss_saver import *
from utils.utils import TrainingDebugger
from torch.utils.tensorboard import SummaryWriter

opts, _ = parser.parse_known_args()

# Save directory
opts.save_dir = os.path.join('results', opts.save_dir)
if not os.path.isdir(opts.save_dir):
    os.makedirs(opts.save_dir)
# Logging file
logger_file = os.path.join(opts.save_dir, 'train.log')
opts.logger = utils.setup_logger(logger_file)
opts.logger('Checkpoints will be saved to directory `{}`'.format(opts.save_dir))
opts.logger(
    'Log file for training will be saved to file `{}`'.format(logger_file))

# logger training loss
logger_file_train_loss = os.path.join(opts.save_dir, 'train_loss.log')
opts.logger_train_loss = utils.setup_loss_logger(logger_file_train_loss)

# Use cuda if it is available
if opts.cpu:
    opts.device = 'cpu'
elif opts.device is None:
    if torch.cuda.is_available():
        opts.device = 'cuda'
    else:
        opts.device = 'cpu'
        opts.logger('WARNING: No CUDA available. Run on CPU instead.')
# Output the type of device used
opts.logger('Using device: {}'.format(opts.device))
opts.dtype = torch.float
# opts.logger('Using tau: {}'.format(opts.tau)) # Output the tau used in current exp

# Set random seed for reproducibility
torch.manual_seed(opts.seed)
random.seed(opts.seed + 7)
np.random.seed(opts.seed + 42)

if opts.unroll_length > opts.optimizer_training_steps:
    opts.unroll_length = opts.optimizer_training_steps
# -----------------------------------------------------------------------
#              Create data for training and validation
# -----------------------------------------------------------------------
# train_seen_loader, val_seen_loader, test_seen_loader, A, W, W_gram, G = create_sc_data(opts)
# A_TEN = torch.from_numpy(A).to(device=opts.device, dtype=opts.dtype)

if opts.identical_dict:
    torch.manual_seed(opts.seed + 777)
    if opts.fixed_dict:
        # W = torch.randn(1, opts.output_dim, opts.input_dim).to(opts.device)
        # Version 1: Fixed M
        # W = torch.normal(0, 1.0 / np.sqrt(opts.output_dim),
        #                  size=(1, opts.output_dim, opts.input_dim)).to(opts.device)
        # Version 2: Random M with mean initialization for We in LISTA.
        W = torch.randn(1, opts.output_dim, opts.input_dim).to(opts.device)
    else:
        W = torch.randn(opts.train_batch_size, opts.output_dim,
                        opts.input_dim).to(opts.device)
else:
    W = None

# Keyword arguments for the optimizees
optimizee_kwargs = {
    'input_dim': opts.input_dim,
    'output_dim': opts.output_dim,
    'rho': opts.rho,
    's': opts.sparsity,
    'device': opts.device,
    'ood': opts.ood,
    'ood_s': opts.ood_s,
    'ood_t': opts.ood_t,
    'dtype': opts.dtype,
}

# Keyword artuments for the optimizers
optimizer_kwargs = {
    'a_use': opts.a_use,
    'a_scale': opts.a_scale,
    'a_scale_learned': opts.a_scale_learned,
    'a_norm': opts.a_norm,

    'p_use': opts.p_use,
    'p_scale': opts.p_scale,
    'p_scale_learned': opts.p_scale_learned,
    'p_norm': opts.p_norm,

}

reset_state_kwargs = {
    'state_scale': opts.state_scale,
    # 'step_size': opts.step_size,
    'momentum1': opts.momentum1,
    'momentum2': opts.momentum2,
    'eps': opts.eps,
    'hyper_step': opts.hyper_step,
    'B_step_size': opts.B_step_size,
    'C_step_size': opts.C_step_size,
}


if opts.optimizer == 'ProximalGradientDescent':
    optimizer = optimizers.ProximalGradientDescent()
elif opts.optimizer == 'ProximalGradientDescentMomentum':
    optimizer = optimizers.ProximalGradientDescentMomentum()
elif opts.optimizer == 'SubGradientDescent':
    optimizer = optimizers.SubGradientDescent()
elif opts.optimizer == 'Adam':
    optimizer = optimizers.Adam()
elif opts.optimizer == 'AdamHD':
    optimizer = optimizers.AdamHD()
elif opts.optimizer == 'Shampoo':
    optimizer = optimizers.Shampoo()
elif opts.optimizer == 'CoordMathLSTM':
    optimizer = optimizers.CoordMathLSTM(
        input_size=2,
        output_size=1,
        hidden_size=opts.lstm_hidden_size,
        layers=opts.lstm_layers,
        **optimizer_kwargs
    )
elif opts.optimizer == 'CoordMathDNN':
    optimizer = optimizers.CoordMathDNN(
        input_size=2,
        output_size=1,
        hidden_size=opts.lstm_hidden_size,
        layers=opts.lstm_layers,
        e=opts.e,
        **optimizer_kwargs
    )
    # Determinisctically initialize former layer to be non-singular, last layer to be zero.
    # optimizer.dnn.apply(optimizer.init_non_singular)
    # optimizer.linear.apply(optimizer.init_non_singular)
    # optimizer.linear_p.apply(optimizer.init_zero)

elif opts.optimizer == 'RNNprop':
    optimizer = optimizers.RNNprop(
        input_size=2,
        output_size=1,
        hidden_size=opts.lstm_hidden_size,
        layers=opts.lstm_layers,
        beta1=opts.rnnprop_beta1,
        beta2=opts.rnnprop_beta2,
        **optimizer_kwargs
    )
elif opts.optimizer == 'CoordBlackboxLSTM':
    optimizer = optimizers.CoordBlackboxLSTM(
        input_size=2,
        output_size=1,
        hidden_size=opts.lstm_hidden_size,
        layers=opts.lstm_layers,
        **optimizer_kwargs
    )
elif opts.optimizer == 'CoordBlackboxDNN':
    optimizer = optimizers.CoordBlackboxDNN(
        input_size=2,
        output_size=1,
        hidden_size=opts.lstm_hidden_size,
        layers=opts.lstm_layers,
        **optimizer_kwargs
    )
elif opts.optimizer == 'LISTA':
    optimizer = optimizers.LISTA(
        input_dim=opts.input_dim,
        output_dim=opts.output_dim,
        T=opts.optimizer_training_steps,
        lamb=opts.lamb,
        percent=opts.p,
        max_percent=opts.max_p,
        # For alignment with our L2O training:
        A=W / torch.sum(W**2, dim=1, keepdim=True).sqrt(),
        We_shared=opts.w_shared,
        S_shared=opts.s_shared,
        theta_shared=opts.theta_shared
    )
elif opts.optimizer == 'LISTACPSS':
    optimizer = optimizers.LISTACPSS(
        input_dim=opts.input_dim,
        output_dim=opts.output_dim,
        T=opts.optimizer_training_steps,
        lamb=opts.lamb,
        percent=opts.p,
        max_percent=opts.max_p,
        # For alignment with our L2O training:
        A=W / torch.sum(W**2, dim=1, keepdim=True).sqrt(),
        We_shared=opts.w_shared,
        theta_shared=opts.theta_shared
    )
elif opts.optimizer == 'LISTACPSSSTEP':
    optimizer = optimizers.LISTACPSSSTEP(
        input_dim=opts.input_dim,
        output_dim=opts.output_dim,
        T=opts.optimizer_training_steps,
        lamb=opts.lamb,
        percent=opts.p,
        max_percent=opts.max_p,
        # For alignment with our L2O training:
        A=W / torch.sum(W**2, dim=1, keepdim=True).sqrt(),
        We_shared=opts.w_shared,
        theta_shared=opts.theta_shared
    )
elif opts.optimizer == 'LISTACPSSWOnly':
    optimizer = optimizers.LISTACPSSWOnly(
        input_dim=opts.input_dim,
        output_dim=opts.output_dim,
        T=opts.optimizer_training_steps,
        percent=opts.p,
        max_percent=opts.max_p,
        # For alignment with our L2O training:
        A=W,
        We_shared=opts.w_shared
    )
else:
    raise ValueError(f'Invalid optimizer name {opts.optimizer}')


def unroll_BP(unroll_length, optimizer_training_steps, meta_optimizer, optimizer, optimizees, opts, training_losses_logs):
    """
    BP every unroll_length.
    """
    is_nan = False
    training_losses_per_batch = []

    global_loss = 0.0

    # loss_last = None
    if opts.loss_func == 'mean':
        all_weights = unroll_length
    elif opts.loss_func == 'weighted_sum':
        all_weights = sum(range(1, unroll_length + 1))

    # start = timer()
    # Add starting
    training_losses_per_batch.append(optimizees.objective(
        compute_grad=False).detach().cpu().item())
    for j in range(optimizer_training_steps):
        # if j > 0:
        #     print("---Unroll step", j)
        #     print("Learned Step Size", optimizer.P.mean())
        #     print("L", optimizer.step_size.mean())
        optimizees = optimizer(optimizees, opts.grad_method)
        # opts.logger("global step {} num roll {} unroll length {} X norm {}".format(
        #     i, num_roll, j, torch.mean(optimizees.X)))
        loss = optimizees.objective(compute_grad=True)
        # print("step", j, "loss", loss)
        if torch.isinf(loss) or torch.isnan(loss):
            is_nan = True
            break
        else:
            training_losses_per_batch.append(loss.detach().cpu().item())
            if (j+1) % 10 == 1:
                training_losses_logs["iter-" +
                                     str(j)].append(loss.detach().cpu().item())
        if opts.loss_func == 'last':
            continue
        elif opts.loss_func == 'mean':
            global_loss += loss
        elif opts.loss_func == 'weighted_sum':
            global_loss += loss * (j+1)
        else:
            raise ValueError(
                f'Invalid loss function name {opts.loss_func}')
        # opts.logger('{} {} {}'.format(num_roll, j, loss))
    if opts.loss_func == 'last':
        global_loss = loss
    else:
        global_loss = global_loss / all_weights

    if not is_nan:
        meta_optimizer.zero_grad()
        global_loss.backward()  # retain_graph=True
        if opts.clip_grad:
            torch.nn.utils.clip_grad_norm_(
                optimizer.parameters(), 1.0, error_if_nonfinite=False)
        # print(i, num_roll, norm)
        meta_optimizer.step()
        training_losses_per_batch.append(global_loss.detach().cpu().item())

        # Clean up the current unrolling segments, including:
        # - Detach the current hidden and cell states.
        # - Clear the `hist` list of the optimizers to release memory
        optimizer.detach_state()
        optimizees.detach_vars()

        # optimizer.clear_hist()

        # time = timer() - start
        # if verbose:
        #     opts.logger(
        #         '--> time consuming [{:.4f}s] optimizer train steps :  [{}] '
        #         '| Global_Loss = [{:.4f}]'.format(
        #             time,
        #             optimizer_training_steps,
        #             training_losses_per_batch[-1]
        #         )
        #     )
    return is_nan, optimizer, meta_optimizer, training_losses_per_batch


if not opts.test:
    torch.autograd.set_detect_anomaly(True)
    config_path = os.path.join(opts.save_dir, 'config.yaml')
    parser.write_config_file(opts, [config_path])

    assert isinstance(
        optimizer, nn.Module), 'Only PyTorch Modules need training.'

    optimizer = optimizer.to(device=opts.device, dtype=opts.dtype)
    # , weight_decay=1e-3
    if opts.meta_optimizer == 'Adam':
        meta_optimizer = optim.Adam(
            optimizer.parameters(), lr=opts.init_lr)
    elif opts.meta_optimizer == 'SGD':
        meta_optimizer = optim.SGD(
            optimizer.parameters(), lr=opts.init_lr)
    else:
        raise NotImplementedError
    
    if opts.scheduler == 'cosine':
        meta_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            meta_optimizer, T_max=opts.global_training_steps, eta_min=1e-5)
    elif opts.scheduler == 'constant':
        meta_scheduler = optim.lr_scheduler.ConstantLR(
            meta_optimizer, factor=1.0, total_iters=opts.global_training_steps)
    elif opts.scheduler == 'step':
        meta_scheduler = optim.lr_scheduler.StepLR(
            meta_optimizer, step_size=1, gamma=0.1)
    else:
        raise NotImplementedError

    # initialize the array storing training loss function
    training_losses = []
    best_validation_mean = 99999999999999
    best_validation_final = 99999999999999

    # if opts.debug:
    writer = SummaryWriter(opts.save_dir)
    if opts.debug:
        debuger = TrainingDebugger(opts.optimizer)
    eval_step = 0

    for epoch in range(opts.epochs):
        # if e >= 2:
        #     opts.optimizer_training_steps = 30
        #     opts.unroll_length = 30

        for i in range(opts.global_training_steps):
            seed = opts.seed + 777  # * (i+1)
            # if (i+1) % opts.print_freq == 0:
            # opts.logger('\n=============> Epoch: {} LR: {}'.format(
            #     epoch, meta_scheduler.get_last_lr()))
            # opts.logger(
            #     '\n=============> global training steps: {}'.format(i))
            # if i > 0:
            # print('\n=============> global training steps: {}'.format(i))
            optimizer.train()

            optimizees = OPTIMIZEE_DICT[opts.optimizee_type](
                opts.train_batch_size, W, seed=seed, **optimizee_kwargs
            )
            # writer.add_scalar('train/L_smoothness_mean',
            #                   optimizees.grad_lipschitz().mean(), i)
            optimizer.reset_state(
                optimizees, opts.step_size, **reset_state_kwargs)

            # Use dict to log loss at different GD iteration
            training_losses_log = {}
            for j in range(opts.optimizer_training_steps):
                if (j+1) % 10 == 1:
                    training_losses_log["iter-" + str(j)] = []

            is_nan, optimizer, meta_optimizer, training_losses_per_batch \
                = unroll_BP(opts.unroll_length, opts.optimizer_training_steps,
                            meta_optimizer, optimizer, optimizees, opts, training_losses_log)
            # if is_nan:
            #     opts.logger(
            #         '\n=============> Objective NaN 1st Time, try {}-{}'.format(5, opts.optimizer_training_steps))
            #     is_nan, optimizer, meta_optimizer = unroll_BP(5, opts.optimizer_training_steps,
            #                                                   meta_optimizer, optimizer, optimizees, opts, verbose)

            # if is_nan:
            #     opts.logger(
            #         '\n=============> Objective NaN 2nd Time, try {}-{}'.format(3, 60))
            #     is_nan, optimizer, meta_optimizer = unroll_BP(3, 60,
            #                                                   meta_optimizer, optimizer, optimizees, opts, verbose)

            # if is_nan:
            #     opts.logger(
            #         '\n=============> Objective NaN 3nd Time, try {}-{}'.format(1, 20))
            #     is_nan, optimizer, meta_optimizer = unroll_BP(1, 20,
            #                                                   meta_optimizer, optimizer, optimizees, opts, verbose)

            # if is_nan:
            #     opts.logger(
            #         '\n=============> Objective NaN 3nd Time, try {}-{}'.format(1, 5))
            #     is_nan, optimizer, meta_optimizer = unroll_BP(1, 5,
            #                                                   meta_optimizer, optimizer, optimizees, opts, verbose)

            if is_nan:
                opts.logger(
                    '\n=============> Iteration {} All NaN !!!'.format(i))
            # Log training monitorings
            # for j in range(len(training_losses_per_batch)):
            #     writer.add_scalar(
            #         'train/train_loss_' + str(opts.unroll_length*(j+1)), training_losses_per_batch[j], i)
                info = 'Epoch {} Iteration {}, training loss NaN\n'.format(
                    epoch, i)
                # print(info)
                opts.logger_train_loss(info)
                training_losses.append(np.nan)
            else:
                # NOTE: Printed loss is the last loss of BP unrolling_length!!

                training_losses.append(training_losses_per_batch[-1])
                info = 'Epoch {} Iteration {}, training loss {}\n'.format(epoch,
                                                                          i, training_losses_per_batch[-1])
                # print(info)
                opts.logger_train_loss(info)

                if (i+1) % 100 == 1:
                    training_losses_per_batch.pop()
                    save_obj_traj(training_losses_per_batch, i, opts.save_dir)

                # for j in range(opts.optimizer_training_steps):
                #     if (j+1) % 10 == 1:
                #         training_losses_log["iter-" + str(j)] = np.array(
                #             training_losses_log["iter-" + str(j)])
                # writer.add_scalars(
                #     'train_loss', training_losses_log, epoch * opts.global_training_steps + i)
            if (i+1) % opts.val_freq == 0:
                # optimizer.eval()
                # optimizees = OPTIMIZEE_DICT[opts.optimizee_type](
                #     opts.val_size, W, seed=opts.seed + 77, **optimizee_kwargs
                # )
                # optimizer.reset_state(
                #     optimizees, opts.step_size, **reset_state_kwargs)
                # validation_losses = []
                # validation_grad = []
                # validation_grad_dirc = []
                # if opts.debug:
                #     debuger.add(opts, optimizer)
                # for j in range(opts.val_length):
                #     # Fixed data samples for validation
                #     optimizees = optimizer(optimizees, opts.grad_method)
                #     # # L1-norm + l1_lambda * l1_norm
                #     # l1_lambda = 0.1
                #     # l1_norm = sum(p.abs().sum() for p in optimizer.parameters())
                #     loss = optimizees.objective()
                #     validation_losses.append(loss.detach().cpu().item())
                #     grad = optimizees.get_grad(
                #         grad_method='subgrad',
                #         compute_grad=False,
                #         retain_graph=False
                #     )
                #     validation_grad.append(torch.linalg.norm(
                #         grad, dim=(-2, -1), ord='fro', keepdim=False).mean().detach().cpu().item())
                #     validation_grad_dirc.append(torch.sign(
                #         grad).sum((-2, -1)).mean().detach().cpu().item())

                # if opts.debug:
                #     debuger.log(opts, writer, validation_losses,
                #                 validation_grad, validation_grad_dirc, eval_step)
                #     eval_step += 1
                # # if (validation_losses[-1] < best_validation_final and
                # #         np.mean(validation_losses) < best_validation_mean) :
                # if np.mean(validation_losses) < best_validation_mean:
                #     best_validation_final = validation_losses[-1]
                #     best_validation_mean = np.mean(validation_losses)
                #     opts.logger(
                #         '\n\n===> best of final LOSS[{}]:={} LOSS[{}]:={} LOSS[{}]:={}, '
                #         'best_mean_loss ={}'.format(
                #             30, validation_losses[29],
                #             50, validation_losses[49],
                #             100, validation_losses[99],
                #             best_validation_mean
                #         )
                #     )

                checkpoint_name = optimizer.name() + '.pth'
                save_path = os.path.join(opts.save_dir, checkpoint_name)
                torch.save(optimizer.state_dict(), save_path)
                opts.logger('Saved the optimizer to file: ' + save_path)
        meta_scheduler.step()

    writer.close()
    # Save training loss
    train_loss_path = os.path.join(opts.save_dir, 'train_loss')
    opts.logger(f'training losses saved to {train_loss_path}')
    np.savetxt(train_loss_path, np.array(training_losses))

else:
    opts.logger(f'********* OOD state {str(opts.ood)} *********')
    if opts.debug:
        debuger = utils.TestingDebugger(opts)
    if isinstance(optimizer, nn.Module):
        checkpoint_name = optimizer.name() + '.pth'
        if not opts.ckpt_path:
            opts.ckpt_path = os.path.join(opts.save_dir, checkpoint_name)
        optimizer.load_state_dict(torch.load(
            opts.ckpt_path, map_location='cpu'))
        opts.logger(f'Trained weight loaded from {opts.ckpt_path}')
        optimizer.to(device=opts.device, dtype=opts.dtype).eval()
        optimizer.eval()

    if not opts.test_batch_size:
        opts.test_batch_size = opts.test_size

    num_test_batches = opts.test_size // opts.test_batch_size
    test_losses = [0.0] * (opts.test_length + 1)
    if opts.save_sol:
        test_losses_batch = np.zeros(
            (opts.test_length + 1, opts.test_batch_size))

    time_start = time.time()
    time_opt = 0

    # # Keep the dimension W, will be re-written by load-mat option
    # if opts.fixed_dict:
    #     # W = W.repeat(opts.test_batch_size, 1, 1)
    #     W = torch.randn(opts.test_batch_size, opts.output_dim,
    #                         opts.input_dim).to(opts.device)

    with torch.no_grad():  # clean computation graph
        for i in range(num_test_batches):
            seed = opts.seed + 7777 * (i+1)

            optimizees = OPTIMIZEE_DICT[opts.optimizee_type](
                opts.test_batch_size, W, seed=seed, **optimizee_kwargs
            )

            if opts.load_mat:
                optimizees.load_from_file(
                    opts.optimizee_dir + '/' + str(i) + '.mat', 0, opts.test_batch_size)
                opts.logger('Batch {} optimizee loaded.'.format(i))

            if opts.load_sol:
                optimizees.load_sol(opts.optimizee_dir + '/sol_' +
                                    str(i) + '.mat', 0, opts.test_batch_size)
                opts.logger('Batch {} optimal objective loaded.'.format(i))

            if opts.save_to_mat:
                if not os.path.exists(opts.optimizee_dir):
                    os.makedirs(opts.optimizee_dir, exist_ok=True)
                optimizees.save_to_file(
                    opts.optimizee_dir + '/' + str(i) + '.mat')

            optimizer.reset_state(
                optimizees, opts.step_size, **reset_state_kwargs)
            if not opts.load_sol:
                test_losses[0] += optimizees.objective().detach().cpu().item()
            else:
                # test_losses[0] += optimizees.objective().detach().cpu().item()
                test_losses[0] += optimizees.objective_shift().detach().cpu().item()

            if opts.save_sol:
                test_losses_batch[0] = optimizees.objective_batch(
                ).cpu().numpy()

            for j in range(opts.test_length):
                time_inner_start = time.time()
                optimizees = optimizer(optimizees, opts.grad_method)
                optimizer.detach_state()
                time_inner_end = time.time()
                time_opt += (time_inner_end - time_inner_start)

                if not opts.load_sol:
                    loss = optimizees.objective()
                else:
                    # loss = optimizees.objective()
                    loss = optimizees.objective_shift()

                test_losses[j+1] += loss.detach().cpu().item()
                if opts.save_sol:
                    test_losses_batch[j +
                                      1] = optimizees.objective_batch().cpu().numpy()
                if opts.debug:
                    debuger.add(opts, optimizer, j)
            opts.logger('Batch {} completed.'.format(i))

        if opts.save_sol:
            if not os.path.exists(opts.optimizee_dir):
                os.makedirs(opts.optimizee_dir, exist_ok=True)
            obj_star = np.min(test_losses_batch, axis=0)
            optimizees.save_sol(
                obj_star, opts.optimizee_dir + '/sol_' + str(i) + '.mat')
            opts.logger('Batch {} optimal objective saved.'.format(i))

    time_end = time.time()
    test_losses = [loss / num_test_batches for loss in test_losses]

    if opts.debug:
        debuger.log(opts, num_test_batches)

    # output the epoch results to the terminal
    opts.logger('Testing losses:')
    for ii, t_loss in enumerate(test_losses):
        opts.logger('{}, {}'.format(ii, t_loss))
    if not opts.loss_save_path:
        opts.loss_save_path = os.path.join(opts.save_dir, 'test_losses.txt')
    else:
        opts.loss_save_path = os.path.join(opts.save_dir, opts.loss_save_path)
    opts.logger(f'testing losses saved to {opts.loss_save_path}')
    np.savetxt(opts.loss_save_path, np.array(test_losses))

    opts.logger("Total time: {}".format(time_end - time_start))
    opts.logger("Time (opt iteration): {}".format(time_opt))
    opts.logger("Time per iter per instance: {}".format(
        time_opt / opts.test_length / opts.test_size))
