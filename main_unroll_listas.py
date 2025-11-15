import os
import numpy as np
import configargparse
from timeit import default_timer as timer

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import optimizers
import random
import utils

from optimizees import OPTIMIZEE_DICT

from config_parser import *
from utils.util_train_loss_saver import *

# Argument Parsing
# parser = configargparse.get_arg_parser(
#     description='Configurations for ALISTA experiement')

parser.add('-c', '--config', is_config_file=True, help='Config file path.')

parser.add('--optimizer', type=str, metavar='STR',
           help='What optimizer to use for the current experiment.')
parser.add('--cpu', action='store_true',
           help='Force to use CPU instead of GPU even if CUDA compatible GPU '
                'devices are available.')
parser.add('--test', action='store_true', help='Run in test mode.')
parser.add('--device', type=str, default=None, help='cuda:0')

# Optimizee general options
parser.add('--optimizee-type',
           choices=['QuadraticUnconstrained',
                    'LASSO', 'LASSO_LISTA', 'LogisticL1'],
           help='Type of optimizees to be trained on')
parser.add('--input-dim', type=int, metavar='INT',
           help='Dimension of the input (optimization variable)')
parser.add('--output-dim', type=int, metavar='INT',
           help='Dimension of the output (labels used to calculate loss)')
parser.add('--rho', type=float, default=0.2, metavar='FLOAT',
           help='Parameter for reg. term in the objective function.')
parser.add('--lamb', type=float, default=0.4, metavar='FLOAT',
           help='Parameter for reg. term in the objective function.')
parser.add('--sparsity', type=int, default=5, metavar='INT',
           help='Sparisty of the input variable.')
parser.add('--W-cond-factor', type=float, default=0.0, metavar='FLOAT',
           help='W: The ratio of randn and ones.')
parser.add('--x-mag', type=float, default=1.0, metavar='FLOAT',
           help='x: magnitude of nonzeros in x.')
parser.add('--W-cond-rand', action='store_true',
           help='Using random W-cond-factor in training.')
parser.add('--dist-rand', action='store_true',
           help='W-cond, x-mag, s and rho are generated randomly.')
parser.add('--save-to-mat', action='store_true',
           help='save optmizees to mat file.')
parser.add('--optimizee-dir', type=str, metavar='STR',
           help='dir of optimizees.')
parser.add('--load-mat', action='store_true',
           help='load optmizees from mat file.')
parser.add('--save-sol', action='store_true',
           help='save solutions of optimizees.')
parser.add('--load-sol', action='store_true',
           help='save solutions of optimizees.')
parser.add('--pb', type=float, default=0.1, metavar='FLOAT',
           help='Probability of non-zero Bernoulli for each entry in Distionary')

# Unconstrained Quadratic
parser.add('--fixed-dict', action='store_true',
           help='Use a fixed dictionary for the optimizees')

# Model parameters
parser.add('--layers', type=int, default=20, metavar='INT',
           help='Number of layers of the neural network')
parser.add('--symm', action='store_true',
           help='Use the new symmetric matrix parameterization')
parser.add('--step-size', type=float, default=None, metavar='FLOAT',
           help='Step size for the classic optimizers')
parser.add('--p', type=float, default=0.012, metavar='FLOAT',
           help='Percent of support selection in LISTA-CPSS')
parser.add('--max-p', type=float, default=0.13, metavar='FLOAT',
           help='Max percent of support selection in LISTA-CPSS')

# Data parameters
parser.add('--seed', type=int, default=118, metavar='INT',
           help='Random seed for reproducibility')

# Training parameters
parser.add('--train-objective',
           type=str, default='GT', metavar='{OBJECTIVE,L2,L1,GT}',
           help='Objective used for the training')
parser.add('--save-dir', type=str, default='temp',
           help='Saving directory for saved models and logs')
parser.add('--ckpt-path', type=str, default=None, metavar='STR',
           help='Path to the checkpoint to be loaded.')
parser.add('--loss-save-path', type=str, default=None, metavar='STR',
           help='Path to save the testing losses.')
parser.add('--train-size', type=int, default=32000, metavar='N',
           help='Number of training samples')
parser.add('--val-size', type=int, default=128, metavar='N',
           help='Number of validation samples')
parser.add('--test-size', type=int, default=1024, metavar='N',
           help='Number of testing samples')
parser.add('--train-batch-size', type=int, default=256, metavar='N',
           help='Batch size for training')
parser.add('--val-batch-size', type=int, default=128, metavar='N',
           help='Batch size for validation')
parser.add('--test-batch-size', type=int, default=32, metavar='N',
           help='Batch size for testing')
parser.add('--init-lr', type=float, default=0.1, metavar='FLOAT',
           help='Initial learning rate')
parser.add('--lr-decay-layer', type=float, default=0.3, metavar='FLOAT',
           help='Decay learning rates of trained layers')
parser.add('--lr-decay-stage2', type=float, default=0.2,
           metavar='FLOAT', help='Decay rate for training stage2 in each layer')
parser.add('--lr-decay-stage3', type=float, default=0.02, metavar='FLOAT',
           help='Decay rate for training stage3 in each layer')
parser.add('--best-wait', type=int, default=5, metavar='N',
           help='Wait time for better validation performance')

# Training
parser.add('--global-training-steps', type=int, default=1000,
           help='Total number of training steps considered.')
parser.add('--optimizer-training-steps', type=int, default=100,
           help='Total number of batches of optimizees generated for training.')
parser.add('--unroll-length', type=int, default=1000,
           help='Total number of training steps considered.')

# Test parameters
parser.add('--test-length', type=int, default=20,
           help='Total length of optimization during testing')
parser.add('--eval-metric', type=str, default='obj',
           help='Evaluation metric in testing: nmse or obj')

parser.add('--debug', action='store_true', help='if debug for evaluation')

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
# Output the type of device used
opts.logger('Using device: {}'.format(opts.device))
# opts.logger('Using tau: {}'.format(opts.tau)) # Output the tau used in current exp


# Set random seed for reproducibility
torch.manual_seed(opts.seed)
random.seed(opts.seed + 7)
np.random.seed(opts.seed + 42)


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


def make_train_step(optimzer, meta_optimizer, is_superwised=False):

    def train_step(optimizees, network_layer, x_gt=None):
        optimzer.train()  # Set the optimizer to training mode

        for _ in range(network_layer):
            optimizees = optimzer(optimizees)
        
        if is_superwised:
            loss = ((optimizees.X_gt - optimizees.X)**2.0).sum(dim=(1, 2)).mean()
        else:
            loss = optimizees.objective(compute_grad=True)

        meta_optimizer.zero_grad()  # Set gradient to zero
        loss.backward()
        meta_optimizer.step()  # Update the weights using the optimizer

        return loss.item()

    return train_step


optimizee_kwargs = {
    'layers': opts.optimizer_training_steps,
    'input_dim': opts.input_dim,
    'output_dim': opts.output_dim,
    'rho': opts.rho,
    's': opts.sparsity,
    'device': opts.device,
    'W_cond_factor': opts.W_cond_factor,
    'x_mag': opts.x_mag,
    'pb': opts.pb,
}

if opts.optimizer == 'AdaLISTA':
    optimizer = optimizers.AdaLISTA(
        layers=opts.layers,
        input_dim=opts.input_dim,
        output_dim=opts.output_dim
    )
elif opts.optimizer == 'LISTA':
    optimizer = optimizers.LISTA(
        input_dim=opts.input_dim,
        output_dim=opts.output_dim,
        T=opts.layers,
        lamb=opts.lamb,
        A=W,
        We_shared=False,
        S_shared=False,
        theta_shared=False
    )
elif opts.optimizer == 'LISTACPSS':
    optimizer = optimizers.LISTACPSS(
        input_dim=opts.input_dim,
        output_dim=opts.output_dim,
        T=opts.layers,
        lamb=opts.lamb,
        A=W,
        percent=opts.p,
        max_percent=opts.max_p,
        We_shared=False,
        theta_shared=False
    )
elif opts.optimizer == 'LISTACPSSWOnly':
    optimizer = optimizers.LISTACPSSWOnly(
        input_dim=opts.input_dim,
        output_dim=opts.output_dim,
        T=opts.optimizer_training_steps,
        percent=opts.p,
        max_percent=opts.max_p,
        # For alignment with our L2O training:
        A=W if opts.fixed_dict else None,
        We_shared=opts.w_shared
    )
else:
    raise ValueError('Invalid optimizer name')

optimizer = optimizer.to(device=opts.device, dtype=opts.dtype)
# fista = optimizers.ProximalGradientDescentMomentum()

if not opts.test:
    training_losses = []  # initialize the array storing training loss function
    validation_losses = []  # initialize the array storing validation loss function

    num_train_batches = opts.train_size // opts.train_batch_size
    
    # Conduct training layer-wise in increasing depth.
    for i in range(opts.train_size):
        loss_layers = []
        for j in range(opts.layers):
            current_layer = j + 1

            epoch = 0
            # batch_losses = []  # Initialize batch losses
            # Loop over stage 1,2,3
            for stage in range(1, 4):
                # Set up optimizer
                meta_optimizer = optimizer.get_meta_optimizer(
                    layer=current_layer,
                    stage=stage,
                    init_lr=opts.init_lr,
                    lr_decay_layer=opts.lr_decay_layer,
                    lr_decay_stage2=opts.lr_decay_stage2,
                    lr_decay_stage3=opts.lr_decay_stage3,
                )
                best_val_nmse = 1e30
                best_val_epoch = epoch  # Starting each stage, the best epoch is the current epoch
                # opts.logger(
                #     'Training layer {} - stage {}'.format(current_layer, stage))
                # print(optimizer)

                train_step = make_train_step(optimizer, meta_optimizer)

                batch_order = np.random.permutation(num_train_batches)

                _seed = opts.seed + 777  # * (i+1)
                optimizees = OPTIMIZEE_DICT[opts.optimizee_type](
                    opts.train_batch_size, W, seed=_seed, **optimizee_kwargs
                )
                optimizees.initialize(_seed)
                optimizer.reset_state(optimizees, opts.step_size)
                loss = train_step(optimizees, network_layer=current_layer)
                # batch_losses.append(loss)  # Add loss to list

                # Compute the average of the batch losses
                training_loss = np.mean(loss)
                # Append this new value to the array of losses
                training_losses.append(training_loss)
                epoch += 1

                # Do validation
                optimizer.eval()
                val_losses = []  # Initialize list of validation losses

                optimizees = OPTIMIZEE_DICT[opts.optimizee_type](
                    opts.val_size, W, seed=opts.seed + 77, **optimizee_kwargs)
                for l in range(current_layer):
                    optimizees = optimizer(optimizees)
                val_loss = optimizees.objective(compute_grad=False).item()

                val_losses.append(val_loss)  # Add current loss to list
                # Compute the average of the batch losses
                validation_loss = np.mean(val_losses)
                # Append this new value to the array of losses
                validation_losses.append(validation_loss)

                # output the epoch results to the terminal
                

                # if validation_loss < best_val_nmse:
                #     best_val_nmse = validation_loss
                #     best_val_epoch = epoch
                # if epoch - best_val_epoch > opts.best_wait or epoch > stage * 200:
                #     break

            

            checkpoint_name = optimizer.name() + '.pt'
            save_path = os.path.join(opts.save_dir, checkpoint_name)
            torch.save(optimizer.state_dict(), save_path)
            # opts.logger('Saved the optimizer to file: ' + save_path)
            # if (i+1) % 100 == 1:
            loss_layers.append(loss)
        print(i)
        # if (i+1) % 100 == 1:
        opts.logger(
                '[%(first)d] Training loss: %(second).5e\t Validation loss: %(third)0.5e' %
                {"first": epoch, "second": training_loss,
                    "third": validation_loss}
            )
        print(f'************{i}************')
        save_obj_traj(loss_layers, i, opts.save_dir)
else:
    checkpoint_name = optimizer.name() + '.pt'
    save_path = os.path.join(opts.save_dir, checkpoint_name)
    optimizer.load_state_dict(torch.load(save_path, map_location='cpu'))
    optimizer.eval()

    if not opts.test_batch_size:
        opts.test_batch_size = opts.test_size
    num_test_batches = opts.test_size // opts.test_batch_size

    # testing_losses_per_layer = [0.0]
    # for current_layer in range(1, model.layers + 1):
    #     # Do testing
    #     test_losses = [] # Initialize list of testing losses

    #     optimizees = OPTIMIZEE_DICT[opts.optimizee_type](
    #         opts.val_size, W, seed=opts.seed + 777, **kwargs)
    #     solved = model(optimizees, K=current_layer)
    #     test_loss = solved.objective(compute_grad=False).item()

    #     test_losses.append(test_loss)  # Add current loss to list
    #     testing_loss = np.mean(test_losses) # Compute the average of the batch losses
    #     testing_losses_per_layer.append(testing_loss) # Append this new value to the array of losses

    if opts.debug:
        signXkXStar = [0.0] * (opts.test_length + 1)
        bXStar = [0.0] * (opts.test_length + 1)
        bXk = [0.0] * (opts.test_length + 1)
        if opts.optimizer == 'LISTA':
            # NOTE Test only
            # WiAi = [0.0] * (opts.test_length + 1)
            # WiAj = [0.0] * (opts.test_length + 1)
            pass
        elif opts.optimizer == 'LISTACPSS':
            WiAi = [0.0] * (opts.test_length + 1)
            WiAj = [0.0] * (opts.test_length + 1)

    test_losses = [0.0] * (opts.test_length + 1)

    if opts.load_mat:
        slice = 32 // opts.test_batch_size

    for i in range(num_test_batches):
        seed = opts.seed + 777 * (i+1)

        if opts.dist_rand:
            optimizee_kwargs['W_cond_factor'] = random.random()
            # 1e-2 ~1e0
            optimizee_kwargs['rho'] = 10 ** (random.random() * (-2))
            # 1e-1 ~1e1
            optimizee_kwargs['x_mag'] = 10 ** (random.random() * (-2) + 1)
            optimizee_kwargs['s'] = int((random.random(
            )*0.15 + 0.1) * optimizee_kwargs['input_dim'])  # input-dim * (0.1 ~ 0.25)
        elif opts.W_cond_rand:
            optimizee_kwargs['W_cond_factor'] = random.random()

        optimizees = OPTIMIZEE_DICT[opts.optimizee_type](
            opts.test_batch_size, W, seed=seed, **optimizee_kwargs
        )

        if opts.load_mat:
            start_index = (i % slice) * opts.test_batch_size
        if opts.load_mat:
            optimizees.load_from_file(
                opts.optimizee_dir + '/' + str(i//slice) + '.mat', start_index, opts.test_batch_size)
            print("Loaded:", opts.optimizee_dir + '/' + str(i//slice) +
                  '.mat', "Start: ", start_index, " Size: ", opts.test_batch_size)

        if opts.load_sol:
            optimizees.load_sol(opts.optimizee_dir + '/sol_' +
                                str(i//slice) + '.mat', start_index, opts.test_batch_size)
            print("Sol Loaded.", i//slice, "Start: ",
                  start_index, " Size: ", opts.test_batch_size)
        else:
            # fista.reset_state(optimizees, None)
            # for _ in range(5000):
            #     optimizees = fista(optimizees)
            # optimizees.X_ref = optimizees.X.detach()
            optimizees.X = optimizees.X_gt
            optimizees.fstar = optimizees.objective_batch()
            optimizees.initialize(seed)

        optimizer.reset_state(optimizees, opts.step_size)
        # if not opts.load_sol:
        #     test_losses[0] += optimizees.objective().detach().cpu().item()
        # else:
        if opts.eval_metric == 'nmse':
            test_losses[0] += optimizees.nmse(
                {'X': optimizees.X, 'X_gt': optimizees.X_gt}).detach().cpu().item()
        elif opts.eval_metric == 'obj':
            test_losses[0] += optimizees.objective_shift().detach().cpu().item()

        if opts.debug:
            signXkXStar[0] += ((torch.sign(optimizees.X) != torch.sign(optimizees.X_gt)).to(
                torch.float32).sum((1, 2)) / opts.input_dim).mean().detach().cpu().item()
            bXStar[0] += ((optimizees.Y != torch.matmul(W, optimizees.X_gt)).to(
                torch.float32).sum((1, 2)) / opts.output_dim).mean().detach().cpu().item()
            bXk[0] += ((optimizees.Y != torch.matmul(W, optimizees.X)).to(
                torch.float32).sum((1, 2)) / opts.output_dim).mean().detach().cpu().item()

            if opts.optimizer == 'LISTA':
                pass
            if opts.optimizer == 'LISTACPSS':
                pass

        for j in range(opts.test_length):
            # Fixed data samples for test
            optimizees = optimizer(optimizees)
            # if not opts.load_sol:
            #     loss = optimizees.objective()
            # else:
            #     loss = optimizees.objective_shift()
            if opts.eval_metric == 'nmse':
                loss = optimizees.nmse(
                    {'X': optimizees.X, 'X_gt': optimizees.X_gt})
            elif opts.eval_metric == 'obj':
                loss = optimizees.objective_shift()

            test_losses[j+1] += loss.detach().cpu().item()

            if opts.debug:
                signXkXStar[j+1] += ((torch.sign(optimizees.X) != torch.sign(optimizees.X_gt)).to(
                    torch.float32).sum((1, 2)) / opts.input_dim).mean().detach().cpu().item()
                bXStar[j+1] += ((optimizees.Y != torch.matmul(W, optimizees.X_gt)).to(
                    torch.float32).sum((1, 2)) / opts.output_dim).mean().detach().cpu().item()
                bXk[j+1] += ((optimizees.Y != torch.matmul(W, optimizees.X)).to(
                    torch.float32).sum((1, 2)) / opts.output_dim).mean().detach().cpu().item()

                if opts.optimizer == 'LISTA':
                    # NOTE Test only
                    # _j = j if j < optimizer.T else -1
                    # We = optimizer.We if optimizer.We_shared else optimizer.We[_j]
                    # WA_all = torch.matmul(We, W)

                    # WiAi_all = torch.diagonal(WA_all, offset=0, dim1=1, dim2=2)
                    # WiAi[j+1] += (((WiAi_all < 0.99).to(torch.float32) + (WiAi_all > 1.01).to(
                    #     torch.float32)).sum((1)) / opts.input_dim).mean().detach().cpu().item()

                    # diagonal_mask = torch.eye(WA_all.shape[1], dtype=torch.bool).unsqueeze(
                    #     0).expand(WA_all.shape[0], -1, -1).to(opts.device)
                    # WiAj_all = WA_all.masked_select(~diagonal_mask)
                    # WiAj[j+1] += ((WiAj_all > 1).to(torch.float32).sum() / (
                    #     opts.input_dim*(opts.input_dim-1)*WA_all.shape[0])).detach().cpu().item()
                    pass

                if opts.optimizer == 'LISTACPSS':
                    _j = j if j < optimizer.T else -1
                    We = optimizer.We if optimizer.We_shared else optimizer.We[_j]
                    WA_all = torch.matmul(We, W)

                    WiAi_all = torch.diagonal(WA_all, offset=0, dim1=1, dim2=2)
                    WiAi[j+1] += (((WiAi_all < 0.99).to(torch.float32) + (WiAi_all > 1.01).to(
                        torch.float32)).sum((1)) / opts.input_dim).mean().detach().cpu().item()

                    diagonal_mask = torch.eye(WA_all.shape[1], dtype=torch.bool).unsqueeze(
                        0).expand(WA_all.shape[0], -1, -1).to(opts.device)
                    WiAj_all = WA_all.masked_select(~diagonal_mask)
                    WiAj[j+1] += ((WiAj_all > 1).to(torch.float32).sum() / (
                        opts.input_dim*(opts.input_dim-1)*WA_all.shape[0])).detach().cpu().item()

        opts.logger('Batch {} completed.'.format(i))

    test_losses = [loss / num_test_batches for loss in test_losses]

    if opts.debug:
        signXkXStar = [_signXkXStar /
                       num_test_batches for _signXkXStar in signXkXStar]
        bXStar = [_bXStar / num_test_batches for _bXStar in bXStar]
        bXk = [_bXk / num_test_batches for _bXk in bXk]

        np.savetxt(os.path.join(opts.save_dir, 'signXk_XStar'),
                   np.array(signXkXStar) * 100, fmt='%.4f')  # save as percentage
        np.savetxt(os.path.join(opts.save_dir, 'bXStar'),
                   np.array(bXStar) * 100, fmt='%.4f')  # save as percentage
        np.savetxt(os.path.join(opts.save_dir, 'bXk'),
                   np.array(bXk) * 100, fmt='%.4f')  # save as percentage
        if opts.optimizer == 'LISTA':
            # NOTE Test only
            # WiAi = [_WiAi / num_test_batches for _WiAi in WiAi]
            # WiAj = [_WiAj / num_test_batches for _WiAj in WiAj]
            # np.savetxt(os.path.join(opts.save_dir, 'WiAi'), np.array(
            #     WiAi) * 100, fmt='%.4f')  # save as percentage
            # np.savetxt(os.path.join(opts.save_dir, 'WiAj'), np.array(
            #     WiAj) * 100, fmt='%.4f')  # save as percentage
            pass
        if opts.optimizer == 'LISTACPSS':
            WiAi = [_WiAi / num_test_batches for _WiAi in WiAi]
            WiAj = [_WiAj / num_test_batches for _WiAj in WiAj]
            np.savetxt(os.path.join(opts.save_dir, 'WiAi'), np.array(
                WiAi) * 100, fmt='%.4f')  # save as percentage
            np.savetxt(os.path.join(opts.save_dir, 'WiAj'), np.array(
                WiAj) * 100, fmt='%.4f')  # save as percentage

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

    # output the epoch results to the terminal
    # opts.logger('Testing losses:')
    # for t_loss in testing_losses_per_layer:
    #     opts.logger('{}'.format(t_loss))
    # loss_save_path = os.path.join(opts.save_dir, 'test_losses.txt')
    # print(f'testing losses saved to {loss_save_path}')
    # np.savetxt(loss_save_path, np.array(testing_losses_per_layer))
