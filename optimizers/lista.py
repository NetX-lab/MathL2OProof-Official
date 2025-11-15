import torch
import numpy as np
import torch.nn as nn

from optimizees.base import BaseOptimizee
import torch.optim as optim
from utils import shrink_free


class LISTA(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, T=16, lamb=0.4, A=None,
                 theta_shared=True, We_shared=True, S_shared=True, **kwargs):
        super(LISTA, self).__init__()
        '''
		LISTA_LeCun: original parametrization of ISTA proposed by Gregor et al.
        Based on framework in https://github.com/mfahes/LPALM/blob/7a0768e5240a8b6c8296287df4bbc75abe4e2b8e/LISTA_LeCun.py
		'''

        A_fixed = A is not None
        self.theta_shared = theta_shared
        self.We_shared = We_shared
        self.S_shared = S_shared

        self.M = output_dim
        self.N = input_dim

        self.T = T
        _step_size = 0.9999 / torch.linalg.norm(
            A, dim=(-2, -1), ord=2, keepdim=True) ** 2 if A_fixed else None
        self.step_size = kwargs.get(
            'step_size', _step_size)  # [batch size, 1, 1]

        if A_fixed:
            _We = A.transpose(1, 2)
        else:
            _We = torch.randn(1, self.N, self.M)

        if self.We_shared:
            self.We = nn.Parameter(
                _We.clone().detach(), requires_grad=True)  # [1, N, M]
        else:
            self.We = nn.ParameterList([nn.Parameter(
                _We.clone().detach(), requires_grad=True) for _ in range(self.T)])  # [T, 1, N, M]

        if A_fixed:
            _S = torch.matmul(A.transpose(1, 2), A)
        else:
            _S = torch.randn(1, self.M, self.N)
            _S = torch.matmul(_S.transpose(1, 2), _S)

        if self.S_shared:
            self.S = nn.Parameter(_S.clone().detach(),
                                  requires_grad=True)  # [1, N, N]
        else:
            self.S = nn.ParameterList([nn.Parameter(
                _S.clone().detach(), requires_grad=True) for _ in range(self.T)])  # [T, 1, N, M]

        # if A_fixed:
        #     _theta = lamb * self.step_size
        # else:
        _theta = lamb

        if self.theta_shared:
            self.theta = nn.Parameter(
                torch.tensor(_theta), requires_grad=True)  # [1]
        else:
            self.theta = nn.ParameterList([nn.Parameter(
                torch.tensor(_theta), requires_grad=True) for _ in range(self.T)])  # [T, 1]

        self.current_step = 0

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model
        """
        return 'LISTA'

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        self.step_size = (step_size if step_size
                          else 0.9999 / optimizees.grad_lipschitz())  # 1 / L, [batch size, 1, 1]

    def detach_state(self):
        pass

    def get_WeS(self, layer):
        params = []
        if self.We_shared:
            params.append(self.We)
        else:
            params.append(self.We[layer])

        if self.S_shared:
            params.append(self.S)
        else:
            params.append(self.S[layer])
        return params

    def get_theta(self, layer):
        params = []
        if self.theta_shared:
            params.append(self.theta)
        else:
            params.append(self.theta[layer])
        return params

    def _get_meta_optimizer_wshared(self, layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3):
        """
        Function: get_optimizer
        Purpose : Return the desired optimizer for the model.
        Based on ./ada_lista.py
        """
        param_groups = []

        # W1, W2 group
        param_groups.append(
            {
                'params': self.get_WeS(layer - 1),
                'lr': init_lr * (lr_decay_layer ** (layer - 1))
            }
        )
        # Current layer
        param_groups.append(
            {
                'params': self.get_theta(layer - 1),
                'lr': init_lr
            }
        )

        # Stage 2 / 3
        if stage > 1:
            # Previous layers
            for i in range(layer - 1):
                param_groups.append(
                    {
                        'params': self.get_theta(i),
                        'lr': init_lr * (lr_decay_layer ** (layer-i-1))
                    }
                )

            # Stage decay
            stage_decay = lr_decay_stage2 if stage == 2 else lr_decay_stage3
            for group in param_groups:
                group['lr'] *= stage_decay

        return optim.Adam(param_groups)

    def _get_meta_optimizer_wnotshared(self, layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3):
        """
        Function: get_optimizer
        Purpose : Return the desired optimizer for the model.
        Based on ./ada_lista.py
        """
        param_groups = []

        # Current layer
        param_groups.append(
            {
                'params': self.get_WeS(layer - 1) + self.get_theta(layer - 1),
                'lr': init_lr
            }
        )

        # Stage 2 / 3
        if stage > 1:
            # Previous layers
            for i in range(layer - 1):
                param_groups.append(
                    {
                        'params': self.get_WeS(i) + self.get_theta(i),
                        'lr': init_lr * (lr_decay_layer ** (layer-i-1))
                    }
                )

            # Stage decay
            stage_decay = lr_decay_stage2 if stage == 2 else lr_decay_stage3
            for group in param_groups:
                group['lr'] *= stage_decay

        return optim.Adam(param_groups)

    def get_meta_optimizer(
        self, layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3
    ):
        """
        Function: get_optimizer
        Purpose : Return the desired optimizer for the model.
        Based on ./ada_lista.py
        """

        if self.We_shared and self.S_shared:
            return self._get_meta_optimizer_wshared(layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3)
        else:
            return self._get_meta_optimizer_wnotshared(layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3)

    def forward(self, optimizees, *args, **kwargs):
        current_step = self.current_step if self.current_step < self.T else -1

        _We = self.step_size * \
            (self.We if self.We_shared else self.We[current_step])
        I = torch.eye(self.N, device=self.S.device)
        _S = I - self.step_size * \
            (self.S if self.S_shared else self.S[current_step])
        _theta = self.step_size * \
            (self.theta if self.theta_shared else self.theta[current_step])

        optimizees.X = shrink_free(torch.matmul(
            _We, optimizees.Y) + torch.matmul(_S, optimizees.X), _theta)

        self.current_step += 1

        return optimizees
