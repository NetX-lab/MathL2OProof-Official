import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from optimizees.base import BaseOptimizee
from utils import shrink_ss


class LISTACPSS(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, T=16, lamb=0.4,
                 percent=0.012, max_percent=0.13, A=None, theta_shared=True, We_shared=True, **kwargs):
        super(LISTACPSS, self).__init__()
        '''
		LISTA_LeCun: original parametrization of ISTA proposed by Gregor et al.
        Based on https://github.com/VITA-Group/LISTA-CPSS/blob/master/models/LISTA_cpss.py
		'''

        A_fixed = A is not None
        self.A_fixed = A_fixed
        self.theta_shared = theta_shared
        self.We_shared = We_shared

        self.M = output_dim
        self.N = input_dim

        self.T = T
        _step_size = 0.9999 / torch.linalg.norm(
            A, dim=(-2, -1), ord=2, keepdim=True) ** 2 if A_fixed else 1
        self.step_size = kwargs.get(
            'step_size', _step_size)  # [batch size, 1, 1]

        if A_fixed:
            _We = (A.transpose(1, 2) * self.step_size).mean(dim=0, keepdim=True)
            # _theta = (lamb * self.step_size).mean(dim=0,
            #                                       keepdim=True).squeeze(-1).squeeze(-1)
        else:
            # _We = torch.normal(0, 1.0 / (self.M*self.N) **
            #                    10, size=(1, self.N, self.M))
            _We = torch.zeros(1, self.N, self.M)
            # _theta = torch.tensor(lamb)

        _theta = torch.tensor(lamb)
        if self.We_shared:
            self.We = nn.Parameter(
                _We.clone().detach(), requires_grad=True)  # [1, N, M]
        else:
            self.We = nn.ParameterList([nn.Parameter(
                _We.clone().detach(), requires_grad=True) for _ in range(self.T)])  # [T, 1, N, M]

        if self.theta_shared:
            # _theta = torch.tensor(lamb)
            self.theta = nn.Parameter(
                _theta.clone().detach(), requires_grad=True)  # [1]
        else:
            # Must this scaling to avoid NaN in not shared and single M mode:
            # _theta = (lamb * self.step_size).mean(dim=0,
            #                                       keepdim=True).squeeze(-1).squeeze(-1)
            self.theta = nn.ParameterList([nn.Parameter(
                _theta.clone().detach(), requires_grad=True) for _ in range(self.T)])  # [T, 1]

        self.p = percent
        self.max_p = max_percent
        self.ps = [(t+1) * self.p for t in range(self.T)]
        self.ps = np.clip(self.ps, 0.0, self.max_p)
        self.current_step = 0

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model
        """
        return 'LISTACPSS'

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        # if not self.A_fixed:
        #     self.step_size = (step_size if step_size
        #                       else 0.9999 / optimizees.grad_lipschitz())  # 1 / L, [batch size, 1, 1]
        # else:
        #     self.step_size = 1
        self.current_step = 0

    def detach_state(self):
        pass

    def get_We(self, layer):
        params = []
        if self.We_shared:
            params.append(self.We)
        else:
            params.append(self.We[layer])

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
                'params': self.get_We(layer - 1),
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
                'params': self.get_We(layer - 1) + self.get_theta(layer - 1),
                'lr': init_lr
            }
        )

        # Stage 2 / 3
        if stage > 1:
            # Previous layers
            for i in range(layer - 1):
                param_groups.append(
                    {
                        'params': self.get_We(i) + self.get_theta(i),
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

        if self.We_shared:
            return self._get_meta_optimizer_wshared(layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3)
        else:
            return self._get_meta_optimizer_wnotshared(layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3)

    def forward(self, optimizees, *args, **kwargs):
        current_step = self.current_step if self.current_step < self.T else -1
        _We = self.We if self.We_shared else self.We[current_step]
        _percent = self.ps[current_step]
        # self.step_size *
        _theta = self.theta if self.theta_shared else self.theta[current_step]

        _res = optimizees.Y - torch.matmul(optimizees.W, optimizees.X)

        optimizees.X = shrink_ss(
            optimizees.X + torch.matmul(_We, _res), _theta, _percent)

        self.current_step += 1

        return optimizees



class LISTACPSSSTEP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, T=16, lamb=0.4,
                 percent=0.012, max_percent=0.13, A=None, theta_shared=True, We_shared=True, **kwargs):
        super(LISTACPSSSTEP, self).__init__()
        '''
		LISTA_LeCun: original parametrization of ISTA proposed by Gregor et al.
        Based on https://github.com/VITA-Group/LISTA-CPSS/blob/master/models/LISTA_cpss.py
		'''

        A_fixed = A is not None
        self.A_fixed = A_fixed
        self.theta_shared = theta_shared
        self.We_shared = We_shared

        self.M = output_dim
        self.N = input_dim

        self.T = T
        _step_size = 0.9999 / torch.linalg.norm(
            A, dim=(-2, -1), ord=2, keepdim=True) ** 2 if A_fixed else 1
        self.step_size = kwargs.get(
            'step_size', _step_size)  # [batch size, 1, 1]

        if A_fixed:
            _We = (A.transpose(1, 2) * self.step_size).mean(dim=0, keepdim=True)
            # _theta = (lamb * self.step_size).mean(dim=0,
            #                                   keepdim=True).squeeze(-1).squeeze(-1)
        else:
            # If w is shared
            _We = torch.normal(0, 1.0 / (self.M*self.N) **
                               2, size=(1, self.N, self.M))
            # If w is not shared
            # _We = torch.normal(0, 1.0 / self.M, size=(1, self.N, self.M))
            # _We = A.transpose(1, 2)
            # _We = torch.randn(1, self.N, self.M)
            # _We = torch.zeros(1, self.N, self.M)
            # _theta = torch.tensor(lamb)

        _theta = torch.tensor(lamb)
        if self.We_shared:
            self.We = nn.Parameter(
                _We.clone().detach(), requires_grad=True)  # [1, N, M]
        else:
            self.We = nn.ParameterList([nn.Parameter(
                _We.clone().detach(), requires_grad=True) for _ in range(self.T)])  # [T, 1, N, M]

        if self.theta_shared:
            # _theta = torch.tensor(lamb)
            self.theta = nn.Parameter(
                _theta.clone().detach(), requires_grad=True)  # [1]
        else:
            # Must this scaling to avoid NaN in not shared and single M mode:
            # _theta = (lamb * self.step_size).mean(dim=0,
            #                                       keepdim=True).squeeze(-1).squeeze(-1)
            self.theta = nn.ParameterList([nn.Parameter(
                _theta.clone().detach(), requires_grad=True) for _ in range(self.T)])  # [T, 1]

        self.p = percent
        self.max_p = max_percent
        self.ps = [(t+1) * self.p for t in range(self.T)]
        self.ps = np.clip(self.ps, 0.0, self.max_p)
        self.current_step = 0

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model
        """
        return 'LISTACPSSSTEP'

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        # if not self.A_fixed:
        self.step_size = (step_size if step_size
                          else 0.9999 / optimizees.grad_lipschitz())  # 1 / L, [batch size, 1, 1]
        # else:
        #     self.step_size = 1
        self.current_step = 0

    def detach_state(self):
        pass

    def get_We(self, layer):
        params = []
        if self.We_shared:
            params.append(self.We)
        else:
            params.append(self.We[layer])

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
                'params': self.get_We(layer - 1),
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
                'params': self.get_We(layer - 1) + self.get_theta(layer - 1),
                'lr': init_lr
            }
        )

        # Stage 2 / 3
        if stage > 1:
            # Previous layers
            for i in range(layer - 1):
                param_groups.append(
                    {
                        'params': self.get_We(i) + self.get_theta(i),
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

        if self.We_shared:
            return self._get_meta_optimizer_wshared(layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3)
        else:
            return self._get_meta_optimizer_wnotshared(layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3)

    def forward(self, optimizees, *args, **kwargs):
        current_step = self.current_step if self.current_step < self.T else -1
        _We = self.step_size * \
            (self.We if self.We_shared else self.We[current_step])
        _percent = self.ps[current_step]
        # self.step_size *
        _theta = self.step_size * \
            (self.theta if self.theta_shared else self.theta[current_step])

        _res = optimizees.Y - torch.matmul(optimizees.W, optimizees.X)

        optimizees.X = shrink_ss(
            optimizees.X + torch.matmul(_We, _res), _theta, _percent)

        self.current_step += 1

        return optimizees
    
    
class LISTACPSSWOnly(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, T=16, 
                 percent=0.012, max_percent=0.13, A=None, We_shared=True, **kwargs):
        '''
		LISTA_LeCun: original parametrization of ISTA proposed by Gregor et al.
        Based on https://github.com/VITA-Group/LISTA-CPSS/blob/master/models/LISTA_cpss.py
		'''
        super(LISTACPSSWOnly, self).__init__()
        
        A_fixed = A is not None
        self.A_fixed = A_fixed
        self.We_shared = We_shared

        self.M = output_dim
        self.N = input_dim

        self.T = T
        _step_size = 0.9999 / torch.linalg.norm(
            A, dim=(-2, -1), ord=2, keepdim=True) ** 2 if A_fixed else 1
        self.step_size = kwargs.get(
            'step_size', _step_size)  # [batch size, 1, 1]

        if A_fixed:
            _We = (A.transpose(1, 2) * self.step_size).mean(dim=0, keepdim=True)
            # _theta = (lamb * self.step_size).mean(dim=0,
            #                                       keepdim=True).squeeze(-1).squeeze(-1)
        else:
            _We = torch.normal(0, 1.0 / (self.M*self.N)**2, size=(1, self.N, self.M))
            # _theta = torch.tensor(lamb)

        if self.We_shared:
            self.We = nn.Parameter(
                _We.clone().detach(), requires_grad=True)  # [1, N, M]
        else:
            self.We = nn.ParameterList([nn.Parameter(
                _We.clone().detach(), requires_grad=True) for _ in range(self.T)])  # [T, 1, N, M]


        self.p = percent
        self.max_p = max_percent
        self.ps = [(t+1) * self.p for t in range(self.T)]
        self.ps = np.clip(self.ps, 0.0, self.max_p)
        self.current_step = 0

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model
        """
        return 'LISTACPSSWOnly'

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        # if not self.A_fixed:
        #     self.step_size = (step_size if step_size
        #                       else 0.9999 / optimizees.grad_lipschitz())  # 1 / L, [batch size, 1, 1]
        # else:
        #     self.step_size = 1
        self.current_step = 0

    def detach_state(self):
        pass

    def get_We(self, layer):
        params = []
        if self.We_shared:
            params.append(self.We)
        else:
            params.append(self.We[layer])

        return params


    def _get_meta_optimizer_wshared(self, layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3):
        """
        Function: get_optimizer
        Purpose : Return the desired optimizer for the model.
        Based on ./ada_lista.py
        """
        param_groups = []

        # W1, W2 group
        # Only one layer, set decayed learning rate.
        param_groups.append(
            {
                'params': self.get_We(layer - 1),
                'lr': init_lr * (lr_decay_layer ** (layer - 1))
            }
        )

        if stage > 1:
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
                'params': self.get_We(layer - 1),
                'lr': init_lr
            }
        )

        # Stage 2 / 3
        if stage > 1:
            # Previous layers
            for i in range(layer - 1):
                param_groups.append(
                    {
                        'params': self.get_We(i),
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

        if self.We_shared:
            return self._get_meta_optimizer_wshared(layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3)
        else:
            return self._get_meta_optimizer_wnotshared(layer, stage, init_lr, lr_decay_layer, lr_decay_stage2, lr_decay_stage3)

    def forward(self, optimizees, *args, **kwargs):
        current_step = self.current_step if self.current_step < self.T else -1
        _We = self.We if self.We_shared else self.We[current_step]
        _percent = self.ps[current_step]
        # self.step_size *

        _res = optimizees.Y - torch.matmul(optimizees.W, optimizees.X)

        optimizees.X = shrink_ss(
            optimizees.X + torch.matmul(_We, _res), 0, _percent)

        self.current_step += 1

        return optimizees