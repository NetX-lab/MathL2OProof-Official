# Modified from lstm2. new models: p,a,b,b1,b2

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from collections import defaultdict
from optimizees.base import BaseOptimizee

from .non_singular import NonSingularLinear, SingularLinear

NORM_FUNC = {
    'exp': torch.exp,
    'eye': nn.Identity(),
    'sigmoid': lambda x: 2.0 * torch.sigmoid(x),
    'softplus': nn.Softplus(),
    'Gaussian': lambda x: 2.0 - 2.0*torch.exp(-x**2),
}


class CoordMathDNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, layers, e,
                 p_use=True, p_scale=1.0, p_scale_learned=True, p_norm='eye',
                 a_use=True, a_scale=1.0, a_scale_learned=True, a_norm='eye',
                 **kwargs):
        """
        Coordinate-wise non-smooth version of our proposed model.
                Please check (18) and (19) in the following paper:
                Liu et al. (2023) "Towards Constituting Mathematical Structures for Learning to Optimize."
        """
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        use_bias = False

        self.hist = defaultdict(list)

        self.layers = layers  # Number of layers for LSTM

        # self.lstm = nn.LSTM(input_size, hidden_size, layers, bias=use_bias)
        self.dnn = nn.Sequential(
            # NonSingularLinear(input_size, input_size, bias=use_bias), nn.ReLU(),
            # NonSingularLinear(input_size, input_size, bias=use_bias), nn.ReLU(),
            # NonSingularLinear(input_size, input_size, bias=use_bias), nn.ReLU(),
            # NonSingularLinear(input_size, input_size, bias=use_bias), nn.ReLU(),
            # NonSingularLinear(input_size, input_size, bias=use_bias), nn.ReLU(),
            # NonSingularLinear(input_size, input_size, bias=use_bias), nn.ReLU(),
            # NonSingularLinear(input_size, input_size, bias=use_bias), nn.ReLU(),
            NonSingularLinear(input_size, input_size, e, bias=use_bias), nn.ReLU())
        # one more hidden laer before the output layer.
        # borrowed from NA-ALISTA: https://github.com/feeds/na-alista
        self.linear = NonSingularLinear(
            input_size, hidden_size, e, bias=use_bias)

        # pre-conditioner
        self.linear_p = SingularLinear(hidden_size, output_size, bias=use_bias)
        # momentum
        # self.linear_a = nn.Linear(hidden_size, output_size, bias=use_bias)

        self.state = None
        self.step_size = kwargs.get('step_size', None)

        self.p_use = p_use
        if p_scale_learned:
            self.p_scale = nn.Parameter(torch.tensor(1.) * p_scale)
        else:
            self.p_scale = p_scale
        self.p_norm = NORM_FUNC[p_norm]

        # self.a_use = a_use
        # if a_scale_learned:
        #     self.a_scale = nn.Parameter(torch.tensor(1.) * a_scale)
        # else:
        #     self.a_scale = a_scale
        # self.a_norm = NORM_FUNC[a_norm]

    @property
    def device(self):
        return self.linear_p.weight.device

    def reset_state(self, optimizees: BaseOptimizee, step_size: float, **kwargs):
        # batch_size = optimizees.X.numel()

        self.step_size = (step_size if step_size
                          else 0.9999 / optimizees.grad_lipschitz())

    def detach_state(self):
        pass
        # if self.state is not None:
        #     self.state = (self.state[0].detach_(), self.state[1].detach_())

    # def clear_hist(self):
    #     for l in self.hist.values():
    #         l.clear()

    def name(self):
        """
        Function: name
        Purpose : Identify the name of the model.
        """
        return 'CoordMathDNN'

    def forward(
        self,
        optimizees: BaseOptimizee,
        grad_method: str,
        reset_state: bool = False,
        detach_grad: bool = True,
    ):
        """docstrings
        TBA
        """
        batch_size = optimizees.batch_size

        # if self.state is None or reset_state:
        #     self.reset_state(optimizees)

        dnn_input = optimizees.get_grad(
            grad_method=grad_method,
            compute_grad=self.training,
            retain_graph=self.training,
        )
        dnn_input2 = optimizees.X

        if detach_grad:
            dnn_input = dnn_input.detach()
            dnn_input2 = dnn_input2.detach()

        dnn_in = torch.cat((dnn_input, dnn_input2), dim=2)

        # Core update by DNN.
        output = self.dnn(dnn_in)
        output = F.relu(self.linear(output))
        P = self.linear_p(output)
        # A = self.linear_a(output)

        P = self.p_norm(P) * self.p_scale if self.p_use else 1.0
        # A = self.a_norm(A) * self.a_scale if self.a_use else 0.0

        # Calculate the update and reshape it back to the shape of the iterate

        # Without momentum
        smooth_grad = optimizees.get_grad(
            grad_method='smooth_grad',
            compute_grad=self.training,
            retain_graph=False
        )
        updateX = - P * self.step_size * smooth_grad
        prox_in = optimizees.X + updateX
        optimizees.X = optimizees.prox(
            {'P': P * self.step_size, 'X': prox_in}, compute_grad=self.training)

        # With momentum
        # smooth_grad2 = optimizees.get_grad(
        #     grad_method='smooth_grad',
        #     inputs={'X': optimizees.get_var('Z')},
        #     compute_grad=self.training,
        #     retain_graph=False
        # )
        # updateZ = - P * self.step_size * smooth_grad2

        # # Apply the update to the iterate
        # # prox_in = B * (optimizees.X + updateX) + (1 - B) * \
        # #     (optimizees.get_var('Z') + updateZ) + B1
        # prox_in = optimizees.get_var('Z') + updateZ
        # prox_out = optimizees.prox(
        #     {'P': P * self.step_size, 'X': prox_in}, compute_grad=self.training)
        # prox_diff = prox_out - optimizees.X
        # optimizees.X = prox_out
        # optimizees.set_var('Z', prox_out + A * prox_diff + B2)
        # Clean up after the current iteration
        # optimizees.Z = prox_out

        return optimizees


# def test():
#     return True


# if __name__ == "__main__":
#     test()
