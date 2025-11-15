import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn

from .lasso import LASSO


class LASSO_LISTA(LASSO):

    def __init__(
        self,
        batch_size: int,
        W=None,
        Y=None,
        rho=0.1,
        s=5,
        device='cpu',
        ood=False,
        ood_s=0.0,
        ood_t=0.0,
        **options
    ) -> None:
        """
        The LASSO optimization problem is formulated as

            minimize (1/2) * ||Y - W @ X||_2^2  + rho * ||X||_1
               X
        """
        super(LASSO_LISTA, self).__init__(batch_size, W, Y,
                                          rho, s, device, ood, ood_s, ood_t, **options)

        seed = options.get('seed', None)
        if seed:
            torch.manual_seed(seed)

        if W is None:
            if seed:
                torch.manual_seed(seed + 1)
            W = torch.normal(0, 1.0 / np.sqrt(self.output_dim), size=(self.batch_size, self.output_dim,
                                                                      self.input_dim)).to(self.device)
            self.W = W / torch.sum(W**2, dim=1, keepdim=True).sqrt()
        else:
            if isinstance(W, np.ndarray):
                W = torch.from_numpy(W).to(self.device)
            elif isinstance(W, torch.Tensor):
                W = W.to(self.device)
            else:
                raise ValueError(f'Invalid type {type(W)} for W')
            assert W.dim() == 3
            # .unsqueeze(0).repeat(self.batch_size, 1, 1)
            self.W = W / torch.sum(W**2, dim=1, keepdim=True).sqrt()

        # Set output Y
        if Y is None:
            if seed:
                torch.manual_seed(seed + 2)
            X_gt = torch.randn(batch_size, self.input_dim,
                               dtype=torch.float32).to(self.device)

            if seed:
                torch.manual_seed(seed + 3)
            # pb = options.get('pb', None)
            # non_zero_idx = torch.bernoulli(
            #     torch.ones_like(X_gt)*pb).to(torch.int64)
            # non_zero_idx = torch.multinomial(
            #     torch.ones_like(X_gt), num_samples=self.s, replacement=False
            # )
            # self.X_gt = torch.zeros_like(X_gt).scatter(
            #     dim=1, index=non_zero_idx, src=X_gt
            # ).unsqueeze(-1)
            self.X_gt = X_gt.unsqueeze(-1)
            self.Y = torch.matmul(self.W, self.X_gt)
        else:
            if isinstance(Y, np.ndarray):
                Y = torch.from_numpy(Y).to(self.device)
            elif isinstance(Y, torch.Tensor):
                Y = Y.to(self.device)
            else:
                raise ValueError(f'Invalid type {type(Y)} for Y')
            assert Y.dim() == 3
            self.Y = Y
            # self.Y = Y.unsqueeze(0).repeat(self.batch_size, 1, 1)
            # self.X_gt = None

    def initialize(self, seed):
        super().initialize(seed)
        self.X = torch.zeros(self.batch_size, self.input_dim, 1).to(
            self.device) + self.x0_shift
        self.set_var('Z', torch.zeros(self.batch_size, self.input_dim, 1).to(
            self.device) + self.x0_shift)

    def nmse(self, inputs: dict = {}):
        if inputs is None:
            inputs = {}
        return 10.0 * torch.log10(
            ((inputs['X'] - inputs['X_gt'])**2).sum(dim=(1, 2)).mean() / (inputs['X_gt']**2).sum(dim=(1, 2)).mean())
