import torch
import torch.nn as nn

def init_non_singular_weights(tensor, times=50):
    """
    Initializes `tensor` (assumed to be 2D) with an invertible matrix using the QR decomposition.
    """
    w = nn.init.kaiming_uniform_(tensor, mode='fan_in', nonlinearity='relu')
    # w = torch.randn_like(tensor)
    x = torch.nn.init.orthogonal_(w) * times
    with torch.no_grad():
        tensor.copy_(x)

    # # Create a random matrix of the same shape as `tensor`
    # x = torch.randn_like(tensor)

    # # Perform the QR decomposition
    # q, r = torch.linalg.qr(x)

    # # To ensure the diagonal of 'r' has only positive entries (avoid sign ambiguity):
    # # Extract the diagonal and its sign
    # diag = torch.diag(r).sign() * times
    # # Multiply 'r' by the sign of its diagonal (broadcasting the sign to match 'r' shape)
    # r *= diag.unsqueeze(0)

    # # The product q @ r is now a random, full-rank matrix
    # with torch.no_grad():
    #     tensor.copy_(q @ r)

# Example usage in a layer


# class OneLinear(nn.Module):
#     '''
#     This model is not from nn.Linear, it initialization with not be over-written by PyTorch.
#     '''

#     def __init__(self, in_features, out_features, bias):
#         super().__init__()
#         self.weight = nn.Parameter(torch.randn(out_features, in_features))

#         if not bias:
#             self.bias = bias
#             self.bias_value = nn.Parameter(torch.zeros(out_features))

#         # Initialize weights to be non-singular
#         init_non_singular_weights(self.weight)

#     def forward(self, x):
#         return x @ self.weight.t() + self.bias_value if self.bias else x @ self.weight.t()


class NonSingularLinear(nn.Module):
    '''
    This model is not from nn.Linear, it initialization with not be over-written by PyTorch.
    '''

    def __init__(self, in_features, out_features, e, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))

        if not bias:
            self.bias = bias
            self.bias_value = nn.Parameter(torch.zeros(out_features))

        # Initialize weights to be non-singular
        init_non_singular_weights(self.weight, e)

    def forward(self, x):
        # return torch.matmul(x, self.weight.t()) + self.bias_value if self.bias else torch.mm(x, self.weight.t())
        return x @ self.weight + self.bias_value if self.bias else x @ self.weight


class SingularLinear(nn.Module):
    '''
    This model is not from nn.Linear, it initialization with not be over-written by PyTorch.
    '''

    def __init__(self, in_features, out_features, bias):
        super().__init__()
        # Initialize weights to be singular
        self.weight = nn.Parameter(torch.zeros(in_features, out_features))

        if not bias:
            self.bias = bias
            self.bias_value = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return x @ self.weight + self.bias_value if self.bias else x @ self.weight
