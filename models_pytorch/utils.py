import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn as nn


def get_activation(activation):

    identity = nn.Sequential()

    activations = {
        'softmax' : nn.Softmax(dim=1),
        'sigmoid' : nn.Sigmoid(),
        'identity': identity,
        'none':  identity,
    }

    if callable(activation):
        return activation

    activation = str(activation).lower()
    if activation in activations:
        return activations[activation]
    else:
        raise ValueError('Activation should be callable or {}'.format('/'.join(activations)))


class Flatten(nn.Module):
    def forward(self, input):
        return input.reshape(input.size(0), -1)
