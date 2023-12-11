"""
Code templated from CS 159 HW 3.
"""

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroParam, PyroSample
import torch
import torch.nn as nn

from collections.abc import Sequence


class BNN(PyroModule):
    def __init__(self, dims: Sequence):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layer = PyroModule[nn.Linear](dims[i], dims[i+1])
            layer.weight = PyroSample(
                prior=dist.Gamma(1, 1).expand([dims[i+1], dims[i]]).to_event(2))
            layer.bias = PyroSample(
                prior=dist.Gamma(1, 1).expand([dims[i+1]]).to_event(1))
            layers.append(layer)
        self.layers = PyroModule[torch.nn.ModuleList](layers)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(start_dim=1)
        print(x.shape, "NICE")
        # print(self.layers)
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i != len(self.layers) - 1:
                x = self.relu(x)
        return x


class BNNRegressor(PyroModule):
    def __init__(self, dims: Sequence):
        super().__init__()
        # assert dims[-1] == 1
        self.bnn = BNN(dims)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        mu = self.bnn(x)
        print(mu.shape, "nah no way")
        
        # sigma = pyro.sample("sigma", dist.Uniform(0, 0.5))
        sigma = torch.eye(mu.shape[1])
        print(sigma)
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.MultivariateNormal(mu, sigma), obs=y)
        return mu
