#ACTION MODEL IS A DENSE NETWORK

import numpy as np
import torch
import torch.distributions
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as td

from utils.functions import SampleDist
from torch.distributions import TanhTransform ,TransformedDistribution, Independent


#REMEMBER: Action model outputs a tanh-transformed Gaussian.This allowes to reparametrized sampling36-93-23

class ActionModel(nn.Module):
    def __init__(self, action_sz, input_sz, hidden_sz , layers, distribution = td.Normal,
                         min_std=1e-4, init_std=5, mean_scale=5, n_layers =3):
        super().__init__()
        self.action_sz = action_sz
        self.input_sz = (input_sz)
        #self.stochastic_sz = stochastic_sz
        #self.deterministic_sz = deterministic_sz
        self.hidden_sz = hidden_sz 
        self.distribution = td.Normal
        self.activation = nn.ELU
        self.mean_scale = mean_scale
        self.min_std = min_std
        self.init_std = init_std
        self.n_layers = n_layers
        self.raw_init_std = np.log(np.exp(self.init_std) - 1)
        self.linear1 = nn.Linear(self.input_sz, self.hidden_sz)
        self.linear2 = nn.Linear(self.hidden_sz, self.hidden_sz)
        self.linear3 = nn.Linear(self.hidden_sz, self.action_sz *2)



    def forward(self, input):
        out = self.linear1(input)
        out = self.activation(out)
        for i in range(1, self.n_layers):
            out = self.linear2(out)
            out = self.activation(out)
        out = self.linear3(out)
        mean, std = torch.chunk(out, 2, -1)
        mean = self.mean_scale * torch.tanh(mean / self.mean_scale)
        std = F.softplus(std + self.raw_init_std) + self.min_std
        dist = self.distribution(mean, std)

        #Apply TanhTransformed Gaussian to our base distribution as reparametrization trick
        dist = TransformedDistribution(dist, TanhTransform ())
        #Let's make the distributions independent along the first dim, which is the batch size
        dist = Independent(dist, 1)
        dist = SampleDist(dist)

        return dist








