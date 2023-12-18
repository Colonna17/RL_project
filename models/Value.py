#VALUE MODEL IS A DENSE NETWORK

import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

class ValueModel(nn.Module):
    def __init__(self,input_sz = 230, output_sz = (1,), hidden_sz = 200, n_layers = 3, distribution= td.Normal):
        super().__init__()

        self.input_size = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.n_layers = n_layers
        self.dist = distribution
        self.linear1 = nn.Linear(self.input_size, self.hidden_sz)
        self.linear2 = nn.Linear(self.hidden_sz, self.hidden_sz)
        self.linear3 = nn.Linear(self.hidden_sz, int(np.prod(self.output_sz)))
        self.activation = nn.ELU

    
    def forward(self, input):
        dense = self.linear1(input)
        dense = self.activation(dense)
        for i in range(self.n_layers -1):
            dense = self.linear2(dense)
            dense = self.activation(dense)
        output = self.linear3(dense)
        output = torch.reshape(output, input.shape[:-1] + self.output_sz)
        # REMEMBER: For Normal -> arg_constraints = {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}
        output = td.independent.Independent( self.distribution(output, 1), len(self.output_sz))

        return output