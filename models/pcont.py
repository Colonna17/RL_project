# REWARD MODEL IS A DENSE NETWORK
import numpy as np
import torch
import torch.distributions as td
import torch.nn as nn

class PcontModel(nn.Module):
    def __init__(self,input_sz = 230, hidden_sz = 200, output_sz = (1,), n_layers = 3, distribution= "binary"):
        super().__init__()

        self.input_size = input_sz
        self.hidden_sz = hidden_sz
        self.output_sz = output_sz
        self.n_layers = n_layers
        self.dist = distribution
        self.linear1 = nn.Linear(self.input_size, self.hidden_sz)
        self.linear2 = nn.Linear(self.hidden_sz, self.hidden_sz)
        self.linear3 = nn.Linear(self.hidden_sz, int(np.prod(self.output_sz)))
        self.activation = nn.ELU()

    
    def forward(self, input): # Input is the concatenation of stoch and deterministic states
        dense = self.linear1(input)
        dense = self.activation(dense)
        for i in range(self.n_layers -1):
            dense = self.linear2(dense)
            dense = self.activation(dense)
        dense = self.linear3(dense)
        output = torch.reshape(dense, input.shape[:-1] + self.output_sz)
        # REMEMBER: For Normal -> arg_constraints = {'loc': Real(), 'scale': GreaterThan(lower_bound=0.0)}
        output = td.independent.Independent( td.Bernoulli(logits=output), len(self.output_sz))#,validate_args=False)
        return output












###########################################################################################################
    

# class DenseModel(nn.Module):
#     def __init__(
#         self,
#         feature_size: int,
#         output_shape: tuple,
#         layers: int,
#         hidden_size: int,
#         dist="normal",
#         activation=nn.ELU,
#     ):
#         super().__init__()
#         self._output_shape = output_shape
#         self._layers = layers
#         self._hidden_size = hidden_size
#         self._dist = dist
#         self.activation = activation
#         # For adjusting pytorch to tensorflow
#         self._feature_size = feature_size
#         # Defining the structure of the NN
#         self.model = self.build_model()

#     def build_model(self):
#         model = [nn.Linear(self._feature_size, self._hidden_size)]
#         model += [self.activation()]
#         for i in range(self._layers - 1):
#             model += [nn.Linear(self._hidden_size, self._hidden_size)]
#             model += [self.activation()]
#         model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
#         return nn.Sequential(*model)

#     def forward(self, features):
#         dist_inputs = self.model(features)
#         reshaped_inputs = torch.reshape(
#             dist_inputs, features.shape[:-1] + self._output_shape
#         )
#         if self._dist == "normal":
#             return td.independent.Independent(
#                 td.Normal(reshaped_inputs, 1), len(self._output_shape)
#             )
#         if self._dist == "binary":
#             return td.independent.Independent(
#                 td.Bernoulli(logits=reshaped_inputs), len(self._output_shape)
#             )
#         raise NotImplementedError(self._dist)