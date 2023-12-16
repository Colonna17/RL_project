import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as tf
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.buffer import buffer_method

RSSMState = namedarraytuple("RSSMState", ["mean", "std", "stoch", "deter"])
"""
    Returns a new subclass of a namedtuple which exposes indexing / slicing
    reads and writes applied to all contained objects, which must share
    indexing (__getitem__) behavior (e.g. numpy arrays or torch tensors).
    Can be used as a wrapper to the state of the RSSM models
"""

class TransitionModel(nn.Module):
    def __init__(self, action_sz, state_sz, stochastic_sz, deterministic_sz, hidden_sz, distribution = td.Normal):
        super().__init__()
        self.action_sz = action_sz
        self.stochastic_sz= stochastic_sz
        self.deterministic_sz = deterministic_sz
        self.hidden_sz = hidden_sz
        self._activation = nn.ELU
        self.lstm = nn.GRUCell(hidden_sz, deterministic_sz)
        #self._rnn_input_model = self._build_rnn_input_model()
        #self._stochastic_prior_model = self._build_stochastic_model()
        self._dist = distribution

    def initial_state(self, batch_sz, ):
        state =RSSMState(torch.zeros(batch_sz, self._stoch_sz, **kwargs),
            torch.zeros(batch_sz, self._stoch_sz, **kwargs),
            torch.zeros(batch_sz, self._stoch_sz, **kwargs),
            torch.zeros(batch_sz, self._deter_sz, **kwargs)
                            )
        return state



    def forward(self, prev_state, prev_action):
        input = nn.Sequential(nn.Linear(self.action_sz + self.stochastic_sz, self.hidden_sz),
                              self._activation())
        

