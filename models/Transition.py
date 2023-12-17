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
        self.activation = nn.ELU
        self.gru = nn.GRUCell(hidden_sz, deterministic_sz)
        self.linear1 = nn.Linear(self.action_sz + self.stochastic_sz, self.hidden_sz)
        #self._rnn_input_model = self._build_rnn_input_model()
        self.stochastic_model = nn.Sequential(nn.Linear(self.hidden_sz,self.hidden_sz),
                                              self.activation(),
                                              nn.Linear(self.hidden_sz, self.hidden_sz *2),
                                              self.activation())
        self._dist = distribution

    def initial_state(self, batch_sz, ):
        state =RSSMState(torch.zeros(batch_sz, self._stoch_sz, **kwargs),
            torch.zeros(batch_sz, self._stoch_sz, **kwargs),
            torch.zeros(batch_sz, self._stoch_sz, **kwargs),
            torch.zeros(batch_sz, self._deter_sz, **kwargs)
                            )
        return state



    def forward(self, prev_action ,prev_state):
        prev_input = self.activation(self.linear1(torch.cat([prev_action, prev_state.stoch], dim=-1)))
        # In GRUcell we pass previous  stochastic state and action as input features and
        # previous deterministic state as previous hidden state of the rnn
        deterministic_state = self.gru(prev_input, prev_state.deter)
        mean, std = torch.chunk(self.stochastic_model(deterministic_state), 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self.distribution(mean, std) #Normal distribution
        stochastic_state = dist.rsample()
        #stochastic_state = dist.sample()
        return RSSMState(mean, std, stochastic_state, deterministic_state)


        
        
class Transition_iterator(nn.Module):
    """
    This class is used to run the transition model for subsequents timesteps
        INPUTS:
        time_steps: number of steps to perform
        actions: actions to perform,  size(time_steps, batch_size, action_size)
        starting_state: RSSM state, size(batch_size, state_size)
        OUTPUT:
        prior states: size(time_steps, batch_size, state_size)
    """
    def __init__(self, transition_model):
        self.transition_model = transition_model

    def forward(self,time_steps, actions, starting_state):
        priors = []
        state = starting_state
        for t in range(time_steps):
            state = self.transition_model(actions[t], state)
            priors.append(state)

        return RSSMState(
        torch.stack([state.mean for state in priors], dim=0),
        torch.stack([state.std for state in priors], dim=0),
        torch.stack([state.stoch for state in priors], dim=0),
        torch.stack([state.deter for state in priors], dim=0)
        )




def stack_states(rssm_states: list, dim):
    return RSSMState(
        torch.stack([state.mean for state in rssm_states], dim=dim),
        torch.stack([state.std for state in rssm_states], dim=dim),
        torch.stack([state.stoch for state in rssm_states], dim=dim),
        torch.stack([state.deter for state in rssm_states], dim=dim),
    )

