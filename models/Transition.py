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


def stack_states(rssm_states: list, dim):
    return RSSMState(
        torch.stack([state.mean for state in rssm_states], dim=dim),
        torch.stack([state.std for state in rssm_states], dim=dim),
        torch.stack([state.stoch for state in rssm_states], dim=dim),
        torch.stack([state.deter for state in rssm_states], dim=dim),
    )

class TransitionModel(nn.Module):
    def __init__(self, action_sz, stochastic_sz = 30, deterministic_sz=200, hidden_sz=200, distribution = td.Normal):
        super().__init__()
        self.action_sz = action_sz
        self.stochastic_sz= stochastic_sz
        self.deterministic_sz = deterministic_sz
        self.hidden_sz = hidden_sz
        self.activation = nn.ELU()
        self.gru = nn.GRUCell(hidden_sz, deterministic_sz)
        self.linear1 = nn.Linear(self.action_sz + self.stochastic_sz, self.hidden_sz)
        #self._rnn_input_model = self._build_rnn_input_model()
        self.stochastic_model = nn.Sequential(nn.Linear(self.hidden_sz,self.hidden_sz),
                                              self.activation,
                                              nn.Linear(self.hidden_sz, self.stochastic_sz *2),
                                                    )
        self.distribution = distribution



    def forward(self, prev_action ,prev_state):
        # print('QUA SIAMO IN TRANSITION')
        # print('prev_state:', prev_state)
        prev_input = self.activation(self.linear1(torch.cat([prev_action, prev_state.stoch], dim=-1)))
        # In GRUcell we pass previous  stochastic state and action as input features and
        # previous deterministic state as previous hidden state of the rnn
        #print('COSA SUCCEDE QUA ?')
        #print('prev_input size:',prev_input.size(), '  prev_state.deter size:', prev_state.deter.size())
        deterministic_state = self.gru(prev_input, prev_state.deter)
        #print('deterministic_state.size = ', deterministic_state.size())
        mean, std = torch.chunk(self.stochastic_model(deterministic_state), 2, dim=-1)
        #print('mean size = ', mean.size(),'std size =', std.size())
        std = tf.softplus(std) + 0.1
        dist = self.distribution(mean, std) #Normal distribution
        stochastic_state = dist.rsample()
        #print( 'stochastic_state size = ', stochastic_state.size())
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
        super().__init__()
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

class Policy_iterator(nn.Module):
    def __init__(self, tranistion_model):
        super().__init__()
        self.transition_model = tranistion_model
        
        """
        Roll out the model with a policy function.
        :param steps: number of steps to roll out
        :param policy: RSSMState -> action
        :param prev_state: RSSM state, size(batch_size, state_size)
        :return: next states size(time_steps, batch_size, state_size),
                 actions size(time_steps, batch_size, action_size)
        """
    def forward(self, time_steps,policy, starting_state):
        state = starting_state
        next_states = []
        actions = []
        state = buffer_method(state, "detach")
        for t in range(time_steps):
            action, _ = policy(buffer_method(state, "detach"))
            state = self.transition_model(action, state)
            next_states.append(state)
            actions.append(action)
        next_states = stack_states(next_states, dim=0)
        actions = torch.stack(actions, dim=0)
        return next_states, actions