import torch
import torch.distributions as td
import torch.nn as nn
import torch.nn.functional as tf
from rlpyt.utils.collections import namedarraytuple

RSSMState = namedarraytuple("RSSMState", ["mean", "std", "stoch", "deter"])
"""
    Returns a new subclass of a namedtuple which exposes indexing / slicing
    reads and writes applied to all contained objects, which must share
    indexing (__getitem__) behavior (e.g. numpy arrays or torch tensors).
    Can be used as a wrapper to the state of the RSSM models
"""
class RepresentationModel(nn.Module):
    def __init__(
        self,
        transition_model,
        action_sz,
        obs_sz,
        stochastic_sz=30,
        deterministic_sz=200,
        hidden_sz=200,
        distribution = td.Normal
        ):
        super().__init__()

        self.observation_sz = obs_sz
        self.action_sz = action_sz
        self.stochastic_sz= stochastic_sz
        self.deterministic_sz = deterministic_sz
        self.hidden_sz = hidden_sz
        self.activation = nn.ELU
        self.gru = nn.GRUCell(hidden_sz, deterministic_sz)
        self.linear1 = nn.Linear(self.action_sz + self.stochastic_sz, self.hidden_sz)
        self.stochastic_model = nn.Sequential(nn.Linear(self.deterministic_sz + self.observation_sz, self._hidden_size),
                                              self.activation(),
                                              nn.Linear(self.hidden_sz, self.hidden_sz *2)
                                            )
        self.dist = distribution
        self.transition = transition_model



    def initial_state(self, batch_sz, **kwargs):
        state =RSSMState(torch.zeros(batch_sz, self._stoch_sz, **kwargs),
            torch.zeros(batch_sz, self._stoch_sz, **kwargs),
            torch.zeros(batch_sz, self._stoch_sz, **kwargs),
            torch.zeros(batch_sz, self._deter_sz, **kwargs)
                            )
        return state
    

#    def forward(self, prev_action ,prev_state, observation):
    def forward(self,observation, prev_action ,prev_state):
        #The prior is obtained from Transition model 
        prior_state = self.transition(prev_action, prev_state)
        stochastic_input = torch.cat([prior_state.deter, observation], -1)
        mean, std = torch.chunk(self.stochastic_model(stochastic_input), 2, dim=-1)
        std = tf.softplus(std) + 0.1
        dist = self.dist(mean, std)
        stoch_state = dist.rsample()
        #stoch_state = dist.sample()
        posterior_state = RSSMState(mean, std, stoch_state, prior_state.deter)
        return prior_state, posterior_state







class Representation_iterator(nn.Module):
    """
    This class is used to run the representation model for subsequents timesteps
        INPUTS:
        time_steps: number of steps to perform
        actions: actions to perform,  size(time_steps, batch_size, action_size)
        starting_state: RSSM state, size(batch_size, state_size)
        observations: embeddings of observations, size(time_steps, batch_size, embedding_size)
        OUTPUT:
        prior states: size(time_steps, batch_size, state_size)
    """

    def __init__(self, transition_model, representation_model, observations):
        super().__init__()
        self.transition_model = transition_model
        self.representation_model = representation_model

    def forward(self,time_steps, actions, starting_state, observations):
        priors =[]
        posteriors = []
        previous_state = starting_state
        for t in range(time_steps):
            prior_state, posterior_state = self.representation_model(
                observations[t], actions[t], previous_state)
            priors.append(prior_state)
            posteriors.append(posterior_state)
            previous_state = posterior_state
        prior = RSSMState(
            torch.stack([state.mean for state in priors], dim=0),
            torch.stack([state.std for state in priors], dim=0),
            torch.stack([state.stoch for state in priors], dim=0),
            torch.stack([state.deter for state in priors], dim=0)
            )
        posterior = RSSMState(
            torch.stack([state.mean for state in posteriors], dim=0),
            torch.stack([state.std for state in posteriors], dim=0),
            torch.stack([state.stoch for state in posteriors], dim=0),
            torch.stack([state.deter for state in posteriors], dim=0)
            )
        return prior, posterior
                       


 