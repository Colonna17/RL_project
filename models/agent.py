import numpy as np
import torch
import torch.nn as nn
from rlpyt.utils.buffer import buffer_func
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import (
    infer_leading_dims,
    restore_leading_dims,
    to_onehot,
    from_onehot,
)

from models.Action import ActionModel
from models.Reward import RewardModel
from models.My_Observation import ObservationDec, ObservationEnc
from models.Value import ValueModel
from models.Transition import TransitionModel, Transition_iterator, Policy_iterator
from models.Representation import RepresentationModel, Representation_iterator
from models.pcont import PcontModel


RSSMState = namedarraytuple("RSSMState", ["mean", "std", "stoch", "deter"])
"""
    Returns a new subclass of a namedtuple which exposes indexing / slicing
    reads and writes applied to all contained objects, which must share
    indexing (__getitem__) behavior (e.g. numpy arrays or torch tensors).
    Can be used as a wrapper to the state of the RSSM models
"""
class AgentModel(nn.Module):
    """
    This class merges together all modules of the latent dynamics model ( transition,
    representation, observation and reward model) and the ones of Dreamer agent, that are 
    action and value models. Dreamer acts on the latent world representation, outputting 
    predictions of state values and actions.
    """
    def __init__(
        self,
        action_shape = (-0.4, 0.4, (17,)),
        image_shape=(3, 64, 64),
        stochastic_sz=30,
        deterministic_sz=200,
        action_hidden_size=200,
        hidden_size=200,
        action_layers=3,
        action_dist="one_hot",
        reward_shape=(1,),
        reward_layers=3,
        reward_hidden=300,
        value_shape=(1,),
        value_layers=3,
        value_hidden=200,
        dtype=torch.float,
        use_pcont=False,
        pcont_layers=3,
        pcont_hidden=200,
        **kwargs,
    ):
        super().__init__()
        self.stochastic_sz = stochastic_sz
        self.deterministic_sz = deterministic_sz
        self.observation_encoder = ObservationEnc()
        encoder_embed_size = self.observation_encoder.embed_size
        decoder_embed_size = stochastic_sz + deterministic_sz
        self.observation_decoder = ObservationDec()
        self.action_shape = action_shape
        self.action_sz = np.prod(action_shape)
        self.transition = TransitionModel(self.action_sz, stochastic_sz, deterministic_sz, hidden_size)
        self.representation = RepresentationModel(self.transition, self.action_sz,
                                                  encoder_embed_size, stochastic_sz,
                                                     deterministic_sz,hidden_size)
        self.rollout = Representation_iterator(self.transition ,self.representation)
        self.rollout_policy = Policy_iterator(self.transition)
        self.rollout_transition = Transition_iterator(self.transition)
        feature_size = stochastic_sz + deterministic_sz
        self.action_dist = action_dist
        self.action_decoder = ActionModel(self.action_sz, feature_size, 
                                          action_hidden_size, action_layers,
                                            action_dist)
        self.reward_model = RewardModel()
        self.value_model = ValueModel()
        #self.rollout = RSSMRollout(self.representation, self.transition)
        self.dtype = dtype
        if use_pcont:
            self.pcont = PcontModel()

    def forward( self, observation, previous_action= None, previous_state = None):
        state = self.state_representation(observation, previous_action, previous_state)
        action, action_dist = self.policy(state)
        value = self.value_model(torch.cat((state.stoch, state.deter), dim=-1))
        reward = self.reward_model(torch.cat((state.stoch, state.deter), dim=-1))
        return action, action_dist, value, reward, state

    def policy(self, state):
        input = torch.cat((state.stoch, state.deter), dim=-1)
        # print('LE DIM INIZIALI SONO: state.stoch = ', state.stoch.size(), 'state.deter =', state.deter.size())
        # print(' E QUA COSA ESCE?', input, input.size())
        action_dist = self.action_decoder(input)
        if self.action_dist == "tanh_normal":
            if self.training:  # use agent.train(bool) or agent.eval()
                action = action_dist.rsample()
            else:
                action = action_dist.mode()
        elif self.action_dist == "one_hot":
            action = action_dist.sample()
            # This doesn't change the value, but gives us straight-through gradients
            # probs -> @lazy_property def probs(self): return logits_to_probs(self.logits, is_binary=True)
            action = action + action_dist.probs - action_dist.probs.detach()
            
        elif self.action_dist == "relaxed_one_hot":
            action = action_dist.rsample()
        else:
            action = action_dist.sample()
        return action, action_dist
    
    def get_state_representation(self,observation, prev_action, prev_state):
        # print('E INVECE QUA COSA SUCCEDE ? (1)')
        # print('prev_state size:', prev_state, prev_state.deter.size())
        obs_embed = self.observation_encoder(observation)
        if prev_action is None:
            prev_action = torch.zeros(
                observation.size(0),
                self.action_shape,
                device=observation.device,
                dtype=observation.dtype,
            )
        if prev_state is None:
            prev_state = self.representation.initial_state(
                prev_action.size(0), device=prev_action.device, dtype=prev_action.dtype
            )
            # print('E INVECE QUA COSA SUCCEDE ? (2)')
            # print('prev_state size:', prev_state.size())
        
        _, state = self.representation(obs_embed, prev_action, prev_state)
        # print('CHE DIMENSIONI HA LO STATE CHE ESCE DA REPRESENTATION?')
        # print( 'RISPOSTA', 'stoch:',state.stoch.size(), 'deter:', state.deter.size())
        return state
    
    def state_representation(
        self, observation, prev_action= None, prev_state= None,):
        """
        In case the state or the action are the first ones of a series,
        they need to be initialized
        """
        obs_embed = self.observation_encoder(observation)
        if prev_action is None:
            prev_action = torch.zeros(
                observation.size(0),
                self.action_sz,
                device=observation.device,
                dtype=observation.dtype,
            )
        if prev_state is None:
            sz = prev_action.size(0)
            prev_state = RSSMState(
            torch.zeros(sz, self.stochastic_sz, device=prev_action.device, dtype=prev_action.dtype),
            torch.zeros(sz, self.stochastic_sz, device=prev_action.device, dtype=prev_action.dtype),
            torch.zeros(sz, self.stochastic_sz, device=prev_action.device, dtype=prev_action.dtype),
            torch.zeros(sz, self.deterministic_sz, device=prev_action.device, dtype=prev_action.dtype),
                                  ) 
            # prev_state = self.representation.initial_state(
            #     sz, device=prev_action.device, dtype=prev_action.dtype
            # )
        _, state = self.representation(obs_embed, prev_action, prev_state)
        return state

    def state_transition(self, prev_action, prev_state):
        state = self.transition(prev_action, prev_state)
        return state


class AtariDreamerModel(AgentModel):

    def forward(
        self,
        observation: torch.Tensor,
        prev_action: torch.Tensor = None,
        prev_state: RSSMState = None,
    ):
        lead_dim, T, B, img_shape = infer_leading_dims(observation, 3)
        observation = (
            observation.reshape(T * B, *img_shape).type(self.dtype) / 255.0 - 0.5
        )
        prev_action = prev_action.reshape(T * B, -1).to(self.dtype)
        if prev_state is None:
            prev_state = self.representation.initial_state(
                prev_action.size(0), device=prev_action.device, dtype=self.dtype
            )
        state = self.get_state_representation(observation, prev_action, prev_state)

        action, action_dist = self.policy(state)
        return_spec = ModelReturnSpec(action, state)
        return_spec = buffer_func(return_spec, restore_leading_dims, lead_dim, T, B)
        return return_spec


ModelReturnSpec = namedarraytuple("ModelReturnSpec", ["action", "state"])

