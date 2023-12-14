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
    def __init__(self, action_size,state_size, hidden_layers):
        super().__init__()
        pass
    
