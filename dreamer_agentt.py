import numpy as np
import torch
from rlpyt.agents.base import BaseAgent, RecurrentAgentMixin, AgentStep
from rlpyt.utils.buffer import buffer_to, buffer_func
from rlpyt.utils.collections import namedarraytuple

from models.agent import Agent

DreamerAgentInfo = namedarraytuple("DreamerAgentInfo", ["prev_state"])