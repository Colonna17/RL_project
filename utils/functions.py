import torch
import torch.nn.functional as F
from torch.nn import Module
import torch.distributions
import numpy as np
from rlpyt.utils.logging import logger
from torch.utils.tensorboard.writer import SummaryWriter
from typing import Iterable
from rlpyt.utils.collections import namedarraytuple
import torch.distributions as td



class SampleDist:
    """
    Class taken from https://github.com/juliusfrost/dreamer-pytorch/blob/main/dreamer/models/distribution.py
    It is used to apply usefull actions and computations on torch distributions.
    """
    def __init__(self, dist: torch.distributions.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return "SampleDist"

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = (
            torch.argmax(logprob, dim=0)
            .reshape(1, batch_size, 1)
            .expand(1, batch_size, feature_size)
        )
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()
    



def video_summary(tag, video, step=None, fps=20):
    writer: SummaryWriter = logger.get_tf_summary_writer()
    writer.add_video(tag=tag, vid_tensor=video, global_step=step, fps=fps)




def get_parameters(modules: Iterable[Module]):
    """
    Given a list of torch modules, returns a list of their parameters.
    :param modules: iterable of modules
    :returns: a list of parameters
    """
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters


class FreezeParameters:
    def __init__(self, modules: Iterable[Module]):
        """
        Context manager to locally freeze gradients.
        In some cases with can speed up computation because gradients aren't calculated for these listed modules.
        example:
        ```
        with FreezeParameters([module]):
            output_tensor = module(input_tensor)
        ```
        :param modules: iterable of modules. used to call .parameters() to freeze gradients.
        """
        self.modules = modules
        self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

    def __enter__(self):
        for param in get_parameters(self.modules):
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(get_parameters(self.modules)):
            param.requires_grad = self.param_states[i]

RSSMState = namedarraytuple("RSSMState", ["mean", "std", "stoch", "deter"])

def get_feat(rssm_state: RSSMState):
    return torch.cat((rssm_state.stoch, rssm_state.deter), dim=-1)


def get_dist(rssm_state: RSSMState):
    return td.independent.Independent(td.Normal(rssm_state.mean, rssm_state.std), 1)

