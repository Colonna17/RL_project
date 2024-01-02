import numpy as np
import torch
from rlpyt.algos.base import RlAlgorithm
from rlpyt.replays.sequence.uniform import UniformSequenceReplayBuffer
from rlpyt.replays.sequence.n_step import SamplesFromReplay
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import infer_leading_dims
from tqdm import tqdm
from random import randint

#from dreamer.models.rnns import get_feat, get_dist
from utils.functions import video_summary, get_parameters, FreezeParameters, get_dist, get_feat

a = torch.load('videos_est3175.pt')
print('a.size()')
print('LO FA IL VIDEO ???')
video_summary("videos/model_error", torch.clamp(a, 0.0, 1.0), 4)
print( 'HA FINITO DI FARE IL VIDEO')