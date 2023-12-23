import numpy as np
import torch
from rlpyt.algos.base import RlAlgorithm
from rlpyt.replays.sequence.n_step import SamplesFromReplay
from rlpyt.utils.buffer import buffer_to, buffer_method
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.tensor import infer_leading_dims
from tqdm import tqdm

from dreamer.algos.replay import initialize_replay_buffer, samples_to_buffer
from dreamer.models.rnns import get_feat, get_dist
from dreamer.utils.logging import video_summary
from dreamer.utils.module import get_parameters, FreezeParameters

torch.autograd.set_detect_anomaly(True)  # used for debugging gradients