import collections
import itertools
import os
import math
import torch

from fairseq import distributed_utils, options, progress_bar, tasks, utils
from fairseq.data import iterators
from fairseq.trainer import Trainer
from fairseq.meters import AverageMeter, StopwatchMeter

from . import train as ori_train



EXPERIMENTS = {
}

for k, v in EXPERIMENTS.items():
	v['id'] = k



