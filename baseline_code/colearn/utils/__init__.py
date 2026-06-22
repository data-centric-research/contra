import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.deterministic = True


def get_test_acc(acc):
    return (acc[0] + acc[1]) / 2.0 if isinstance(acc, tuple) else acc


__all__ = ("set_seed", "get_test_acc")
