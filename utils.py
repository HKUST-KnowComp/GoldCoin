import os
import numpy as np
import openai
import random
import time
import openai
import torch


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] =str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # print("CUDA is Available")
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic =True