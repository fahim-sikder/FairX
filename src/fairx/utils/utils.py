import random
import os
import difflib
import numpy as np
import torch


def setSeed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


# def split_data(data, target, sensitive_attr):

#     splitted_data = 

#     return splitted_data