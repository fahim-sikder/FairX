import random
import os
import difflib
import numpy as np
import torch


def setSeed(seed=2022):

    """
    Utility function for adding seed.
    """
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def add_dict_result(dict1, dict2):

    """
    Utility function to update result dictionary while benchmarking.
    """
    
    for keys in dict1.keys():

        if not dict2[keys]:
    
            dict1[keys].append('-')
            
        else:
            dict1[keys].append(dict2[keys])

    return dict1