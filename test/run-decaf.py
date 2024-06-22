import fairx
from fairx.dataset import BaseDataClass
from fairx.models.inprocessing import Decaf

import argparse

import time

import warnings
warnings.filterwarnings('ignore')



def main(args):

    dataset_name = args.dataset_name
    sensitive_attr = args.sensitive_attr
    batch_size = args.batch_size
    n_iter = args.n_iter


    data_module = BaseDataClass(dataset_name, sensitive_attr, attach_target = True)

    decaf = Decaf(data_module, batch_size = batch_size, n_iter = n_iter, generated_sample_size = 10000)
    
    model = decaf.fit()


print(f'Training_done!')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--dataset_name',
        choices=['Adult-Income','Compass', 'Student-performance', 'Predict-diagnosis'],
        default='Adult-Income',
        type=str)

    parser.add_argument(
        '--sensitive_attr',
        type=str)         
    
    parser.add_argument(
        '--batch_size',
        help='batch size for the network',
        default=256,
        type=int)

    parser.add_argument(
        '--n_iter',
        help='epochs',
        default=1000,
        type=int)
    
    args = parser.parse_args() 
    
    main(args)