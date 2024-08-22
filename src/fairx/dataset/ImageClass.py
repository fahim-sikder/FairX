import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

from fairx.utils import setSeed

setSeed(2022)

class CelebaLoader():

    """
    Dataset loader for CelebA dataset [1].

    [1] Liu, Ziwei, et al. "Deep learning face attributes in the wild." Proceedings of the IEEE international conference on computer vision. 2015.
    """

    def __init__(self, data_dir, target = 'Male', sensitive_attr = 'Eyeglasses'):

        """
        Input: data_dir, string, path to the dataset directory. Download the dataset from here: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html, and check the following link for details: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset.

            target: string

            sensitive_attr: string, protected attribute

        Here, we use `Eyeglasses` as sensitive attribute and `Gender` as target.

        Return, Numpy arrays of data_x (features), data_y (target), data_s (sensitive_attribute)

        """

        super().__init__()

        self.data_dir = data_dir

        self.target = target

        self.sensitive_attr = sensitive_attr

    def create_celebA(self):

        data_X=[]
        
        data_y=[]
        
        data_s=[]
        
        attribute_file = pd.read_csv(f'{self.data_dir}/list_attr_celeba.csv')
        
        transform = transforms.Compose([transforms.Resize(64),
                            transforms.CenterCrop(64),
                            transforms.ToTensor()])
        
        for index, row in tqdm(attribute_file.iterrows()):
            
            image = Image.open(f'{self.data_dir}/img_align_celeba/img_align_celeba/'+row['image_id'])
    
            tensor = transform(image)
            
            data_X.append(tensor.numpy())
            
            data_y.append(int(row[self.target]))
            
            data_s.append(int(row[self.sensitive_attr]))
            
        return np.array(data_X), np.array(data_y), np.array(data_s)


class VectorDataset(Dataset):
    """
    Helper function, adapted from https://github.com/SoftWiser-group/FairDisCo/blob/main/utils.py
    """
    def __init__(self, X, S, Y):
        
        self.X = X
        self.S = S
        self.Y = Y

    def __getitem__(self, i):
        
        x, s, y = self.X[i], self.S[i], self.Y[i]
        
        return x, s, y
    
    def __len__(self):
        
        return self.X.shape[0]

class ColorMNIST():

    """
    Helper class to load color mnist,

    adapted from https://github.com/SoftWiser-group/FairDisCo/blob/main/utils.py
    """

    def __init__(self):

        super().__init__()

    def load_colormnist():
    
        transform = transforms.Compose([transforms.ToPILImage(),transforms.Scale(64), transforms.CenterCrop(64),transforms.ToTensor()])
        
        train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
        test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
        # train
        n = len(train_data)
        X_train = torch.zeros(n, 3, 64, 64)
        S_train = torch.arange(n) % 3
        Y_train = train_data.targets
        for i in range(n):
            X_train[i,S_train[i]] = transform(train_data.data[i]).squeeze()
        X_train /= X_train.max()
        
        # test
        n = len(test_data)
        X_test = torch.zeros(n, 3, 64, 64)
        S_test = torch.arange(n) % 3
        Y_test = test_data.targets
        for i in range(n):
            X_test[i,S_test[i]] = transform(test_data.data[i]).squeeze()
        X_test /= X_test.max()
        
        train_data = VectorDataset(X_train, S_train, Y_train)
        test_data = VectorDataset(X_test, S_test, Y_test)
    
        return train_data, test_data