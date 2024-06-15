import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
import torchvision.utils as vutils
from PIL import Image
from tqdm import tqdm

class CelebaLoader():

    def __init__(self, data_dir):

        super().__init__()

        self.data_dir = data_dir

    def create_celebA(self):
        ##https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
    
        #read  attributes file-> get X: use  'image_id' and get image file!
        # get y: Eyeglasses Male 1 if so and -1 if not
        # get s: Male 1 if so and -1 if not
        # save into numpy arrays.
        data_X=[]
        data_y=[]
        data_s=[]
        attribute_file = pd.read_csv(f'{self.data_dir}/list_attr_celeba.csv')
        transform = transforms.Compose([transforms.Resize(64),
                            transforms.CenterCrop(64),
                            transforms.ToTensor()])
        
        for index, row in tqdm(attribute_file.iterrows()):
            #image = Image.open('data//celeba//images//'+row['image_id'])
            image = Image.open(f'{data_dir}/img_align_celeba/img_align_celeba/'+row['image_id'])
    
            tensor = transform(image)
    
            # image_array = tensor.permute(1,2,0)
    
    #         gray_image = image_array @ torch.tensor([[0.299, 0.587, 0.114], [0.299, 0.587, 0.114], [0.299, 0.587, 0.114]])
            
    #         gray_image = gray_image.unsqueeze(0)
            
            data_X.append(tensor.numpy())
            
            data_y.append(int(row['Male']))
            
            data_s.append(int(row['Eyeglasses']))
            
        return np.array(data_X), np.array(data_y), np.array(data_s)