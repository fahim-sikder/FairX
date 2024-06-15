from fairx.dataset import BaseDataClass
from fairx.models.baseclass import BaseModelClass

import torch

from .modules import *
from .utils import *



class FairDisco(BaseModelClass):

    def __init__(self, data_module, beta = 7, epochs = 1000, batch_size = 2048):

        super(BaseModelClass, self).__init__()

        self.data_module = data_module

        self.train_data, self.test_data, self.D = load_dataset(self.data_module)
        self.S_train, self.S_test = self.train_data.S.numpy(), self.test_data.S.numpy()
        self.Y_train, self.Y_test = self.train_data.Y.numpy(), self.test_data.Y.numpy()

        self.batch_size = batch_size
        self.epochs = 200
        self.verbose = 100
        
        self.lr = 1e-3
        self.x_dim = self.train_data.X.shape[1]
        self.s_dim = self.train_data.S.max().item()+1
        self.h_dim = 64
        self.z_dim = 8

        self.lg_beta = 7

        self.beta = 10**self.lg_beta

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'



    def fit(self):

        self.model = FairDisCo(self.x_dim, self.h_dim, self.z_dim, self.s_dim, self.D)
        
        print(f'model training started!')
        
        self.model.fit(train_data = self.train_data, epochs = self.epochs, lr = self.lr, batch_size = self.batch_size, verbose = self.verbose,  beta = self.beta, device = self.device)
        
        torch.save(self.model.state_dict(), './model/FairDisCo_{}_{}.pkl'.format(self.lg_beta, self.data_module.sensitive_attr))  # todo: add functionalities to automatically create folder