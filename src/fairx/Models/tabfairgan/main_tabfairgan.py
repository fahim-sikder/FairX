# Copyright (c) 2021 Amirarsalan Rajabi

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE
#
# Modified work Copyright (c) 2024 Md Fahim Sikder
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE

### 

import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from collections import OrderedDict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import QuantileTransformer

from .utils import *


class TabFairGAN():

    def __init__(self, under_previlaged= None, y_desire = None):

        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.S_under = under_previlaged

        self.Y_desire = y_desire


    def data_preprocess_for_tabfairgan(self, dataset_module):

        self.batch_size = 256

        self.categorical_transformer_tab = OneHotEncoder(handle_unknown="ignore")

        self.numeric_transformer_tab = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')

        self.cat_tf_tab = self.categorical_transformer_tab.fit_transform(dataset_module.cat_data)
        
        self.num_tf_tab = self.numeric_transformer_tab.fit_transform(dataset_module.num_data)

        self.catenated_data = np.hstack((self.num_tf_tab, self.cat_tf_tab.toarray()))

        self.cat_lens = [i.shape[0] for i in self.categorical_transformer_tab.categories_]

        self.discrete_columns_ordereddict = OrderedDict(zip(dataset_module.cat_feat, self.cat_lens))

        self.S_start_index = len(dataset_module.num_feat) + sum(
            list(self.discrete_columns_ordereddict.values())[:list(self.discrete_columns_ordereddict.keys()).index(dataset_module.sensitive_attr)])
        
        self.Y_start_index = len(dataset_module.num_feat) + sum(
            list(self.discrete_columns_ordereddict.values())[:list(self.discrete_columns_ordereddict.keys()).index(dataset_module.target_attr[0])])

        if self.categorical_transformer_tab.categories_[list(self.discrete_columns_ordereddict.keys()).index(dataset_module.sensitive_attr)][0] == self.S_under:
            self.underpriv_index = 0
            self.priv_index = 1
        else:
            self.underpriv_index = 1
            self.priv_index = 0
            
        if self.categorical_transformer_tab.categories_[list(self.discrete_columns_ordereddict.keys()).index(dataset_module.target_attr[0])][0] == self.Y_desire:
            self.desire_index = 0
            self.undesire_index = 1
        else:
            self.desire_index = 1
            self.undesire_index = 0

        self.input_dim = self.catenated_data.shape[1]
        
        self.X_train, self.X_test = train_test_split(self.catenated_data,test_size=0.1, shuffle=True)
        
        self.data_train = self.X_train.copy()
        self.data_test = self.X_test.copy()

        self.torch_data = torch.from_numpy(self.data_train).float()


        self.train_ds = TensorDataset(self.torch_data)
        self.train_dl = DataLoader(self.train_ds, batch_size = self.batch_size, drop_last=True)

        return self.categorical_transformer_tab, self.numeric_transformer_tab, self.input_dim, self.discrete_columns_ordereddict, dataset_module.num_feat, self.train_dl, \
            self.data_train, self.data_test, self.S_start_index, self.Y_start_index, self.underpriv_index, self.priv_index, self.undesire_index, self.desire_index

    def fit(self, dataset_module, batch_size, epochs):

        fair_epochs=10
        lamda=0.5

        ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_index, priv_index, \
                undesire_index, desire_index = self.data_preprocess_for_tabfairgan(dataset_module)

        self.generator = Generator(input_dim, continuous_columns, discrete_columns).to(self.device)
        self.critic = Critic(input_dim).to(self.device)

        self.second_critic = FairLossFunc(S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index).to(self.device)

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.gen_optimizer_fair = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.crit_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # loss = nn.BCELoss()
        critic_losses = []
        cur_step = 0
        for i in range(epochs):
            # j = 0
            print("epoch {}".format(i + 1))
            ############################
            if i + 1 <= (epochs - fair_epochs):
                print("training for accuracy")
            if i + 1 > (epochs - fair_epochs):
                print("training for fairness")
            for data in train_dl:
                data[0] = data[0].to(self.device)
                crit_repeat = 4
                mean_iteration_critic_loss = 0
                for k in range(crit_repeat):
                    # training the critic
                    self.crit_optimizer.zero_grad()
                    fake_noise = torch.randn(size=(batch_size, input_dim), device=self.device).float()
                    fake = self.generator(fake_noise)
    
                    crit_fake_pred = self.critic(fake.detach())
                    crit_real_pred = self.critic(data[0])
    
                    epsilon = torch.rand(batch_size, input_dim, device=self.device, requires_grad=True)
                    gradient = get_gradient(self.critic, data[0], fake.detach(), epsilon)
                    gp = gradient_penalty(gradient)
    
                    crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda=10)
    
                    mean_iteration_critic_loss += crit_loss.item() / crit_repeat
                    crit_loss.backward(retain_graph=True)
                    self.crit_optimizer.step()
                #############################
                if cur_step > 50:
                    critic_losses += [mean_iteration_critic_loss]
    
                #############################
                if i + 1 <= (epochs - fair_epochs):
                    # training the generator for accuracy
                    self.gen_optimizer.zero_grad()
                    fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=self.device).float()
                    fake_2 = self.generator(fake_noise_2)
                    crit_fake_pred = self.critic(fake_2)
    
                    gen_loss = get_gen_loss(crit_fake_pred)
                    gen_loss.backward()
    
                    # Update the weights
                    self.gen_optimizer.step()
    
                ###############################
                if i + 1 > (epochs - fair_epochs):
                    # training the generator for fairness
                    self.gen_optimizer_fair.zero_grad()
                    fake_noise_2 = torch.randn(size=(batch_size, input_dim), device=self.device).float()
                    fake_2 = self.generator(fake_noise_2)
    
                    crit_fake_pred = self.critic(fake_2)
    
                    gen_fair_loss = self.second_critic(fake_2, crit_fake_pred, lamda)
                    gen_fair_loss.backward()
                    self.gen_optimizer_fair.step()
                cur_step += 1

        print(f'Training Complete!')