# Original work Copyright (c) 2021 Amirarsalan Rajabi

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

import time

from .utils import *
from fairx.models.baseclass import BaseModelClass

from fairx.dataset import CustomDataClass
from fairx.metrics import SyntheticEvaluation
from fairx.utils import setSeed

setSeed(2022)


class TabFairGAN(BaseModelClass):

    def __init__(self, under_previlaged= None, y_desire = None):

    # def __init__(self, dataset_module, batch_size, epochs, under_previlaged= None, y_desire = None):

        """
        TabFairGAN [1] implementation, 
        
        Input: 
        
        under_previlaged:under privileged feature in the dataset,
        y_desire: target that needs to optimize
        
        [1] Rajabi, Amirarsalan, and Ozlem Ozmen Garibay. "Tabfairgan: Fair tabular data generation with generative adversarial networks." Machine Learning and Knowledge Extraction 4.2 (2022): 488-501.
        """

        super(BaseModelClass, self).__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.S_under = under_previlaged

        self.Y_desire = y_desire


    def preprocess_data(self, dataset_module):

        self.batch_size = 256

        self.cat_feat = []

        self.num_feat = []

        self.categorical_transformer_tab = OneHotEncoder(handle_unknown="ignore")

        self.numeric_transformer_tab = QuantileTransformer(n_quantiles=2000, output_distribution='uniform')

        self.data = dataset_module.raw_data.frame.copy()

        for col in self.data.columns:

            if(self.data[col].dtype == 'object'):
    
                self.cat_feat.append(col)
        
                self.data[col] = self.data[col].astype('category')
    
            elif(self.data[col].dtype == 'category'):
    
                self.cat_feat.append(col)
    
            else:
    
                self.num_feat.append(col)
                

        self.num_data = self.data[self.num_feat].values

        self.cat_data = self.data[self.cat_feat].values
    
        self.cat_tf_tab = self.categorical_transformer_tab.fit_transform(self.cat_data)
        
        self.num_tf_tab = self.numeric_transformer_tab.fit_transform(self.num_data)

        self.catenated_data = np.hstack((self.num_tf_tab, self.cat_tf_tab.toarray()))

        self.cat_lens = [i.shape[0] for i in self.categorical_transformer_tab.categories_]

        self.discrete_columns_ordereddict = OrderedDict(zip(self.cat_feat, self.cat_lens))

        self.S_start_index = len(self.num_feat) + sum(
            list(self.discrete_columns_ordereddict.values())[:list(self.discrete_columns_ordereddict.keys()).index(dataset_module.sensitive_attr)])
        
        if self.categorical_transformer_tab.categories_[list(self.discrete_columns_ordereddict.keys()).index(dataset_module.sensitive_attr)][0] == self.S_under:
            self.underpriv_index = 0
            self.priv_index = 1
        else:
            self.underpriv_index = 1
            self.priv_index = 0

        if self.data[dataset_module.target_attr[0]].dtype == 'category':
        
            self.Y_start_index = len(dataset_module.num_feat) + sum(
                list(self.discrete_columns_ordereddict.values())[:list(self.discrete_columns_ordereddict.keys()).index(dataset_module.target_attr[0])])
                
            if self.categorical_transformer_tab.categories_[list(self.discrete_columns_ordereddict.keys()).index(dataset_module.target_attr[0])][0] == self.Y_desire:
                self.desire_index = 0
                self.undesire_index = 1
            else:
                self.desire_index = 1
                self.undesire_index = 0
        

        else:

            self.Y_start_index = self.num_feat.index(dataset_module.target_attr[0])

            if self.Y_desire == 1:

                self.desire_index = 1
                self.undesire_index = 0

            else:
                self.desire_index = 0
                self.undesire_index = 1 

        self.input_dim = self.catenated_data.shape[1]
        
        self.X_train, self.X_test = train_test_split(self.catenated_data,test_size=0.1, shuffle=True)
        
        self.data_train = self.X_train.copy()
        self.data_test = self.X_test.copy()

        self.torch_data = torch.from_numpy(self.data_train).float()


        self.train_ds = TensorDataset(self.torch_data)
        self.train_dl = DataLoader(self.train_ds, batch_size = self.batch_size, drop_last=True)

        return self.categorical_transformer_tab, self.numeric_transformer_tab, self.input_dim, self.discrete_columns_ordereddict, self.num_feat, self.train_dl, \
            self.data_train, self.data_test, self.S_start_index, self.Y_start_index, self.underpriv_index, self.priv_index, self.undesire_index, self.desire_index

    def get_original_data(self, df_transformed, df_orig, ohe, scaler):
        df_ohe_int = df_transformed[:, :df_orig.select_dtypes(['float', 'integer']).shape[1]]
        df_ohe_int = scaler.inverse_transform(df_ohe_int)
        df_ohe_cats = df_transformed[:, df_orig.select_dtypes(['float', 'integer']).shape[1]:]
        df_ohe_cats = ohe.inverse_transform(df_ohe_cats)
        df_int = pd.DataFrame(df_ohe_int, columns=df_orig.select_dtypes(['float', 'integer']).columns)
        df_cat = pd.DataFrame(df_ohe_cats, columns=df_orig.select_dtypes('category').columns)
        return pd.concat([df_int, df_cat], axis=1)

    def fit(self, dataset_module, batch_size, epochs):

        self.dataset_module = dataset_module

        self.batch_size = batch_size

        self.epochs = epochs

        fair_epochs=10
        lamda=0.5

        ohe, scaler, input_dim, discrete_columns, continuous_columns, train_dl, data_train, data_test, S_start_index, Y_start_index, underpriv_index, priv_index, \
                undesire_index, desire_index = self.preprocess_data(self.dataset_module)

        self.generator = Generator(input_dim, continuous_columns, discrete_columns).to(self.device)
        self.critic = Critic(input_dim).to(self.device)

        self.second_critic = FairLossFunc(S_start_index, Y_start_index, underpriv_index, priv_index, undesire_index, desire_index).to(self.device)

        self.gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.gen_optimizer_fair = torch.optim.Adam(self.generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
        self.crit_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.0002, betas=(0.5, 0.999))

        # loss = nn.BCELoss()
        critic_losses = []
        cur_step = 0
        for i in range(self.epochs):
            # j = 0
            print("epoch {}".format(i + 1))
            ############################
            if i + 1 <= (self.epochs - fair_epochs):
                print("training for accuracy")
            if i + 1 > (self.epochs - fair_epochs):
                print("training for fairness")
            for data in train_dl:
                data[0] = data[0].to(self.device)
                crit_repeat = 4
                mean_iteration_critic_loss = 0
                for k in range(crit_repeat):
                    # training the critic
                    self.crit_optimizer.zero_grad()
                    fake_noise = torch.randn(size=(self.batch_size, input_dim), device=self.device).float()
                    fake = self.generator(fake_noise)
    
                    crit_fake_pred = self.critic(fake.detach())
                    crit_real_pred = self.critic(data[0])
    
                    epsilon = torch.rand(self.batch_size, input_dim, device=self.device, requires_grad=True)
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
                if i + 1 <= (self.epochs - fair_epochs):
                    # training the generator for accuracy
                    self.gen_optimizer.zero_grad()
                    fake_noise_2 = torch.randn(size=(self.batch_size, input_dim), device=self.device).float()
                    fake_2 = self.generator(fake_noise_2)
                    crit_fake_pred = self.critic(fake_2)
    
                    gen_loss = get_gen_loss(crit_fake_pred)
                    gen_loss.backward()
    
                    # Update the weights
                    self.gen_optimizer.step()
    
                ###############################
                if i + 1 > (self.epochs - fair_epochs):
                    # training the generator for fairness
                    self.gen_optimizer_fair.zero_grad()
                    fake_noise_2 = torch.randn(size=(self.batch_size, input_dim), device=self.device).float()
                    fake_2 = self.generator(fake_noise_2)
    
                    crit_fake_pred = self.critic(fake_2)
    
                    gen_fair_loss = self.second_critic(fake_2, crit_fake_pred, lamda)
                    gen_fair_loss.backward()
                    self.gen_optimizer_fair.step()
                cur_step += 1

        print(f'Training Complete!')

        df_generated = self.generator(torch.randn(size=(32561, input_dim), device=self.device)).cpu().detach().numpy()

        generated_tf = self.get_original_data(df_generated, self.data, ohe, scaler)

        csv_file_name = f'{time.time():.2f}-TabFairGAN-{self.dataset_module.dataset_name}-{self.dataset_module.sensitive_attr}.csv'

        generated_tf.to_csv(f'{csv_file_name}', index = False)

        fake_data_class = CustomDataClass(csv_file_name, self.dataset_module.sensitive_attr, self.dataset_module.target_attr, attach_target = self.dataset_module.attach_target)

        synth = SyntheticEvaluation(self.dataset_module, fake_data_class)

        print(synth.calculate_alpha_precision())

        print(f'generated data saved!')