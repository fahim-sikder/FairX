# Original work Copyright 2022, Synthcity Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


from fairx.dataset import BaseDataClass

from synthcity.plugins import Plugin, Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

from fairx.dataset import CustomDataClass
from fairx.metrics import SyntheticEvaluation

from fairx.utils import setSeed

import warnings
warnings.filterwarnings('ignore')

import time

setSeed(2022)

class Decaf():

    def __init__(self, dataset_module, batch_size, n_iter, generated_sample_size):

        """
        Decaf [1] implementation, 

        Input: 
        
        dataset_module: module of BaseDataClass,
        batch_size: batch size for the model,
        n_iter: iteration for training,
        generated_sample_size: (int) number of samples to generate

        Return: trained model
        
        [1] Van Breugel, Boris, et al. "Decaf: Generating fair synthetic data using causally-aware generative networks." Advances in Neural Information Processing Systems 34 (2021): 22221-22233.
        
        """

        super().__init__()

        self.dataset_module = dataset_module

        self.batch_size = batch_size

        self.n_iter = n_iter

        self.generated_sample_size = generated_sample_size

    def fit(self):

        self.decaf_loader = GenericDataLoader(self.dataset_module.data,
                                            target_column=self.dataset_module.target_attr[0],
                                            sensitive_columns=self.dataset_module.sensitive_attr)

        self.train_data, self.test_data = self.decaf_loader.train(), self.decaf_loader.test()

        self.model = Plugins().get("decaf", batch_size = self.batch_size, n_iter = self.n_iter)

        print(f'Training Started')

        self.model.fit(self.train_data)

        generated_tf = self.model.generate(self.generated_sample_size)

        generated_tf_decaf = generated_tf.dataframe()

        dataframe_name = f'{time.time():.2f}-Decaf-{self.dataset_module.dataset_name}-{self.dataset_module.sensitive_attr}.csv'
    
        generated_tf_decaf.to_csv(f'{dataframe_name}', index = False)

        fake_data_class = CustomDataClass(dataframe_name, self.dataset_module.sensitive_attr, self.dataset_module.target_attr, attach_target = self.dataset_module.attach_target)

        synth = SyntheticEvaluation(self.dataset_module, fake_data_class)

        print(synth.calculate_alpha_precision())

        print(f'Training Completed')

        print(f'Generated Sample Saved')

        return self.model

