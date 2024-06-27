# Original work Copyright (c) Microsoft Corporation and Fairlearn contributors.

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

import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from fairlearn.preprocessing import CorrelationRemover

from fairx.metrics import FairnessUtils, DataUtilsMetrics

from fairx.utils import setSeed

#####


setSeed(2022)

class CorrRemover():

    """
    Correlation Remover Technique, Pre-processing bias removal technique.
    """

    def __init__(self, data_module, sensitive_attr_to_remove, remove_intensity = 1.0):

        """
        Input: data_module, BaseDataClass module
                sensitive_attr_to_remove: string, feature name
                remove_intensity: intensity of the feature removal, number between 0.0 - 1.0, where 1.0 means the correlation will be removed maximum.
        """

        super().__init__()

        self.sensitive_attr_to_remove = sensitive_attr_to_remove

        self.remove_intensity = remove_intensity

        self.data_module = data_module

        self.enc = OrdinalEncoder()

        self.res = {'Methods': 'Correlation Remover'}

        self.df = self.data_module.data.copy()

        self.col_list = self.df.columns.to_list()

        for col in self.col_list:
    
            self.df[[col]] = self.enc.fit_transform(self.df[[col]])

        if data_module.attach_target:

            self.target = self.df[self.data_module.target_attr].values
    
            self.df = self.df.drop(self.data_module.target_attr, axis=1)

            self.col_list.remove(self.data_module.target_attr[0])

        else:

            self.target = self.enc.fit_transform(self.data_module.raw_data.frame[self.data_module.target_attr])

        self.col_list.remove(self.sensitive_attr_to_remove)
        

        self.corr_remover = CorrelationRemover(sensitive_feature_ids = [self.sensitive_attr_to_remove], alpha = self.remove_intensity)

    def fit(self):

        """
        Return: Repaired dataset, Result
        """

        self.new_df =  self.corr_remover.fit_transform(self.df)

        self.new_df = pd.DataFrame(self.new_df, columns = self.col_list)

        self.new_df[self.sensitive_attr_to_remove] = self.df[self.sensitive_attr_to_remove]

        self.sensitive_attr_val = self.new_df[self.sensitive_attr_to_remove].values

        self.splitted_data = train_test_split(self.new_df.values, self.target, self.sensitive_attr_val, test_size=0.3, random_state=42, stratify=self.target)

        self.fairness_utils = FairnessUtils(self.splitted_data)

        self.data_utils = DataUtilsMetrics(self.splitted_data)

        self.res.update(self.data_utils.evaluate_utility())

        self.res.update(self.fairness_utils.evaluate_fairness())

        return self.new_df, self.res
