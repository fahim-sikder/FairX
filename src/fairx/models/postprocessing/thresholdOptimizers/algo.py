# Copyright (c) Microsoft Corporation and Fairlearn contributors.

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

# TODO: Integrate fairx.metrics 

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio

from fairx.metrics import DataUtilsMetrics

from fairx.utils import setSeed

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb


#####

setSeed(2022)


class ThresholdAlgorithm():

    """
    Threshold Optimizer Technique, Post-processing bias removal technique.
    """

    def __init__(self, data_module):

        """
        Input: data_module, BaseDataClass module

        """

        super().__init__()

        self.res = {'Methods': 'Threshold Optimizer'}

        self.data_module = data_module

        self.enc = OrdinalEncoder()

        self.df = self.data_module.data.copy()

        self.col_list = self.df.columns.to_list()

        for col in self.col_list:
    
            self.df[[col]] = self.enc.fit_transform(self.df[[col]])

        if data_module.attach_target:

            self.target = self.df[self.data_module.target_attr].values
    
            self.df = self.df.drop(self.data_module.target_attr, axis=1)

        else:

            self.target = self.enc.fit_transform(self.data_module.raw_data.frame[self.data_module.target_attr])

        self.sensitive_attr_val = self.df[self.data_module.sensitive_attr].values

        self.splitted_data = train_test_split(self.df.values, self.target, self.sensitive_attr_val, test_size=0.3, random_state=42, stratify=self.target)


    def fit(self):

        """
        Return: Result in dictionary format
        """

        X_train, X_test, y_train, y_test, A_train, A_test = self.splitted_data

        pipeline_cls = Pipeline(
            steps=[
                (
                    "classifier",
                    xgb.XGBClassifier(objective="binary:logistic", random_state=42),
                ),
            ]
        )
        
        pipeline_cls.fit(X_train, y_train)

        y_pred_opt_before = pipeline_cls.predict(X_test)


        threshold_optimizer = ThresholdOptimizer(
            estimator=pipeline_cls,
            constraints="demographic_parity",
            predict_method="predict_proba",
            prefit=False,
        )
        
        threshold_optimizer.fit(X_train, y_train, sensitive_features=A_train)

        y_pred_opt = threshold_optimizer.predict(X_test, sensitive_features=A_test)

        self.data_utils = DataUtilsMetrics(self.splitted_data)

        self.res.update(self.data_utils.evaluate_utility())

        m_dpr_opt = demographic_parity_ratio(y_test, y_pred_opt, sensitive_features=A_test)
        
        m_eqo_opt = equalized_odds_ratio(y_test, y_pred_opt, sensitive_features=A_test)

        fair_output =  {'Demographic Parity Ratio' : m_dpr_opt,
       'Equalized Odd Ratio': m_eqo_opt}

        self.res.update(fair_output)

        return self.res