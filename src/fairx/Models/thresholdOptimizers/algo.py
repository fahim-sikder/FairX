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

from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio

from sklearn.pipeline import Pipeline
import xgboost as xgb


#####


class ThresholdAlgorithm():

    def __init__(self):

        super().__init__()

        pass

    def fit(self, dataset):

        X_train, X_test, y_train, y_test, A_train, A_test = dataset

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

        m_dpr_opt = demographic_parity_ratio(y_test, y_pred_opt_before, sensitive_features=A_test)
        
        m_eqo_opt = equalized_odds_ratio(y_test, y_pred_opt_before, sensitive_features=A_test)
        
        print(f'Value of demographic parity ratio (before post-processing): {round(m_dpr_opt, 2)}')
        
        print(f'Value of equal odds ratio (before post-processing): {round(m_eqo_opt, 2)}') 
        

        
        threshold_optimizer = ThresholdOptimizer(
            estimator=pipeline_cls,
            constraints="demographic_parity",
            predict_method="predict_proba",
            prefit=False,
        )
        
        threshold_optimizer.fit(X_train, y_train, sensitive_features=A_train)

        y_pred_opt = threshold_optimizer.predict(X_test, sensitive_features=A_test)

        m_dpr_opt = demographic_parity_ratio(y_test, y_pred_opt, sensitive_features=A_test)
        
        m_eqo_opt = equalized_odds_ratio(y_test, y_pred_opt, sensitive_features=A_test)
        
        print(f'Value of demographic parity ratio (after post-processing): {round(m_dpr_opt, 2)}')
        
        print(f'Value of equal odds ratio (after post-processing): {round(m_eqo_opt, 2)}') 