import matplotlib.pylab as pl
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split

import shap



def explain_prediction(X, y, save_fig = False):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
    
    d_train = xgboost.DMatrix(X_train, label=y_train, enable_categorical = True)
    
    d_test = xgboost.DMatrix(X_test, label=y_test, enable_categorical = True)

    params = {
        "eta": 0.01,
        "objective": "binary:logistic",
        "subsample": 0.5,
        "base_score": np.mean(y_train),
        "eval_metric": "logloss",
        }
    
    model = xgboost.train(
        params,
        d_train,
        5000,
        evals=[(d_test, "test")],
        verbose_eval=100,
        early_stopping_rounds=20,
        )


    # this takes a minute or two since we are explaining over 30 thousand samples in a model with over a thousand trees
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return shap.summary_plot(shap_values, X)