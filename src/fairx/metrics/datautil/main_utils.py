import xgboost as xgb

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

import numpy as np

class DataUtilsMetrics():
    
    """
    DataUtilsMetrics is the class that handles the data utility evaluation metrics. It calculates the Precision, Recall, AUROC, Accuracy and F1 Score. 
    """

    def __init__(self, dataset, enable_categorical = False):

        """

        Input: dataset: spllited dataset in the format of a tuple (train_x, test_x, train_y, test_y, train_s, test_s) 

        Return: Dictionary containing the result (Precision, Recall, Accuracy, F1 Score, and AUROC) 
        """

        super().__init__()

        self.dataset = dataset

        self.enable_categorical = enable_categorical

    def evaluate_utility(self):

        train_x, test_x, train_y, test_y, train_s, test_s = self.dataset

        clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42, enable_categorical = self.enable_categorical)

        clf = clf.fit(train_x, train_y)
        
        Y_test_hat = clf.predict_proba(test_x)[:,1]
        
        Y_hat=clf.predict(test_x)
        
        prs_scr = precision_score(test_y, Y_hat)
        
        rec_scr = recall_score(test_y, Y_hat)
        
        acc_scr = accuracy_score(test_y, Y_hat)
        
        f1_scr = f1_score(test_y, Y_hat)
        
        auroc = roc_auc_score(test_y, Y_hat)
        
        output = {
        
            'Precision' : prs_scr,
            'Recall' : rec_scr,
            'Accuracy' : acc_scr,
            'F1 Score': f1_scr,
            'Auroc' : auroc
            
        }

        return output

        