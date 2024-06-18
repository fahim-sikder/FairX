import xgboost as xgb
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio


class FairnessUtils():

    def __init__(self, dataset, enable_categorical = False):

        super().__init__()

        self.dataset = dataset
        self.enable_categorical = enable_categorical

    def evaluate_fairness(self):
    
        train_x, test_x, train_y, test_y, train_s, test_s = self.dataset
    
        clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42, enable_categorical = self.enable_categorical)
    
        clf = clf.fit(train_x, train_y)
    
        Y_hat=clf.predict(test_x)
    
        demo_pari = demographic_parity_ratio(test_y, Y_hat, sensitive_features=test_s)
    
        eq_odd = equalized_odds_ratio(test_y, Y_hat, sensitive_features=test_s)
    
        output =  {'Demographic Parity Ratio' : demo_pari,
               'Equalized Odd Ratio': eq_odd}
    
        return output