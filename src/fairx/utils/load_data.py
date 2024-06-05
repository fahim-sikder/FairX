import os
import random
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.utils import shuffle


import xgboost as xgb
from fairlearn.metrics import demographic_parity_ratio, equalized_odds_ratio

import warnings
warnings.filterwarnings('ignore')


class LoadData():

    def __init__(self, dataset_name):

        super(LoadData, self).__init__()

        self.dataset_name = dataset_name

        

    pass

def setSeed(seed=2022):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

class VectorDataset(Dataset):
    def __init__(self, X, S, Y, C=None):
        self.X = X
        self.S = S
        self.Y = Y
        self.C = C

    def __getitem__(self, i):
        x, s, y = self.X[i], self.S[i], self.Y[i]
        if self.C != None:
            return x, self.C[i], s, y
        else:
            return x, s, y
    
    def __len__(self):
        return self.X.shape[0]

def getOneHot(df):
    C = df.values
    D = df.nunique().values.tolist()
    X = []
    for col in df.columns:
        X.append(pd.get_dummies(df[col]).values)
    X = np.concatenate(X, axis=1)
    return X, C, D

def getBinary(df, cols):
    labels = df[cols].apply(lambda s: np.median(s)).values
    x = df[cols].values
    xs = np.zeros_like(x)
    for j in range(len(labels)):
        if x[:,j].max() == labels[j]:
            xs[:,j] = x[:,j]
        else:
            xs[:,j] = (x[:,j] > labels[j]).astype(int)
    df = pd.DataFrame(xs, columns=cols)
    return df

def getDataset(df, S, Y, num_train):
    S = torch.LongTensor(S)
    Y = torch.FloatTensor(Y)
    S_train, S_test = S[:num_train], S[num_train:]
    Y_train, Y_test = Y[:num_train], Y[num_train:]

    df = df.apply(lambda col: LabelEncoder().fit_transform(col))

    X, C, D = getOneHot(df)
    X = torch.FloatTensor(X)
    C = torch.LongTensor(C)

    X_train, X_test = X[:num_train], X[num_train:]
    C_train, C_test = C[:num_train], C[num_train:]
    
    train_data = VectorDataset(X_train, S_train, Y_train, C_train)
    test_data = VectorDataset(X_test, S_test, Y_test, C_test)

    return train_data, test_data, D

def fairloadAdult(pro_att):
    """
    Adult Census Income: Individual information from 1994 U.S. census. Goal is predicting income >$50,000.
    Protected Attribute: sex / race
    """
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', \
        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 'native-country', 'salary']

    df_train = pd.read_csv('./data/Adult/adult.data', names=cols)
    df_test = pd.read_csv('./data/Adult/adult.test', names=cols, skiprows=1)


    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)
    # print('train_size {}, test_size {}'.format(num_train, num_test))

    male_df = df[df['sex']=='Male'][:16192]
    female_df = df[df['sex']=='Female']

    num_train = 26000
    # num_test = df_test.shape[0]

    train_set = pd.concat([male_df[:num_train//2], female_df[:num_train//2]], ignore_index = True)

    test_set = pd.concat([male_df[num_train//2:], female_df[num_train//2:]], ignore_index = True)

    train_set = train_set.sample(n = len(train_set))

    test_set = test_set.sample(n = len(test_set))

    df = pd.concat([train_set, test_set], ignore_index=True)
    
    df['age'] = pd.cut(df['age'], bins=8, labels=False)
    df['hours-per-week'] = pd.cut(df['hours-per-week'], bins=8, labels=False)
    df['fnlwgt'] = pd.cut(np.log(df['fnlwgt']), bins=8, labels=False)
    df[['capital-gain', 'capital-loss']] = getBinary(df, ['capital-gain', 'capital-loss'])

    if pro_att == 'sex':
        S = (df['sex'] == 'Male').values.astype(int)
        del df['sex']

    if pro_att == 'race':
        S = (df['race'] == 'Black').values.astype(int)
        del df['race']
    
    Y = (df['salary'].apply(lambda x: x == '<=50K' or x == '<=50K.')).values.astype(int)
    del df['salary']

    return getDataset(df, S, Y, num_train)

def loadAdult(pro_att):
    """
    Adult Census Income: Individual information from 1994 U.S. census. Goal is predicting income >$50,000.
    Protected Attribute: sex / race
    """
    cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', \
        'relationship', 'race', 'sex', 'capital-gain', 'capital-loss','hours-per-week', 'native-country', 'salary']

    df_train = pd.read_csv('./data/Adult/adult.data', names=cols)
    df_test = pd.read_csv('./data/Adult/adult.test', names=cols, skiprows=1)

    num_train = df_train.shape[0]
    num_test = df_test.shape[0]
    df = pd.concat([df_train, df_test], ignore_index=True)
    df = df.apply(lambda v: v.astype(str).str.strip() if v.dtype == "object" else v)
    print('train_size {}, test_size {}'.format(num_train, num_test))
    
    df['age'] = pd.cut(df['age'], bins=8, labels=False)
    df['hours-per-week'] = pd.cut(df['hours-per-week'], bins=8, labels=False)
    df['fnlwgt'] = pd.cut(np.log(df['fnlwgt']), bins=8, labels=False)
    df[['capital-gain', 'capital-loss']] = getBinary(df, ['capital-gain', 'capital-loss'])

    if pro_att == 'sex':
        S = (df['sex'] == 'Male').values.astype(int)
        del df['sex']

    if pro_att == 'race':
        S = (df['race'] == 'Black').values.astype(int)
        del df['race']
    
    Y = (df['salary'].apply(lambda x: x == '<=50K' or x == '<=50K.')).values.astype(int)
    del df['salary']

    return getDataset(df, S, Y, num_train)

def loadCompas(pro_att):
    """
    Compas: Contains criminal history of defendants. Goal predicting re-offending in future
    Protected Attribute: sex / race
    """
    df = pd.read_csv('./data/compas-scores-two-years.csv')
    drop_cols = ['id','name','first','last','compas_screening_date',
                'dob', 'juv_fel_count', 'decile_score',
                'juv_misd_count','juv_other_count','days_b_screening_arrest',
                'c_jail_in','c_jail_out','c_case_number','c_offense_date','c_arrest_date',
                'c_days_from_compas','c_charge_desc','is_recid','r_case_number','r_charge_degree',
                'r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in','r_jail_out',
                'violent_recid','is_violent_recid','vr_case_number','vr_charge_degree','vr_offense_date',
                'vr_charge_desc','type_of_assessment','decile_score','score_text','screening_date',
                'v_type_of_assessment','v_decile_score','v_score_text','v_screening_date','in_custody',
                'out_custody','start','end','event']
    df = df.drop(drop_cols, axis=1)
    
    df = shuffle(df)
    num_train = int(0.8*df.shape[0])
    print('train_size {}, test_size {}'.format(num_train, df.shape[0]-num_train))
    
    df['age'] = pd.cut(df['age'], bins=5, labels=False)
    df['priors_count'] = df['priors_count'].apply(lambda x: x if x<9 else 9)

    if pro_att == 'sex':
        S = (df['sex'] == 'Male').values.astype(int)
        del df['sex']

    if pro_att == 'race':
        S = (df['race'] == 'African-American').values.astype(int)
        del df['race']
    
    Y = df['two_year_recid'].values.astype(int)
    del df['two_year_recid']

    return getDataset(df, S, Y, num_train)

def loadBank():
    """
    Bank Marketing: Contains marketing data of a Portuguese bank. Goal predicting term deposit.
    Protected Attribute: Age
    """
    cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month',
        'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    df = pd.read_csv('./data/BankMarketing/bank-full.csv', delimiter=';', header=0)
    df = shuffle(df).reset_index(drop=True)
    num_train = int(df.shape[0]*0.8)
    print('train_size {}, test_size {}'.format(num_train, df.shape[0]-num_train))

    for col in ['balance', 'day', 'duration']:
        df[col] = pd.qcut(df[col], q=5, labels=False)
    
    del df['pdays']
    del df['previous']
    del df['campaign']

    S = (df['age'] >= 30).values.astype(int)
    del df['age']
    Y = (df['y'] == 'yes').values.astype(int)
    del df['y']

    return getDataset(df, S, Y, num_train)

def loadCredit():
    """
    Default Credit: Customer information for people from Taiwan. Goal is predicting default payment.
    Protected Attribute: Sex
    """
    df = pd.read_csv('./data/DefaultCredit.csv', header=0)
    df = shuffle(df).reset_index(drop=True)
    num_train = int(df.shape[0]*0.8)
    print('train_size {}, test_size {}'.format(num_train, df.shape[0]-num_train))

    df['X1'] = pd.qcut(df['X1'], q=5, labels=False)
    df['X5'] = pd.qcut(df['X5'], q=5, labels=False)
    for i in range(6, 12):
        df['X'+str(i)] = pd.qcut(df['X'+str(i)], q=2, labels=False)
    
    for i in range(12, 24):
        df['X'+str(i)] = pd.qcut(df['X'+str(i)], q=4, labels=False)

    S = (df['X2']==1).values.astype(int)
    del df['X2']
    Y = df['Y'].values.astype(int)
    del df['Y']
    
    return getDataset(df, S, Y, num_train)

def loadStudent():
    """
    Student Performance: Student achievement of two Portuguese schools. Target is final year grade.
    Protected Attribute: Sex
    """
    df = pd.read_csv('./data/Student.csv')
    df = shuffle(df).reset_index(drop=True)
    num_train = int(df.shape[0]*0.8)
    print('train_size {}, test_size {}'.format(num_train, df.shape[0]-num_train))

    df['age'] = df['age'].apply(lambda x: 0 if x >= 18 else (1 if x >= 16 else 2))
    df['absences'] = pd.cut(df['absences'], bins=5, labels=False)
    del df['school']
    del df['G1']
    del df['G2']

    S = (df['sex'] == 'M').values.astype(int)
    Y = (df['Probability'] > 13).values.astype(int)
    del df['sex']
    del df['Probability']

    return getDataset(df, S, Y, num_train)

def loadHealth():
    """
    Heritage Health Prize: The task is to predict whether a patient will spend any days in the hospital in the next year.
    Protected Attribute: Age
    """
    df = pd.read_csv('./data/Health.csv')
    df = df[df['YEAR_t'].isin(['Y2'])]
    df = df[(df['sexMISS'] == 0)&(df['age_MISS'] == 0)]
    df = shuffle(df).reset_index(drop=True)
    
    num_train = int(df.shape[0]*0.8)
    print('train_size {}, test_size {}'.format(num_train, df.shape[0]-num_train))
    
    age = df[['age_%d5' % (i) for i in range(0, 9)]].values.argmax(axis=1)
    sex = df[['sexMALE', 'sexFEMALE']].values.argmax(axis=1)
    
    Y = (df['DaysInHospital'] > 0).values.astype(int)
    drop_cols = ['age_%d5' % (i) for i in range(0, 9)] + ['sexMALE', 'sexFEMALE',  'sexMISS', 'age_MISS', 'trainset', 'DaysInHospital', 'MemberID_t', 'YEAR_t']
    
    df = df.drop(drop_cols, axis=1)
    df = getBinary(df, df.columns)

    df['sex'] = sex
    S = (age > 6).astype(int)

    return getDataset(df, S, Y, num_train)

def loadGerman():
    """
    German Credit: Personal information about individuals & predicts good or bad credit.
    Protected Attribute: Sex
    """
    df = pd.read_csv('./data/German/german.data', names=range(1,22), sep=' ')
    df = shuffle(df)
    num_train = int(0.8*df.shape[0])
    print('train_size {}, test_size {}'.format(num_train, df.shape[0]-num_train))
    
    category_cols = [1,3,4,6,7,8,9,10,11,12,14,15,16,17,18,19,20]
    continuous_cols = [2,5]
    sensitive_col = 13
    label_col = 21
    
    df[2] = pd.cut(df[2], bins=5, labels=False)
    df[5] = pd.cut(df[5], bins=5, labels=False)

    S = (df[sensitive_col].values>=30).astype(int)
    Y = (df[label_col]-1).values
    del df[sensitive_col]
    del df[label_col]

    return getDataset(df, S, Y, num_train)

def loadMnist(color=True):
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(64), transforms.CenterCrop(64),transforms.ToTensor()])
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    if not color:
        n = len(train_data)
        X_train = torch.zeros(n, 1, 64, 64)
        Y_train = train_data.targets
        S_train = Y_train
        for i in range(n):
            X_train[i,0] = transform(train_data.data[i]).squeeze()
        X_train /= X_train.max()
        
        n = len(test_data)
        X_test = torch.zeros(n, 1, 64, 64)
        Y_test = test_data.targets
        S_test = Y_train
        for i in range(n):
            X_test[i,0] = transform(test_data.data[i]).squeeze()
        X_test /= X_test.max()
    else:
        # train
        n = len(train_data)
        X_train = torch.zeros(n, 3, 64, 64)
        S_train = torch.arange(n) % 3
        Y_train = train_data.targets
        for i in range(n):
            X_train[i,S_train[i]] = transform(train_data.data[i]).squeeze()
        X_train /= X_train.max()
        
        # test
        n = len(test_data)
        X_test = torch.zeros(n, 3, 64, 64)
        S_test = torch.arange(n) % 3
        Y_test = test_data.targets
        for i in range(n):
            X_test[i,S_test[i]] = transform(test_data.data[i]).squeeze()
        X_test /= X_test.max()
    
    train_data = VectorDataset(X_train, S_train, Y_train)
    test_data = VectorDataset(X_test, S_test, Y_test)
    
    return train_data, test_data

all_datasets = ['Adult-sex', 'Adult-race', 'Health', 'Bank', 'German', 'Creadit', 'Student', 'Compas-sex', 'Compas-race', 'Adult-fair-sex']

def load_dataset(dataset):
    assert dataset in all_datasets
    if dataset == 'Adult-sex':
        train_data, test_data, D = loadAdult(pro_att='sex')
    elif dataset == 'Adult-race':
        train_data, test_data, D = loadAdult(pro_att='race')
    elif dataset == 'Adult-fair-sex':
        train_data, test_data, D = fairloadAdult(pro_att='sex')
    elif dataset == 'Compas-sex':
        train_data, test_data, D = loadCompas(pro_att='sex')
    elif dataset == 'Compas-race':
        train_data, test_data, D = loadCompas(pro_att='race')
    elif dataset == 'German':
        train_data, test_data, D = loadGerman()
    elif dataset == 'Health':
        train_data, test_data, D = loadHealth()
    elif dataset == 'Bank':
        train_data, test_data, D = loadBank()
    elif dataset == 'Credit':
        train_data, test_data, D = loadCredit()
    elif dataset == 'Student':
        train_data, test_data, D = loadStudent()
    return train_data, test_data, D

def computeMI(c, d, n_neighbors=5):
    n_samples = c.shape[0]
    radius = np.empty(n_samples)
    label_counts = np.empty(n_samples)
    k_all = np.empty(n_samples)
    nn = NearestNeighbors()
    
    for label in np.unique(d):
        mask = d == label
        count = np.sum(mask)
        if count > 1:
            k = min(n_neighbors, count - 1)
            nn.set_params(n_neighbors=k)
            nn.fit(c[mask])
            r = nn.kneighbors()[0]
            radius[mask] = np.nextafter(r[:, -1], 0)
            k_all[mask] = k
        label_counts[mask] = count
    
    # Ignore points with unique labels.
    mask = label_counts > 1
    n_samples = np.sum(mask)
    label_counts = label_counts[mask]
    k_all = k_all[mask]
    c = c[mask]
    radius = radius[mask]
    
    kd = KDTree(c)
    m_all = kd.query_radius(c, radius, count_only=True, return_distance=False)
    m_all = np.array(m_all) - 1.0
    
    mi = (digamma(n_samples)
        + np.mean(digamma(k_all))
        - np.mean(digamma(label_counts))
        - np.mean(digamma(m_all + 1))
        )
    
    return max(0, mi)

eval_col_name = ['yauc', 'sauc', 'dp', 'di', 'ynn', 'zs_mi', 'ys_mi', 'precsn_scr', 'rec_score', 'acc_scr', 'f1_scr', 'auroc']

def evaluate(Z_train, Z_test, S_train, S_test, Y_train, Y_test, cls_name='lr'):
    assert cls_name in {'rf', 'lr'}
    clf = RandomForestClassifier() if cls_name == 'rf' else LogisticRegression()

    # zs_mi
    zs_mi = computeMI(Z_test, S_test)

    # ys_mi
    S, Y = np.concatenate([S_train, S_test]), np.concatenate([Y_train, Y_test])
    ys_mi = mutual_info_classif(S.reshape(-1,1), Y, discrete_features=True)[0]

    # sauc
    clf = clf.fit(Z_train, S_train)
    S_test_hat = clf.predict_proba(Z_test)[:,0]
    sauc = roc_auc_score(S_test, S_test_hat)
    if sauc < 0.5: sauc = roc_auc_score(S_test, 1-S_test_hat)

    # yauc
    clf = clf.fit(Z_train, Y_train)
    Y_test_hat = clf.predict_proba(Z_test)[:,1]
    yauc = roc_auc_score(Y_test, Y_test_hat)
    if yauc < 0.5: yauc = roc_auc_score(Y_test, 1-Y_test_hat)

    # dp
    Y_test_hat = clf.predict(Z_test)
    Y_test_hat_0_mean = Y_test_hat[S_test==0].mean()
    Y_test_hat_1_mean = Y_test_hat[S_test==1].mean()
    dp = abs(Y_test_hat_0_mean - Y_test_hat_1_mean)

    # di
    di = abs(1-((Y_test_hat_1_mean+1e-10)/(Y_test_hat_0_mean+1e-10)))

    # ynn
    knn = NearestNeighbors(n_neighbors=5, n_jobs=-1).fit(Z_test)
    neighbor_idx = knn.kneighbors(Z_test, return_distance=False)
    nbr_test_Y_hat = Y_test_hat[neighbor_idx]
    ynn = np.abs(Y_test_hat.reshape(-1,1) - nbr_test_Y_hat).mean()
    
    # precision score, recall score, accuracy score, f1 score
    
    clf = clf.fit(Z_train, Y_train)
    Y_test_hat = clf.predict_proba(Z_test)[:,1]
    Y_hat=clf.predict(Z_test)
    
    prs_scr = precision_score(Y_test, Y_hat)
    
    rec_scr = recall_score(Y_test, Y_hat)
    
    acc_scr = accuracy_score(Y_test, Y_hat)
    
    f1_scr = f1_score(Y_test, Y_hat)

    auroc = roc_auc_score(Y_test, Y_hat)

    return np.array([yauc, sauc, dp, di, ynn, zs_mi, ys_mi, prs_scr, rec_scr, acc_scr, f1_scr, auroc])

def data_utility_metrics(x_train, x_test, y_train, y_test):

    clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    clf = clf.fit(x_train, y_train)
    Y_test_hat = clf.predict_proba(x_test)[:,1]
    Y_hat=clf.predict(x_test)
    
    prs_scr = precision_score(y_test, Y_hat)
    
    rec_scr = recall_score(y_test, Y_hat)
    
    acc_scr = accuracy_score(y_test, Y_hat)
    
    f1_scr = f1_score(y_test, Y_hat)

    auroc = roc_auc_score(y_test, Y_hat)

    output = {

        'Precision' : prs_scr,
        'Recall' : rec_scr,
        'Accuracy' : acc_scr,
        'F1 Score': f1_scr,
        'Auroc' : auroc
        
    }

    return output

def fairness_metrics(x_train, x_test, y_train, y_test, s):

    clf = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

    clf = clf.fit(x_train, y_train)

    Y_hat=clf.predict(x_test)

    demo_pari = demographic_parity_ratio(y_test, Y_hat, sensitive_features=s)

    eq_odd = equalized_odds_ratio(y_test, Y_hat, sensitive_features=s)

    return {'demo parity ratio' : demo_pari,
           'Equalized Odd': eq_odd}