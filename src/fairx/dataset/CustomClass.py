
# from .BaseClass import BaseDataClass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_openml

import pathlib

import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder

class CustomDataClass():

    def __init__(self, dataset_name, sensitive_attr, target_attr, columns = None, attach_target = False):

        super().__init__()

        self.dataset_name = 'custom'

        self.data_path = dataset_name

        self.sensitive_attr = sensitive_attr

        self.target_attr = target_attr

        self.columns = columns

        self.labelencoder = LabelEncoder()

        if self.columns is None:

            self.tmp_data = pd.read_csv(self.data_path)

        else:

            self.tmp_data = pd.read_csv(self.data_path, names = self.columns)

        self.data = self.tmp_data.copy()

        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        
        for col in self.data.columns:
            
            self.data[[col]] = self.simple_imputer.fit_transform(self.data[[col]])

        self.frame_data = self.data.copy()

        self.target = self.labelencoder.fit_transform(self.data[self.target_attr])

        if not attach_target:

            self.data = self.data.drop(self.target_attr, axis = 1)

        self.cat_feat = []
        self.num_feat = []


        for col in self.data.columns:
    
            if(self.data[col].dtype == 'object'):
    
                self.cat_feat.append(col)
        
                self.data[col] = self.data[col].astype('category')

            elif(self.data[col].dtype == 'category'):

                self.cat_feat.append(col)
    
            else:
    
                self.num_feat.append(col)

        self.sensitive_data = self.data[self.sensitive_attr].values

        self.num_data = self.data[self.num_feat].values

        self.cat_data = self.data[self.cat_feat].values

        print(f'Data loading complete')

    def preprocess_data(self):

        self.categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        self.numeric_transformer = StandardScaler()

        cat_tf = self.categorical_transformer.fit_transform(self.cat_data)
        
        num_tf = self.numeric_transformer.fit_transform(self.num_data)

        self.catenated_data = np.hstack((num_tf, cat_tf.toarray()))

        return self.categorical_transformer, self.numeric_transformer, self.catenated_data
        
    def split_data(self, dataset):

        (X_train, X_test, y_train, y_test, A_train, A_test) = train_test_split(
            dataset, self.target, self.sensitive_data, test_size=0.3, random_state=42, stratify=self.target
            )

        return (X_train, X_test, y_train, y_test, A_train, A_test)