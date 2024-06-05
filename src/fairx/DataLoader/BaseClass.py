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

class BaseDataClass():

    def __init__(self, dataset_name, sensitive_attr = None, attach_target = False):

        super().__init__()

        self.cat_feat = []
        self.num_feat = []

        self.dataset_name = dataset_name

        if self.dataset_name == 'Adult-Income':

            data_id = 1590

        elif self.dataset_name == 'Credit-card':

            data_id = 42477
            
        elif self.dataset_name == 'Boston':
            
            data_id = 531

        elif self.dataset_name == 'Compass':

            data_id = 42193

        self.raw_data = fetch_openml(
                data_id=data_id,
                cache=True,
                as_frame=True,
                return_X_y=False,
                parser="auto",
            )

        if attach_target:

            self.data = self.raw_data.frame

        else:         
                    
            self.data = self.raw_data.data

        self.sensitive_attr = sensitive_attr

        self.target_attr = self.raw_data.target_names

        for col in self.data.columns:
    
            if(self.data[col].dtype == 'object'):
    
                self.cat_feat.append(col)
        
                self.data[col] = self.data[col].astype('category')

            elif(self.data[col].dtype == 'category'):

                self.cat_feat.append(col)
    
            else:
    
                self.num_feat.append(col)


        if dataset_name == 'Adult-Income':

            self.target = (self.raw_data.target == ">50K") * 1

        elif dataset_name == 'Compass':

            self.target = (self.raw_data.target == "1") * 1
            
        else:

             self.target = self.raw_data.target

        self.sensitive_data = self.data[self.sensitive_attr]

        self.num_data = self.data[self.num_feat].values

        self.cat_data = self.data[self.cat_feat].values

        print(f'Data loading complete')

    def preprocess_data(self):
        
        self.categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        self.numeric_transformer = StandardScaler()

        cat_tf = self.categorical_transformer.fit_transform(self.cat_data)
        
        num_tf = self.numeric_transformer.fit_transform(self.num_data)

        catenated_data = np.hstack((num_tf, cat_tf.toarray()))

        return catenated_data

    def split_data(self, dataset):

        ## `dataset` is concatenated dataset which contains both numerical and categorical feature

        (X_train, X_test, y_train, y_test, A_train, A_test) = train_test_split(
            dataset, self.target, self.sensitive_data, test_size=0.3, random_state=42, stratify=self.target
            )

        return (X_train, X_test, y_train, y_test, A_train, A_test)