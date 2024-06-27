import numpy as np
import pandas as pd

from collections import OrderedDict
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.datasets import fetch_openml

from fairx.utils import setSeed

import pathlib

import warnings
warnings.filterwarnings('ignore')


setSeed(2022)

class BaseDataClass():
    """
    Dataset loader.
    """

    def __init__(self, dataset_name, sensitive_attr = None, target_feature = None, attach_target = False):

        """
        Input: dataset_name: string (currently options are: 'Adult-Income', 'Student-performance', 'Predict-diagnosis', 'Compass')
                target_feature: list, name of the target feature
                sensitive_attr: string, name of the protected attribute
                attach_target: Boolean, if True, target is attached with the main dataframe.

        """

        super().__init__()

        self.attach_target = attach_target

        self.cat_feat = []
        self.num_feat = []

        self.target_feature = target_feature

        self.dataset_name = dataset_name

        if self.dataset_name == 'Adult-Income':

            data_id = 1590

        elif self.dataset_name == 'Compass':

            data_id = 42193

        elif self.dataset_name == 'Student-performance':

            data_id = 42351

        elif self.dataset_name == 'Predict-diagnosis':

            data_id = 45040

        self.raw_data = fetch_openml(
                data_id=data_id,
                cache=True,
                as_frame=True,
                return_X_y=False,
                parser="auto",
            )

        if self.attach_target:

            self.tmp_data = self.raw_data.frame

        else:         
                    
            self.tmp_data = self.raw_data.data

        self.data = self.tmp_data.copy()

        self.simple_imputer = SimpleImputer(strategy='most_frequent')
        
        for col in self.data.columns:
            
            self.data[[col]] = self.simple_imputer.fit_transform(self.data[[col]])

        self.sensitive_attr = sensitive_attr

        if target_feature is None:

            self.target_attr = self.raw_data.target_names

        else:

            self.target_attr = self.target_feature

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

            self.target = self.target.to_numpy()

        elif dataset_name == 'Compass' or dataset_name == 'Predict-diagnosis':

            self.target = (self.raw_data.target == "1") * 1

            self.target = self.target.to_numpy()
            
        else:

             self.target = self.raw_data.target

        self.sensitive_data = self.data[self.sensitive_attr].to_numpy()

        self.num_data = self.data[self.num_feat].values

        self.cat_data = self.data[self.cat_feat].values

        print(f'Data loading complete')

        print(f'Target attribute: {self.target_attr[0]}')


    def preprocess_data(self):
        
        self.categorical_transformer = OneHotEncoder(handle_unknown="ignore")

        self.numeric_transformer = StandardScaler()

        cat_tf = self.categorical_transformer.fit_transform(self.cat_data)
        
        num_tf = self.numeric_transformer.fit_transform(self.num_data)

        self.cat_lens = [i.shape[0] for i in self.categorical_transformer.categories_]

        self.discrete_columns_ordereddict = OrderedDict(zip(self.cat_feat, self.cat_lens))

        self.catenated_data = np.hstack((num_tf, cat_tf.toarray()))

        self.sensitive_start_index = len(self.num_feat) + sum(
            list(self.discrete_columns_ordereddict.values())[:list(self.discrete_columns_ordereddict.keys()).index(self.sensitive_attr)])

        self.len_sensitive_attr_ = self.discrete_columns_ordereddict[list(self.discrete_columns_ordereddict.keys())[list(self.discrete_columns_ordereddict).index(self.sensitive_attr)]]

        ## debugging
        
        # print(self.sensitive_start_index)

        # print(self.len_sensitive_attr_)

        # print(self.discrete_columns_ordereddict)

        ##

        return self.categorical_transformer, self.numeric_transformer, self.catenated_data

    def split_data(self, dataset):

        """
        Split the dataset, using Sklearn's train_test_split function. Returns the splitted dataset as a tuple
        """

        ## `dataset` is concatenated dataset which contains both numerical and categorical feature

        (X_train, X_test, y_train, y_test, A_train, A_test) = train_test_split(
            dataset, self.target, self.sensitive_data, test_size=0.3, random_state=42, stratify=self.target
            )

        return (X_train, X_test, y_train, y_test, A_train, A_test)