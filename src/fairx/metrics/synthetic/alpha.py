# Original work Copyright 2022, Synthcity Authors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


import platform
from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from synthcity.plugins.core.dataloader import GenericDataLoader

from .utils import *
from .OneClass import OneClassLayer

class SyntheticEvaluation():

    def __init__(self, ori_data, gen_data, use_cache = True):

        super().__init__()

        self._workspace = Path("workspace")

        self._use_cache = use_cache

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self._task_type = 'survival_analysis' 

        self.ori_data = ori_data

        self.gen_data = gen_data

        self.col_list = self.ori_data.data.columns.to_list()

        self.ori_data_frame = self.ori_data.data

        self.gen_data_frame = self.gen_data.data[self.col_list] ## for alligining the two data frame by column

        if len(self.ori_data_frame) > len(self.gen_data_frame):

            self.ori_data_frame_2 = self.ori_data_frame[:len(self.gen_data_frame)]
            self.gen_data_frame_2 = self.gen_data_frame

        else:
            self.gen_data_frame_2 = self.gen_data_frame[:len(self.ori_data_frame)]

            self.ori_data_frame_2 = self.ori_data_frame

        print(self.ori_data_frame_2.shape)

        print(self.gen_data_frame_2.shape)

        self.ori_data_loader = GenericDataLoader(

                self.ori_data_frame_2,
                target_column= self.ori_data.target_attr[0],
                sensitive_columns=self.ori_data.sensitive_attr
                
            )

        self.gen_data_loader = GenericDataLoader(

                self.gen_data_frame_2,
                target_column= self.ori_data.target_attr[0],
                sensitive_columns=self.ori_data.sensitive_attr
                
            )

        self.ori_encoded_data = self.ori_data_loader.encode()[0]

        self.gen_encoded_data = self.gen_data_loader.encode()[0]


    def metrics(
        self,
        X,
        X_syn,
        emb_center,
    ):
        if len(X) != len(X_syn):
            raise RuntimeError("The real and synthetic data must have the same length")
    
        if emb_center is None:
            emb_center = np.mean(X, axis=0)
    
        n_steps = 30
        alphas = np.linspace(0, 1, n_steps)
    
        Radii = np.quantile(np.sqrt(np.sum((X - emb_center) ** 2, axis=1)), alphas)
    
        synth_center = np.mean(X_syn, axis=0)
    
        alpha_precision_curve = []
        beta_coverage_curve = []
    
        synth_to_center = np.sqrt(np.sum((X_syn - emb_center) ** 2, axis=1))
    
        nbrs_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(X)
        real_to_real, _ = nbrs_real.kneighbors(X)
    
        nbrs_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(X_syn)
        real_to_synth, real_to_synth_args = nbrs_synth.kneighbors(X)
    
        # Let us find closest real point to any real point, excluding itself (therefore 1 instead of 0)
        real_to_real = real_to_real[:, 1].squeeze()
        real_to_synth = real_to_synth.squeeze()
        real_to_synth_args = real_to_synth_args.squeeze()
    
        real_synth_closest = X_syn[real_to_synth_args]
    
        real_synth_closest_d = np.sqrt(
            np.sum((real_synth_closest - synth_center) ** 2, axis=1)
        )
        closest_synth_Radii = np.quantile(real_synth_closest_d, alphas)
    
        for k in range(len(Radii)):
            precision_audit_mask = synth_to_center <= Radii[k]
            alpha_precision = np.mean(precision_audit_mask)
    
            beta_coverage = np.mean(
                (
                    (real_to_synth <= real_to_real)
                    * (real_synth_closest_d <= closest_synth_Radii[k])
                )
            )
    
            alpha_precision_curve.append(alpha_precision)
            beta_coverage_curve.append(beta_coverage)
    
        # See which one is bigger
    
        authen = real_to_real[real_to_synth_args] < real_to_synth
        authenticity = np.mean(authen)
    
        Delta_precision_alpha = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(alpha_precision_curve))
        ) / np.sum(alphas)
    
        if Delta_precision_alpha < 0:
            raise RuntimeError("negative value detected for Delta_precision_alpha")
    
        Delta_coverage_beta = 1 - np.sum(
            np.abs(np.array(alphas) - np.array(beta_coverage_curve))
        ) / np.sum(alphas)
    
        if Delta_coverage_beta < 0:
            raise RuntimeError("negative value detected for Delta_coverage_beta")
    
        return (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            Delta_precision_alpha,
            Delta_coverage_beta,
            authenticity,
        )

    

    def get_oneclass_model(self, X_gt):
        
        X_hash = dataframe_hash(pd.DataFrame(X_gt))
    
        cache_file = (
            self._workspace
            / f"sc_metric_cache_model_oneclass_{X_hash}_{platform.python_version()}.bkp"
        )
        
        if cache_file.exists() and self._use_cache:
            return load_from_file(cache_file)
    
        model = OneClassLayer(
            input_dim=X_gt.shape[1],
            rep_dim=X_gt.shape[1],
            center=torch.ones(X_gt.shape[1]) * 10,
        )
        model.fit(X_gt)
    
        save_to_file(cache_file, model)
    
        return model.to(self.device)

    def oneclass_predict(self, model, X):
        
        with torch.no_grad():
            return model(torch.from_numpy(X).float().to(self.device)).cpu().detach().numpy()


    def _normalize_covariates(
            self,
            X,
            X_syn
        ):
            """_normalize_covariates
            This is an internal method to replicate the old, naive method for evaluating
            AlphaPrecision.
    
            Args:
                X (DataLoader): The ground truth dataset.
                X_syn (DataLoader): The synthetic dataset.
    
            Returns:
                Tuple[pd.DataFrame, pd.DataFrame]: normalised version of the datasets
            """
            X_gt_norm = X.dataframe().copy()
            X_syn_norm = X_syn.dataframe().copy()
            if self._task_type != "survival_analysis":
                if hasattr(X, "target_column"):
                    X_gt_norm = X_gt_norm.drop(columns=[X.target_column])
                if hasattr(X_syn, "target_column"):
                    X_syn_norm = X_syn_norm.drop(columns=[X_syn.target_column])
            scaler = MinMaxScaler().fit(X_gt_norm)
            if hasattr(X, "target_column"):
                X_gt_norm_df = pd.DataFrame(
                    scaler.transform(X_gt_norm),
                    columns=[
                        col
                        for col in X.train().dataframe().columns
                        if col != X.target_column
                    ],
                )
            else:
                X_gt_norm_df = pd.DataFrame(
                    scaler.transform(X_gt_norm), columns=X.train().dataframe().columns
                )
    
            if hasattr(X_syn, "target_column"):
                X_syn_norm_df = pd.DataFrame(
                    scaler.transform(X_syn_norm),
                    columns=[
                        col
                        for col in X_syn.dataframe().columns
                        if col != X_syn.target_column
                    ],
                )
            else:
                X_syn_norm_df = pd.DataFrame(
                    scaler.transform(X_syn_norm), columns=X_syn.dataframe().columns
                )
    
            return (X_gt_norm_df, X_syn_norm_df)


    def calculate_alpha_precision(
        self):

        X = self.ori_encoded_data

        X_syn = self.gen_encoded_data
    
        results = {}
    
        X_ = X.numpy().reshape(len(X), -1)
        X_syn_ = X_syn.numpy().reshape(len(X_syn), -1)
    
        # OneClass representation
        emb = "_OC"
        oneclass_model = self.get_oneclass_model(X_)
        X_ = self.oneclass_predict(oneclass_model, X_)
        X_syn_ = self.oneclass_predict(oneclass_model, X_syn_)
        emb_center = oneclass_model.c.detach().cpu().numpy()
    
        (
            alphas,
            alpha_precision_curve,
            beta_coverage_curve,
            Delta_precision_alpha,
            Delta_coverage_beta,
            authenticity,
        ) = self.metrics(X_, X_syn_, emb_center=emb_center)
    
        results[f"Alpha-Precision"] = Delta_precision_alpha
        results[f"Beta-Recall"] = Delta_coverage_beta
        results[f"Authenticity"] = authenticity
    
        # X_df, X_syn_df = self._normalize_covariates(X, X_syn)
        # (
        #     alphas_naive,
        #     alpha_precision_curve_naive,
        #     beta_coverage_curve_naive,
        #     Delta_precision_alpha_naive,
        #     Delta_coverage_beta_naive,
        #     authenticity_naive,
        # ) = self.metrics(X_df.to_numpy(), X_syn_df.to_numpy(), emb_center=None)
    
        # results["delta_precision_alpha_naive"] = Delta_precision_alpha_naive
        # results["delta_coverage_beta_naive"] = Delta_coverage_beta_naive
        # results["authenticity_naive"] = authenticity_naive
    
        return results