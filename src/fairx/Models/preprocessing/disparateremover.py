# Copyright 2018-2021 The AI Fairness 360 (AIF360) Authors

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

import numpy as np

from aif360.algorithms import Transformer


class DisparateImpactRemover(Transformer):

    def __init__(self, repair_level=1.0, sensitive_attribute=''):

        super(DisparateImpactRemover, self).__init__(repair_level=repair_level)

        from BlackBoxAuditing.repairers.GeneralRepairer import Repairer
        self.Repairer = Repairer

        if not 0.0 <= repair_level <= 1.0:
            raise ValueError("'repair_level' must be between 0.0 and 1.0.")
        self.repair_level = repair_level

        self.sensitive_attribute = sensitive_attribute

    def fit_transform(self, dataset):
        """Run a repairer on the non-protected features and return the
        transformed dataset.

        Args:
            dataset (BinaryLabelDataset): Dataset that needs repair.
        Returns:
            dataset (BinaryLabelDataset): Transformed Dataset.

        Note:
            In order to transform test data in the same manner as training data,
            the distributions of attributes conditioned on the protected
            attribute must be the same.
        """
        if not self.sensitive_attribute:
            self.sensitive_attribute = dataset.protected_attribute_names[0]

        features = dataset.features.tolist()
        index = dataset.feature_names.index(self.sensitive_attribute)
        repairer = self.Repairer(features, index, self.repair_level, False)

        repaired = dataset.copy()
        repaired_features = repairer.repair(features)
        repaired.features = np.array(repaired_features, dtype=np.float64)
        # protected attribute shouldn't change
        repaired.features[:, index] = repaired.protected_attributes[:, repaired.protected_attribute_names.index(self.sensitive_attribute)]

        return repaired