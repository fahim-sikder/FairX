import fairx
from fairx.models.postprocessing import ThresholdAlgorithm
from fairx.dataset import BaseDataClass
import numpy as np
import random

random.seed(42)
np.random.seed(42)

dataset_name = 'Adult-Income' # Compass or Adult-Income
sensitive_attr = 'sex'

data_module = BaseDataClass(dataset_name, sensitive_attr, False)

model = ThresholdAlgorithm(data_module)

res = model.fit()

print(res)