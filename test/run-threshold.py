import fairx
from fairx.Models import ThresholdAlgorithm
from fairx.DataLoader import BaseDataClass

dataset_name = 'Compass' # Compass or Adult-Income
sensitive_attr = 'sex'

data_module = BaseDataClass(dataset_name, sensitive_attr, False)

tf_data = data_module.preprocess_data()

splitted_data = data_module.split_data(tf_data)

model = ThresholdAlgorithm()

model.fit(splitted_data)