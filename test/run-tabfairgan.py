import fairx
from fairx.models.inprocessing import TabFairGAN
from fairx.dataset import BaseDataClass

dataset_name = 'Adult-Income' # Compass or Adult-Income
sensitive_attr = 'sex'

data_module = BaseDataClass(dataset_name, sensitive_attr, True)

under_prev = 'Female'
y_desire = '>50K'

tabfairgan = TabFairGAN(under_prev, y_desire)

tabfairgan.fit(data_module, batch_size = 256, epochs = 5)