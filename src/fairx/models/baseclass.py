from abc import ABC, abstractmethod


class BaseModelClass(ABC):

    def __init__(self):

        super().__init__()

        pass

    def preprocess_data(self):

        pass

    def fit(self):

        pass

    def hyper_optimize(self):

        pass