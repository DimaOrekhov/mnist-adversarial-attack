from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import torch
import numpy as np


class TorchToSklearnClassifierWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, model):
        self.model = model

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.nn.functional.softmax(self.model(x), dim=-1).numpy()

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=-1)

