from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Sequence, Union, Optional


class DatasetWithTransforms(Dataset):

    def __init__(
        self,
        dataset,
        transforms,
        idx_to_apply: Optional[Union[int, Sequence[int]]]=None
    ):
        self.dataset = dataset
        self.transforms = transforms
        self.idx_to_apply = (idx_to_apply if isinstance(idx_to_apply, Sequence) 
                                          else [idx_to_apply])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        element = self.dataset[idx]
        if self.idx_to_apply is None:
            return self.transforms(element)
        if isinstance(element, tuple):
            element = list(element)
        for i in self.idx_to_apply:
            element[i] = self.transforms(element[i])
        return element


class TorchToSklearnClassifierWrapper(BaseEstimator, ClassifierMixin):

    def __init__(self, model):
        self.model = model

    def predict_proba(self, x):
        with torch.no_grad():
            return torch.nn.functional.softmax(self.model(x), dim=-1).numpy()

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=-1)
