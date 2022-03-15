import torch
from sklearn.metrics import accuracy_score
import numpy as np

from src.utils.logger import setup_logger
logger = setup_logger(__name__)


class Metric(object):
    def name(self):
        return self.__class__.__name__

    def __call__(self, *args, **kwargs):
        pass


class Accuracy(Metric):
    def __init__(self):
        super(Accuracy, self).__init__()

    def __call__(self, pred, labels):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach() if labels.requires_grad else labels
            labels = labels.cpu() if labels.is_cuda else labels
            labels = labels.numpy()

        if isinstance(pred, torch.Tensor):
            pred = pred.detach() if pred.requires_grad else pred
            pred = pred.cpu() if pred.is_cuda else pred
            pred = pred.numpy()
            pred = np.argmax(pred, axis=1)

        return accuracy_score(pred, labels)
