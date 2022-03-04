import torch
from sklearn.metrics import accuracy_score
import numpy as np

from src.utils.logger import setup_logger
logger = setup_logger(__name__)


class Metric(object):
    def name(self):
        return self.__class__.__name__

    def reset(self):
        pass

    def accumulate(self):
        pass

    def log(self):
        pass

    def get_results(self):
        pass


class AccuracyMetric(Metric):
    def __init__(self):
        super(AccuracyMetric, self).__init__()
        self.reset()

    def reset(self):
        self.results = {'accuracy': []}
        self.eval_result = 0

    def update(self, pred, labels):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach() if labels.requires_grad else labels
            labels = labels.cpu() if labels.is_cuda else labels
            labels = labels.numpy()

        if isinstance(pred, torch.Tensor):
            pred = pred.detach() if pred.requires_grad else pred
            pred = pred.cpu() if pred.is_cuda else pred
            pred = pred.numpy()
            pred = np.argmax(pred, axis=1)

        self.results['accuracy'].append(accuracy_score(pred, labels))

    def accumulate(self):
        self.eval_result = sum(self.results['accuracy']) / len(self.results['accuracy'])

    def get_results(self):
        return self.eval_result
