import torch.nn as nn
from typing import List
from torchsummary import summary

from .extractors import build_extractor
from configs import get_cfg_defaults


class Model(nn.Module):
    def __init__(self, extractor, classifier):
        super(Model, self).__init__()
        self.extractor = extractor
        self.classifier = classifier

    def forward(self, x):
        feature = self.extractor(x)
        categorical_probs = self.classifier(feature)

        return feature, categorical_probs


class Classifier(nn.Module):
    def __init__(self, dims: List = [1280, 512, 9], dropout_prob=0.5):
        super(Classifier, self).__init__()
        self.heads = self.get_heads(dims, dropout_prob)

    @staticmethod
    def get_heads(dims, prob):
        heads = []
        for i in range(1, len(dims) - 1):
            heads.append(nn.Linear(dims[i-1], dims[i]))
            heads.append(nn.Dropout(prob))

        heads.append(nn.Linear(dims[len(dims) - 2], dims[len(dims) - 1]))

        return nn.Sequential(*heads)

    def forward(self, x):
        output = self.heads(x)

        return output


def make_model(cfg, **kwargs):
    extractor = build_extractor(cfg.MODEL.EXTRACTOR.NAME, cfg.MODEL.PRETRAINED)
    classifier = Classifier(cfg.MODEL.HEAD.DIMS, cfg.MODEL.HEAD.DROPOUT_PROB)

    return Model(extractor, classifier)


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    model = make_model(cfg)
    summary(model, (3, 112, 112))
    pass
