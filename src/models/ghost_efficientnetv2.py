import math
import numpy as np
import torch
from torch import nn
from torchvision.ops import StochasticDepth
from torchsummary import summary

from ghostnet import GhostModule, SELayer
from .helpers import round_filter, _make_divisible
from src.utils import register_model, create_model


class GhostMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio=6, stride=1, se_ratio=4, dropout=None,
                 shortcut=1, survival=1, epsilon=1e-5):
        super(GhostMBConv, self).__init__()
        self.expand_ratio_filter = round_filter(in_channels * expand_ratio)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.dropout = dropout
        self.use_shortcut = shortcut
        self.se_ratio = se_ratio
        self.survival = survival

        if stride == 2:
            self.average_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if in_channels != out_channels:
            self.shortcut = GhostModule(in_channels, out_channels, kernel_size=1, stride=1)

        if expand_ratio != 1:
            self.ghost1 = GhostModule(in_channels, self.expand_ratio_filter, kernel_size=1, stride=1, ratio=2)
            self.ghost1_bn = nn.BatchNorm2d(num_features=self.expand_ratio_filter, eps=epsilon)
            self.ghost1_activation = nn.ReLU()

        self.depthwise = nn.Conv2d(self.expand_ratio_filter, self.expand_ratio_filter, kernel_size, stride,
                                   padding=1, groups=self.expand_ratio_filter, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(self.expand_ratio_filter, eps=epsilon)
        self.depthwise_activation = nn.ReLU()

        if self.expand_ratio != 1 and (dropout is not None) and (dropout != 0):
            self.dropout = nn.Dropout(dropout)

        if self.se_ratio is not None:
            self.se = SELayer(self.in_channels, self.expand_ratio_filter, se_ratio)

        self.ghost2 = GhostModule(self.expand_ratio_filter, out_channels, kernel_size=1, stride=1, ratio=2)
        self.ghost2_bn = nn.BatchNorm2d(num_features=out_channels, eps=epsilon)
        
    def forward(self, x):
        shortcut = x
        if self.stride == 2:
            shortcut = self.average_pooling(shortcut)
        if self.in_channels != self.out_channels:
            shortcut = self.shortcut(shortcut)
        if self.expand_ratio != 1:  # conv1x1
            x = self.ghost1(x)
            x = self.ghost1_bn(x)
            x = self.ghost1_activation(x)
        # depthwise conv
        x = self.depthwise(x)
        x = self.depthwise_bn(x)
        x = self.depthwise_activation(x)
        # dropout
        if (self.expand_ratio != 1) and (self.dropout is not None) and (self.dropout != 0):
            x = self.dropout(x)

        # se
        if self.se_ratio is not None:
            x = self.se(x)
        # conv1x1
        x = self.ghost2(x)
        x = self.ghost2_bn(x)

        # short and stochastic depth
        if self.use_shortcut:
            if self.survival is not None and self.survival < 1:
                sto_depth = StochasticDepth(p=self.survival, mode='batch')

                return sto_depth(x) + shortcut
            else:
                return torch.add(shortcut, x)
        else:
            return x


class GhostFusedMBConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, expand_ratio=6, stride=1, se_ratio=4, dropout=None,
                 shortcut=1, survival=None, epsilon=1e-5):
        super(GhostFusedMBConv, self).__init__()
        self.expand_ratio_filter = round_filter(in_channels * expand_ratio)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.drop = dropout
        self.se_ratio = se_ratio
        self.use_shortcut = shortcut
        self.survival = survival

        if stride == 2:
            self.average_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if in_channels != out_channels:
            self.shortcut = GhostModule(in_channels, out_channels, kernel_size=1, stride=1)

        if expand_ratio != 1:
            self.ghost1 = GhostModule(in_channels, self.expand_ratio_filter, kernel_size=kernel_size,
                                      stride=stride, ratio=2)
            self.ghost1_bn = torch.nn.BatchNorm2d(self.expand_ratio_filter)
            self.ghost1_act = torch.nn.ReLU()
            if (dropout is not None) and (dropout != 0):
                self.ghost1_dropout = torch.nn.Dropout(p=dropout)

        # SE
        if se_ratio is not None:
            self.se = SELayer(self.in_channels, self.expand_ratio_filter, se_ratio)

        self.ghost2 = GhostModule(in_channels=self.expand_ratio_filter, out_channels=out_channels,
                                  kernel_size=1 if expand_ratio != 1 else kernel_size,
                                  stride=1 if expand_ratio != 1 else stride)
        self.out_bn = nn.BatchNorm2d(out_channels, eps=epsilon)

    def forward(self, x):
        shortcut = x
        if self.stride == 2:
            shortcut = self.average_pooling(shortcut)
        if self.in_channels != self.out_channels:
            shortcut = self.shortcut(shortcut)

        if self.expand_ratio != 1:
            x = self.ghost1(x)
            x = self.ghost1_bn(x)
            x = self.ghost1_act(x)
            if (self.drop is not None) and (self.drop != 0):
                x = self.ghost1_dropout(x)

        # SE
        if self.se_ratio is not None:
            x = self.se(x)

        x = self.ghost2(x)
        x = self.out_bn(x)

        if self.use_shortcut:
            if self.survival is not None and self.survival < 1:
                sto_depth = StochasticDepth(p=self.survival, mode='batch')

                return sto_depth(x) + shortcut
            else:
                return torch.add(x, shortcut)
        else:
            return x


class GhostEfficientnetV2(nn.Module):
    def __init__(self, cfg, num_classes, width_mult, depth_mult, conv_dropout_rate=None,
                 dropout_rate=None, drop_connect=None, epsilon=1e-5):
        super(GhostEfficientnetV2, self).__init__()
        self.dropout_rate = dropout_rate
        # stage 0
        self.stage0 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24, eps=epsilon),
            nn.ReLU()
        )

        self.stage1to6 = []
        for stage in cfg:
            count = int(math.ceil(stage[0] * depth_mult))
            for j in range(count):
                self.stage1to6.append(self.handle_stage(index=j, in_channels=round_filter(stage[4], width_mult),
                                                        out_channels=round_filter(stage[5], width_mult),
                                                        kernel_size=stage[1], expand_ratio=stage[3],
                                                        use_fused=stage[6], stride=stage[2], se_ratio=stage[7],
                                                        dropout=conv_dropout_rate, drop_connect=drop_connect,
                                                        shortcut=stage[8], survival=stage[9]))
        self.stage1to6 = nn.Sequential(*self.stage1to6)
        self.stage7_conv = nn.Sequential(
            nn.Conv2d(256, round_filter(1280, width_mult), kernel_size=1, bias=False),
            nn.BatchNorm2d(round_filter(1280, width_mult), eps=epsilon),
            nn.ReLU()
        )
        self.stage7_global_avg_pooling = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        if (self.dropout_rate is not None) and (self.dropout_rate != 0):
            self.stage7_drop = nn.Dropout(dropout_rate)
        self.stage7_classifier = nn.Linear(round_filter(1280, width_mult), num_classes)

    def forward(self, x):
        x = self.stage0(x)
        # x = self.get_grim_top_right(x)
        x = self.stage1to6(x)
        feats = self.stage7_global_avg_pooling(self.stage7_conv(x))

        if (self.dropout_rate is not None) and (self.dropout_rate != 0):
            feats = self.stage7_drop(feats)

        feats = torch.squeeze(feats)
        out = self.stage7_classifier(feats)

        return out

    @staticmethod
    def handle_stage(index, in_channels, out_channels, kernel_size, expand_ratio, use_fused,
                     stride=1, se_ratio=None, dropout=None, drop_connect=0.2, shortcut=1, survival=None):
        if use_fused:
            return GhostFusedMBConv(in_channels=out_channels if index != 0 else in_channels,
                                    out_channels=out_channels, kernel_size=kernel_size,
                                    stride=1 if index != 0 else stride,
                                    expand_ratio=expand_ratio, se_ratio=se_ratio, dropout=dropout,
                                    shortcut=shortcut, survival=survival)

        else:
            return GhostMBConv(in_channels=out_channels if index != 0 else in_channels,
                               out_channels=out_channels, kernel_size=kernel_size,
                               stride=1 if index != 0 else stride,
                               expand_ratio=expand_ratio, se_ratio=se_ratio, dropout=dropout,
                               shortcut=shortcut, survival=survival)

    @staticmethod
    def get_grim_top_right(matrix):
        batch_size = matrix.shape[0]
        matrix = np.array(matrix)
        features = []
        for index in range(batch_size):
            gram_matrix = np.dot(matrix[index].T, matrix[index])
            feature = []
            gram_len = gram_matrix.shape[1]
            for row in range(gram_len):
                tem_row = []
                for clo in range(gram_len):
                    clos = clo + row
                    if clos > gram_len - 1:
                        break
                    tem_row.append(gram_matrix[row][row + clo])
                feature.append(tem_row)
            features.append(feature)
        return np.array(features)


@register_model
def ghost_efficientnet_v2_s(**kwargs):
    cfg = [
        [2, 3, 1, 1, 24, 24, True, None, 1, 0.5],  # stage 1
        [4, 3, 2, 4, 24, 48, True, None, 1, 0.5],  # stage 2
        [4, 3, 2, 4, 48, 64, True, None, 1, 0.5],  # stage 3
        [6, 3, 2, 4, 64, 128, False, 4, 1, 0.5],  # stage 4
        [9, 3, 1, 6, 128, 160, False, 4, 1, 0.5],  # stage 5
        [15, 3, 2, 6, 160, 256, False, 4, 1, 0.5],  # stage 6
    ]

    return GhostEfficientnetV2(cfg, num_classes=kwargs['num_classes'], width_mult=kwargs['width_mult'],
                               depth_mult=kwargs['depth_mult'],
                               conv_dropout_rate=kwargs['conv_dropout_rate'],
                               dropout_rate=kwargs['dropout_rate'], drop_connect=kwargs['drop_connect'])


if __name__ == '__main__':
    model = create_model('ghost_efficientnet_v2_s', num_classes=9, width_mult=1.0, depth_mult=1.0,
                         conv_dropout_rate=0.2, dropout_rate=0.2, drop_connect=None)
    print(model)
    pass
