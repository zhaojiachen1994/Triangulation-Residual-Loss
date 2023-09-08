# copy from learnable triangulation /pose_resnet.py

import torch.nn as nn
from mmcv.cnn import (constant_init, normal_init)
from mmpose.models import HEADS
import torch.nn as nn
from mmcv.cnn import (constant_init, normal_init)


BN_MOMENTUM = 0.1


@HEADS.register_module()
class GlobalAveragePoolingHead(nn.Module):
    def __init__(self,
                 in_channels,
                 n_classes,
                 ):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 256, 3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.MaxPool2d(2),
            nn.ReLU(inplace=True),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_classes),
            nn.Sigmoid()
            # nn.Tanh()
        )

    def forward(self, x):
        x = self._transform_inputs(x)
        x = self.conv_layers(x)

        batch_size, n_channels = x.shape[:2]
        x = x.view((batch_size, n_channels, -1))
        x = x.mean(dim=-1)

        out = self.fc_layers(x)
        # out = (out + 5) / 10
        return out

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            return inputs
        else:
            inputs = inputs[0]
        return inputs

    def init_weights(self):
        """Initialize model weights."""
        for m in self.conv_layers.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.fc_layers.modules():
            if isinstance(m, nn.Linear):
                normal_init(m, std=0.001, bias=0)
