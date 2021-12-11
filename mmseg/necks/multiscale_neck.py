# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.cnn.bricks.activation import build_activation_layer
import torch.nn as nn
from mmcv.cnn import build_norm_layer, xavier_init
from mmseg.models.builder import NECKS


@NECKS.register_module()
class MultiScaleNeck(nn.Module):
    """MultiScaleNeck.

    A neck structure connect vit backbone and decoder_heads.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        scales (List[float]): Scale factors for each input feature map.
            Default: [0.5, 1, 2, 4]
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 patch_size=16,
                 norm_cfg=None,
                 act_cfg=dict(type='GELU')):
        super(MultiScaleNeck, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(in_channels[0],
                                   out_channels,
                                   kernel_size=2,
                                   stride=2),
                build_norm_layer(norm_cfg, out_channels)[0],
                build_activation_layer(act_cfg),
                nn.ConvTranspose2d(out_channels,
                                   out_channels,
                                   kernel_size=2,
                                   stride=2))

            self.fpn2 = nn.ConvTranspose2d(in_channels[1],
                                           out_channels,
                                           kernel_size=2,
                                           stride=2)

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.ConvTranspose2d(in_channels[0],
                                           out_channels,
                                           kernel_size=2,
                                           stride=2)

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.MaxPool2d(kernel_size=2, stride=2)

            self.fpn4 = nn.MaxPool2d(kernel_size=4, stride=4)

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        outputs = []
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
        for i in range(len(inputs)):
            outputs.append(ops[i](inputs[i]))

        return tuple(outputs)
