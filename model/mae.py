import math
from operator import index, pos
import warnings
from mmcv.runner.checkpoint import _load_checkpoint

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (build_norm_layer, constant_init, kaiming_init,
                      normal_init, trunc_normal_init)
from mmcv.runner import BaseModule, ModuleList
from mmseg.models.backbones.vit import TransformerEncoderLayer
from mmseg.models.utils import PatchEmbed
from mmseg.ops import resize
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple


def build_2d_sincos_position_embedding(patch_size,
                                       embed_dim,
                                       temperature=10000.):
    h, w = patch_size
    grid_w = torch.arange(w, dtype=torch.float32)
    grid_h = torch.arange(h, dtype=torch.float32)
    grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')

    assert embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
    pos_dim = embed_dim // 4
    omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
    omega = 1. / (temperature**omega)
    out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
    out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
    pos_emb = torch.cat([
        torch.sin(out_w),
        torch.cos(out_w),
        torch.sin(out_h),
        torch.cos(out_h)
    ],
                        dim=1)[None, :, :]

    pe_token = torch.zeros([1, 1, embed_dim], dtype=torch.float32)
    pos_embed = torch.cat([pe_token, pos_emb], dim=1)
    return pos_embed


class VisionTransformer(BaseModule):
    """Vision Transformer.

    This backbone is the implementation of `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at
    Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Default: True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """
    def __init__(self,
                 mask_ratio=0.75,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=dict(type='Kaiming',
                               layer='Conv2d',
                               mode='fan_in',
                               nonlinearity='linear')):
        super(VisionTransformer, self).__init__(init_cfg=init_cfg)

        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.mask_ratio = mask_ratio
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dims = embed_dims
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained

        self.patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )

        self.patch_h = (img_size[0] // patch_size)
        self.patch_w = (img_size[1] // patch_size)
        self.num_total = self.patch_h * self.patch_w

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        pos_embed = build_2d_sincos_position_embedding(
            (self.patch_h, self.patch_w), embed_dims)
        self.register_buffer('pos_embed', pos_embed)
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(embed_dims=embed_dims,
                                        num_heads=num_heads,
                                        feedforward_channels=mlp_ratio *
                                        embed_dims,
                                        attn_drop_rate=attn_drop_rate,
                                        drop_rate=drop_rate,
                                        drop_path_rate=dpr[i],
                                        num_fcs=num_fcs,
                                        qkv_bias=qkv_bias,
                                        act_cfg=act_cfg,
                                        norm_cfg=norm_cfg,
                                        batch_first=True))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg,
                                                      embed_dims,
                                                      postfix=1)
            self.add_module(self.norm1_name, norm1)
        self.init_weights()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_num_layers(self):
        return len(self.layers)

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            # logger = get_root_logger()
            checkpoint = _load_checkpoint(self.init_cfg['checkpoint'],
                                          map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            self.load_state_dict(state_dict, True)
        elif self.init_cfg is not None:
            super(VisionTransformer, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            # trunc_normal_init(self.pos_embed, std=.02)
            trunc_normal_init(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            normal_init(m.bias, std=1e-6)
                        else:
                            constant_init(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m.weight, mode='fan_in')
                    if m.bias is not None:
                        constant_init(m.bias, 0)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m.bias, 0)
                    constant_init(m.weight, 1.0)

    def mask_patch(self, x, label=None):
        '''
        Input:
            x: B, 1+L, C
            label: B, L, C
        Output:
            x': B, 1+L', C (L' = (1 - mask_ratio) * L
            pos_embed: B, L, C
            label: B, (L - L'), C
        '''
        # for downstream task
        if self.mask_ratio == 0:
            x = x + self.pos_embed.clone().detach()
            return x, None, None
        # B, 1, C / B, L, C
        B, L, C = x.size()
        pos_embed = self.pos_embed.clone().detach().expand(B, -1, -1)
        x_cls, x_vis = x.split([1, L - 1], dim=1)
        pos_cls, pos_vis = pos_embed.split([1, L - 1], dim=1)

        num_mask = int(self.mask_ratio * (L - 1))
        prob = torch.rand((B, L - 1, 1), requires_grad=False)
        inds = torch.argsort(prob, dim=1).cuda(x.device)
        mask_inds = inds[:, :num_mask]
        unmask_inds = inds[:, num_mask:].expand(-1, -1, C)

        x_vis = torch.gather(x_vis, dim=1, index=unmask_inds)
        pos_unmask = torch.gather(pos_vis, dim=1, index=unmask_inds)
        pos_mask = torch.gather(pos_vis,
                                dim=1,
                                index=mask_inds.expand(-1, -1, C))

        pos_embed = torch.cat([pos_unmask, pos_mask], dim=1)
        x_vis = x_vis + pos_unmask
        x = torch.cat([x_cls, x_vis], dim=1)
        label = torch.gather(label,
                             dim=1,
                             index=mask_inds.expand(-1, -1, label.size(-1)))
        assert (x_vis.size(1) + label.size(1)) == (
            L - 1), 'Wrong. num_unmask={}, num_mask={}.'.format(
                x_vis.size(1), label.size(1))
        return x, pos_embed, label

    def forward(self, inputs, label=None):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)
        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # mask tokens & position embedding
        x, sort_pos_embed, label = self.mask_patch(x, label)

        # Remove class token for transformer encoder input
        if not self.with_cls_token:
            x = x[:, 1:]
        # encoder forward
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x = self.norm1(x)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x[:, 1:]
                else:
                    out = x
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs), sort_pos_embed, label


class MAE(BaseModule):
    """Masked Autoencoders

    This backbone is the implementation of `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at
    Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Default: True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """
    def __init__(self,
                 mask_ratio=0.75,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 encoder_embed_dims=768,
                 encoder_num_layers=12,
                 encoder_num_heads=12,
                 decoder_num_classes=768,
                 decoder_embed_dims=512,
                 decoder_num_layers=8,
                 decoder_num_heads=8,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=True,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=True,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None):
        super().__init__()

        self.num_total = (img_size // patch_size) ** 2
        self.encoder = VisionTransformer(
            mask_ratio=mask_ratio,
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dims=encoder_embed_dims,
            num_layers=encoder_num_layers,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            out_indices=out_indices,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            with_cls_token=with_cls_token,
            output_cls_token=output_cls_token,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            patch_norm=patch_norm,
            final_norm=final_norm,
            interpolate_mode=interpolate_mode,
            num_fcs=num_fcs,
            norm_eval=norm_eval,
            with_cp=with_cp,
            pretrained=pretrained,
        )
        self.encoder_to_decoder = nn.Linear(encoder_embed_dims,
                                            decoder_embed_dims,
                                            bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_embed_dims))
        trunc_normal_init(self.mask_token, std=.02)
        self.decoder = ModuleList()
        # stochastic depth decay rule
        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, decoder_num_heads)
        ]
        for i in range(decoder_num_layers):
            self.decoder.append(
                TransformerEncoderLayer(embed_dims=decoder_embed_dims,
                                        num_heads=decoder_num_heads,
                                        feedforward_channels=mlp_ratio *
                                        decoder_embed_dims,
                                        attn_drop_rate=attn_drop_rate,
                                        drop_rate=drop_rate,
                                        drop_path_rate=dpr[i],
                                        num_fcs=num_fcs,
                                        qkv_bias=qkv_bias,
                                        act_cfg=act_cfg,
                                        norm_cfg=norm_cfg,
                                        batch_first=True))
        self.norm2_name, norm2 = build_norm_layer(norm_cfg,
                                                  decoder_embed_dims,
                                                  postfix=2)
        self.add_module(self.norm2_name, norm2)
        self.head = nn.Linear(decoder_embed_dims, decoder_num_classes)
        self.init_weights()
        # L2 Reconstruction Loss
        self.criterion = nn.MSELoss()

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def init_weights(self):
        # We only implement the 'jax_impl' initialization implemented at
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
        # trunc_normal_init(self.pos_embed, std=.02)
        # trunc_normal_init(self.cls_token, std=.02)
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                trunc_normal_init(m.weight, std=.02)
                if m.bias is not None:
                    if 'ffn' in n:
                        normal_init(m.bias, std=1e-6)
                    else:
                        constant_init(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                kaiming_init(m.weight, mode='fan_in')
                if m.bias is not None:
                    constant_init(m.bias, 0)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                constant_init(m.bias, 0)
                constant_init(m.weight, 1.0)

    def forward(self, x, label):
        # encode
        x, sort_pos_embed, label = self.encoder(x, label)  # [B, 1+L', C_e]
        x = x[-1]
        num_unmask = x.size(1)
        num_mask = self.num_total - num_unmask
        # linear transform
        mask_token = self.mask_token.expand(x.size(0), num_mask, -1)
        x = torch.cat([x, mask_token], dim=1) + sort_pos_embed
        x = self.encoder_to_decoder(x)  # [B, 1+L, C_d]
        # decode
        for blk in self.decoder:
            x = blk(x)
        # only return the mask tokens predict pixels
        x = self.head(self.norm1(x[:, -num_mask:]))
        loss = self.criterion(x, label)

        return loss
