# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
from mmcv import deprecated_api_warning
from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.cnn.utils.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmcv.runner import BaseModule, ModuleList, Sequential, _load_checkpoint
from mmseg.ops import resize
from mmseg.utils import get_root_logger
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple

from mmseg.models import BACKBONES
from mmseg.models.utils import PatchEmbed
from mmcv.cnn.bricks.drop import build_dropout


class MultiheadAttentionGamma(BaseModule):
    """A wrapper for ``torch.nn.MultiheadAttention``.
    This module implements MultiheadAttention with identity connection,
    and positional encoding  is also passed as input.
    Args:
        embed_dims (int): The embedding dimension.
        num_heads (int): Parallel attention heads.
        attn_drop (float): A Dropout layer on attn_output_weights.
            Default: 0.0.
        proj_drop (float): A Dropout layer after `nn.MultiheadAttention`.
            Default: 0.0.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        batch_first (bool): When it is True,  Key, Query and Value are shape of
            (batch, n, embed_dim), otherwise (n, batch, embed_dim).
             Default to False.
    """
    def __init__(self,
                 embed_dims,
                 num_heads,
                 attn_drop=0.,
                 proj_drop=0.,
                 dropout_layer=dict(type='DropOut', drop_prob=0.),
                 init_cfg=None,
                 batch_first=False,
                 **kwargs):
        super(MultiheadAttentionGamma, self).__init__(init_cfg)
        if 'dropout' in kwargs:
            warnings.warn('The arguments `dropout` in MultiheadAttention '
                          'has been deprecated, now you can separately '
                          'set `attn_drop`(float), proj_drop(float), '
                          'and `dropout_layer`(dict) ')
            attn_drop = kwargs['dropout']
            dropout_layer['drop_prob'] = kwargs.pop('dropout')

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.batch_first = batch_first

        self.attn = nn.MultiheadAttention(embed_dims, num_heads, attn_drop,
                                          **kwargs)

        self.proj_drop = nn.Dropout(proj_drop)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else nn.Identity()

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='MultiheadAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                gamma=None,
                identity=None,
                query_pos=None,
                key_pos=None,
                attn_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `MultiheadAttention`.
        **kwargs allow passing a more general data flow when combining
        with other operations in `transformerlayer`.
        Args:
            query (Tensor): The input query with shape [num_queries, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
                If None, the ``query`` will be used. Defaults to None.
            value (Tensor): The value tensor with same shape as `key`.
                Same in `nn.MultiheadAttention.forward`. Defaults to None.
                If None, the `key` will be used.
            identity (Tensor): This tensor, with the same shape as x,
                will be used for the identity link.
                If None, `x` will be used. Defaults to None.
            query_pos (Tensor): The positional encoding for query, with
                the same shape as `x`. If not None, it will
                be added to `x` before forward function. Defaults to None.
            key_pos (Tensor): The positional encoding for `key`, with the
                same shape as `key`. Defaults to None. If not None, it will
                be added to `key` before forward function. If None, and
                `query_pos` has the same shape as `key`, then `query_pos`
                will be used for `key_pos`. Defaults to None.
            attn_mask (Tensor): ByteTensor mask with shape [num_queries,
                num_keys]. Same in `nn.MultiheadAttention.forward`.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor with shape [bs, num_keys].
                Defaults to None.
        Returns:
            Tensor: forwarded results with shape
                [num_queries, bs, embed_dims]
                if self.batch_first is False, else
                [bs, num_queries embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        # Because the dataflow('key', 'query', 'value') of
        # ``torch.nn.MultiheadAttention`` is (num_query, batch,
        # embed_dims), We should adjust the shape of dataflow from
        # batch_first (batch, num_query, embed_dims) to num_query_first
        # (num_query ,batch, embed_dims), and recover ``attn_output``
        # from num_query_first to batch_first.
        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(query=query,
                        key=key,
                        value=value,
                        attn_mask=attn_mask,
                        key_padding_mask=key_padding_mask)[0]

        if self.batch_first:
            out = out.transpose(0, 1)

        if gamma is not None:
            return identity + self.dropout_layer(gamma * self.proj_drop(out))
        else:
            return identity + self.dropout_layer(self.proj_drop(out))


class FFNGamma(BaseModule):
    """Implements feed-forward networks (FFNs) with identity connection.
    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`. Defaults: 256.
        feedforward_channels (int): The hidden dimension of FFNs.
            Defaults: 1024.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        ffn_drop (float, optional): Probability of an element to be
            zeroed in FFN. Default 0.0.
        add_identity (bool, optional): Whether to add the
            identity connection. Default: `True`.
        dropout_layer (obj:`ConfigDict`): The dropout_layer used
            when adding the shortcut.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """
    @deprecated_api_warning(
        {
            'dropout': 'ffn_drop',
            'add_residual': 'add_identity'
        },
        cls_name='FFN')
    def __init__(self,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None,
                 **kwargs):
        super(FFNGamma, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(Linear(in_channels, feedforward_channels),
                           self.activate, nn.Dropout(ffn_drop)))
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = build_dropout(
            dropout_layer) if dropout_layer else torch.nn.Identity()
        self.add_identity = add_identity

    @deprecated_api_warning({'residual': 'identity'}, cls_name='FFN')
    def forward(self, x, gamma=None, identity=None):
        """Forward function for `FFN`.
        The function would add x to the output tensor if residue is None.
        """
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        if gamma is not None:
            return identity + self.dropout_layer(gamma * out)
        else:
            return identity + self.dropout_layer(out)


class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
    """
    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 init_residual=0.1,
                 window_size=(16, 16),
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True):
        super(TransformerEncoderLayer, self).__init__()

        self.window_size = window_size
        self.norm1_name, norm1 = build_norm_layer(norm_cfg,
                                                  embed_dims,
                                                  postfix=1)
        self.add_module(self.norm1_name, norm1)

        self.attn = MultiheadAttentionGamma(embed_dims=embed_dims,
                                            num_heads=num_heads,
                                            attn_drop=attn_drop_rate,
                                            proj_drop=drop_rate,
                                            dropout_layer=dict(
                                                type='DropPath',
                                                drop_prob=drop_path_rate),
                                            batch_first=batch_first,
                                            bias=qkv_bias,
                                            window_size=window_size)

        self.norm2_name, norm2 = build_norm_layer(norm_cfg,
                                                  embed_dims,
                                                  postfix=2)
        self.add_module(self.norm2_name, norm2)

        self.ffn = FFNGamma(embed_dims=embed_dims,
                            feedforward_channels=feedforward_channels,
                            num_fcs=num_fcs,
                            ffn_drop=drop_rate,
                            dropout_layer=dict(type='DropPath',
                                               drop_prob=drop_path_rate),
                            act_cfg=act_cfg)

        # Relative position embedding
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                        num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # About 2x faster than original impl
        Wh, Ww = self.window_size
        rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer('relative_position_index', rel_position_index)

        if init_residual is not None:
            gamma_1 = init_residual * torch.ones((embed_dims))
            gamma_2 = init_residual * torch.ones((embed_dims))
            self.gamma_1 = nn.Parameter(gamma_1, requires_grad=True)
            self.gamma_2 = nn.Parameter(gamma_2, requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        # Relative Position Embedding
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).unsqueeze(0).contiguous()  # 1, nH, Wh*Ww, Wh*Ww
        # bs, num_heads, Wh*Ww, Wh*Ww -> bs*num_heads, Wh*Ww, Wh*Ww
        relative_position_bias = relative_position_bias.repeat(x.size(0))
        relative_position_bias = relative_position_bias.reshape(
            -1, self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1])

        x = self.attn(self.norm1(x),
                      gamma=self.gamma_1,
                      identity=x,
                      attn_mask=relative_position_bias)
        x = self.ffn(self.norm2(x), gamma=self.gamma_2, identity=x)
        return x


@BACKBONES.register_module()
class MaskedAutoencoder(BaseModule):
    """Masked Autoencoder.

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
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 init_residual=0.1,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super(MaskedAutoencoder, self).__init__(init_cfg=init_cfg)

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

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.init_residual = init_residual
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

        window_size = ((img_size[0] // patch_size),
                       (img_size[1] // patch_size))

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        # self.pos_embed = nn.Parameter(
        #     torch.zeros(1, num_patches + 1, embed_dims))
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
                                        init_residual=self.init_residual,
                                        window_size=window_size,
                                        act_cfg=act_cfg,
                                        norm_cfg=norm_cfg,
                                        batch_first=True))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(norm_cfg,
                                                      embed_dims,
                                                      postfix=1)
            self.add_module(self.norm1_name, norm1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def init_weights(self):
        if (isinstance(self.init_cfg, dict)
                and self.init_cfg.get('type') == 'Pretrained'):
            logger = get_root_logger()
            checkpoint = _load_checkpoint(self.init_cfg['checkpoint'],
                                          logger=logger,
                                          map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint

            # if 'pos_embed' in state_dict.keys():
            #     if self.pos_embed.shape != state_dict['pos_embed'].shape:
            #         logger.info(msg=f'Resize the pos_embed shape from '
            #                     f'{state_dict["pos_embed"].shape} to '
            #                     f'{self.pos_embed.shape}')
            #         h, w = self.img_size
            #         pos_size = int(
            #             math.sqrt(state_dict['pos_embed'].shape[1] - 1))
            #         state_dict['pos_embed'] = self.resize_pos_embed(
            #             state_dict['pos_embed'],
            #             (h // self.patch_size, w // self.patch_size),
            #             (pos_size, pos_size), self.interpolate_mode)

            self.load_state_dict(state_dict, False)
        elif self.init_cfg is not None:
            super(MaskedAutoencoder, self).init_weights()
        else:
            # We only implement the 'jax_impl' initialization implemented at
            # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L353  # noqa: E501
            # trunc_normal_(self.pos_embed, std=.02)
            trunc_normal_(self.cls_token, std=.02)
            for n, m in self.named_modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_(m.weight, std=.02)
                    if m.bias is not None:
                        if 'ffn' in n:
                            nn.init.normal_(m.bias, mean=0., std=1e-6)
                        else:
                            nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Conv2d):
                    kaiming_init(m, mode='fan_in', bias=0.)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm, nn.LayerNorm)):
                    constant_init(m, val=1.0, bias=0.)

    # def _pos_embeding(self, patched_img, hw_shape, pos_embed):
    #     """Positiong embeding method.

    #     Resize the pos_embed, if the input image size doesn't match
    #         the training size.
    #     Args:
    #         patched_img (torch.Tensor): The patched image, it should be
    #             shape of [B, L1, C].
    #         hw_shape (tuple): The downsampled image resolution.
    #         pos_embed (torch.Tensor): The pos_embed weighs, it should be
    #             shape of [B, L2, c].
    #     Return:
    #         torch.Tensor: The pos encoded image feature.
    #     """
    #     assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
    #         'the shapes of patched_img and pos_embed must be [B, L, C]'
    #     x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
    #     if x_len != pos_len:
    #         if pos_len == (self.img_size[0] // self.patch_size) * (
    #                 self.img_size[1] // self.patch_size) + 1:
    #             pos_h = self.img_size[0] // self.patch_size
    #             pos_w = self.img_size[1] // self.patch_size
    #         else:
    #             raise ValueError(
    #                 'Unexpected shape of pos_embed, got {}.'.format(
    #                     pos_embed.shape))
    #         pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
    #                                           (pos_h, pos_w),
    #                                           self.interpolate_mode)
    #     return self.drop_after_pos(patched_img + pos_embed)

    # @staticmethod
    # def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
    #     """Resize pos_embed weights.

    #     Resize pos_embed using bicubic interpolate method.
    #     Args:
    #         pos_embed (torch.Tensor): Position embedding weights.
    #         input_shpae (tuple): Tuple for (downsampled input image height,
    #             downsampled input image width).
    #         pos_shape (tuple): The resolution of downsampled origin training
    #             image.
    #         mode (str): Algorithm used for upsampling:
    #             ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
    #             ``'trilinear'``. Default: ``'nearest'``
    #     Return:
    #         torch.Tensor: The resized pos_embed of shape [B, L_new, C]
    #     """
    #     assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
    #     pos_h, pos_w = pos_shape
    #     cls_token_weight = pos_embed[:, 0]
    #     pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
    #     pos_embed_weight = pos_embed_weight.reshape(
    #         1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
    #     pos_embed_weight = resize(
    #         pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
    #     cls_token_weight = cls_token_weight.unsqueeze(1)
    #     pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
    #     pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
    #     return pos_embed

    def forward(self, inputs):
        B = inputs.shape[0]

        x, hw_shape = self.patch_embed(inputs)

        # stole cls_tokens impl from Phil Wang, thanks
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # x = self._pos_embeding(x, hw_shape, self.pos_embed)

        if not self.with_cls_token:
            # Remove class token for transformer encoder input
            x = x[:, 1:]

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
                B, _, C = out.shape
                out = out.reshape(B, hw_shape[0], hw_shape[1],
                                  C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    out = [out, x[:, 0]]
                outs.append(out)

        return tuple(outs)

    def train(self, mode=True):
        super(MaskedAutoencoder, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()
