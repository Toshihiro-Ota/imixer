"""
Copyright (c) 2023. Toshihiro Ota
Licensed under the Apache License, Version 2.0 (the "License");

iMixer in PyTorch
This impl is based on `mlp_mixer` provided in timm by Ross Wightman.

Hacked together by / Copyright 2021 Ross Wightman
"""
import math
from functools import partial

import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg, named_apply, checkpoint_seq
from timm.models.layers import PatchEmbed, Mlp, DropPath, lecun_normal_, to_2tuple
from timm.models.registry import register_model

from utils.spectral_norm_fc import spectral_norm_fc


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': 0.875, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        'first_conv': 'stem.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = dict(
    mixer_s32_224=_cfg(),
    mixer_s16_224=_cfg(),
    mixer_b32_224=_cfg(),
    mixer_b16_224=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth',
    ),
    mixer_l32_224=_cfg(),
    mixer_l16_224=_cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_l16_224-92f9adc4.pth',
    ),
)


class iMlp(nn.Module):
    """ MLP with inverted residual connection
    using fixed-point iteration method and power-iteration spectral normalization
    """
    def __init__(self, in_features, internal_features=None, hidden_features=None, out_features=None,
                act_layer=nn.GELU, drop=0.,
                coeff=0.9, n_power_iterations=None, n_repeat=None): # for spectral normalization and fixed-point iteration
        super().__init__()
        out_features = out_features or in_features
        internal_features = internal_features or hidden_features
        hidden_features = hidden_features or internal_features
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, internal_features)
        self.fc_sn1 = spectral_norm_fc(nn.Linear(internal_features, hidden_features),
                                        coeff, n_power_iterations=n_power_iterations)
        self.fc_sn2 = spectral_norm_fc(nn.Linear(hidden_features, internal_features),
                                        coeff, n_power_iterations=n_power_iterations)
        self.fc2 = nn.Linear(internal_features, out_features)
        self.act = act_layer()

        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[1])

        self.n_repeat = n_repeat

    def forward(self, x):
        x = self.fc1(x)
        x_in = x
        for _ in range(self.n_repeat):
            x = self.act(x)
            x = self.fc_sn1(x)
            x = self.act(x)
            x = self.fc_sn2(x)
            x = x + x_in
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token-mixing and channel-mixing MLPs
    token-mixing blocks compose of iMLP modules defined above
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0), mlp_layer=Mlp, i_mlp_layer=iMlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), act_layer=nn.GELU, drop=0., drop_path=0.,
            hidden_ratio=1.0, n_power_iterations=8, n_repeat=4):
        super().__init__()
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.norm1 = norm_layer(dim)
        self.mlp_tokens = i_mlp_layer(seq_len, internal_features=int(tokens_dim),
                                        hidden_features=int(tokens_dim*hidden_ratio),
                                        act_layer=act_layer, drop=drop,
                                        n_power_iterations=n_power_iterations, n_repeat=n_repeat)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp_channels = mlp_layer(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.mlp_tokens(self.norm1(x).transpose(1, 2)).transpose(1, 2))
        x = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class MlpMixer(nn.Module):

    def __init__(
            self,
            num_classes=1000,
            img_size=224,
            in_chans=3,
            patch_size=16,
            num_blocks=8,
            embed_dim=512,
            mlp_ratio=(0.5, 4.0),
            block_layer=MixerBlock,
            mlp_layer=Mlp,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            act_layer=nn.GELU,
            drop_rate=0.,
            drop_path_rate=0.,
            nlhb=False,
            stem_norm=False,
            global_pool='avg',
            hidden_ratio=1.0,
            n_power_iterations=8,
            n_repeat=4,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.grad_checkpointing = False

        self.stem = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, norm_layer=norm_layer if stem_norm else None)
        # FIXME drop_path (stochastic depth scaling rule or all the same?)
        self.blocks = nn.Sequential(*[
            block_layer(
                embed_dim, self.stem.num_patches, mlp_ratio, mlp_layer=mlp_layer, norm_layer=norm_layer,
                act_layer=act_layer, drop=drop_rate, drop_path=drop_path_rate,
                hidden_ratio=hidden_ratio, n_power_iterations=n_power_iterations, n_repeat=n_repeat)
            for _ in range(num_blocks)])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.init_weights(nlhb=nlhb)

    @torch.jit.ignore
    def init_weights(self, nlhb=False):
        head_bias = -math.log(self.num_classes) if nlhb else 0.
        named_apply(partial(_init_weights, head_bias=head_bias), module=self)  # depth-first

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^stem',  # stem and embed
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.stem(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.global_pool == 'avg':
            x = x.mean(dim=1)
        x = self.head(x)
        return x


def _init_weights(module: nn.Module, name: str, head_bias: float = 0., flax=False):
    """ Mixer weight initialization (trying to match Flax defaults)
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            if flax:
                # Flax defaults
                lecun_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            else:
                # like MLP init in vit (my original init)
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        # NOTE if a parent module contains init_weights method, it can override the init of the
        # child modules as this will be called in depth-first order.
        module.init_weights()


def checkpoint_filter_fn(state_dict, model):
    """ Remap checkpoints if needed """
    if 'patch_embed.proj.weight' in state_dict:
        # Remap FB ResMlp models -> timm
        out_dict = {}
        for k, v in state_dict.items():
            k = k.replace('patch_embed.', 'stem.')
            k = k.replace('attn.', 'linear_tokens.')
            k = k.replace('mlp.', 'mlp_channels.')
            k = k.replace('gamma_', 'ls')
            if k.endswith('.alpha') or k.endswith('.beta'):
                v = v.reshape(1, 1, -1)
            out_dict[k] = v
        return out_dict
    return state_dict


def _create_mixer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for MLP-Mixer models.')

    model = build_model_with_cfg(
        MlpMixer, variant, pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        **kwargs)
    return model


# iMixer models
@register_model
def imixer_s32_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=32, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer('mixer_s32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def imixer_s16_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    model = _create_mixer('mixer_s16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def imixer_b32_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=32, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def imixer_b16_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    model = _create_mixer('mixer_b16_224', pretrained=pretrained, **model_args)
    return model


@register_model
def imixer_l32_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=32, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l32_224', pretrained=pretrained, **model_args)
    return model


@register_model
def imixer_l16_224(pretrained=False, **kwargs):
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    model = _create_mixer('mixer_l16_224', pretrained=pretrained, **model_args)
    return model
