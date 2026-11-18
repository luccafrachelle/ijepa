"""Projection head, identical to SPHERE-JEPA lightningmodules/pretrained.py (`self.g`).

torchvision MLP: Linear(in,2048)-BN-GELU-Linear(2048,2048)-BN-GELU-Linear(2048,out),
i.e. num_layers=3, hidden 2048, out 256 by default. (PLAN.md wrote
"192 -> 2048 -> 256"; the official head has two hidden layers and BatchNorm.)
"""
from torch import nn
from torchvision.ops import MLP


def make_projection_head(in_dim=192, proj_hidden_dim=2048, out_dim=256, num_layers=3):
    hidden_channels = [proj_hidden_dim] * (num_layers - 1) + [out_dim]
    return MLP(in_channels=in_dim, hidden_channels=hidden_channels,
               activation_layer=nn.GELU, norm_layer=nn.BatchNorm1d, dropout=0.0)
