# -*- coding: utf-8 -*-

# Copyright 2019 Hao-Wen Dong
# Copyright 2023 KateSawada
#  MIT License (https://opensource.org/licenses/MIT)

import torch


class LayerNorm(torch.nn.Module):
    """An implementation of Layer normalization that does not require size
    information. Copied from https://github.com/pytorch/pytorch/issues/1959."""
    def __init__(self, n_features, eps=1e-5, affine=True):
        super().__init__()
        self.n_features = n_features
        self.affine = affine
        self.eps = eps
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.Tensor(n_features).uniform_())
            self.beta = torch.nn.Parameter(torch.zeros(n_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        mean = x.view(x.size(0), -1).mean(1).view(*shape)
        std = x.view(x.size(0), -1).std(1).view(*shape)
        y = (x - mean) / (std + self.eps)
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            y = self.gamma.view(*shape) * y + self.beta.view(*shape)
        return y


class EncoderBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, kernel, stride, activation="relu"):
        """Generator Block

        Args:
            in_dim (int): input channels
            out_dim (int): output channels
            kernel (array like): kernel size. Must has three elements
            stride (array like): stride. Must has three elements
            activation (str, optional):
                activation function. "relu", "leaky_relu", "sigmoid", "tanh"
                and "x" are available. Defaults to "relu".
        """
        super().__init__()
        self.conv = torch.nn.Conv3d(in_dim, out_dim, kernel, stride)
        self.layernorm = LayerNorm(out_dim)
        if activation == "relu":
            self.activation = torch.nn.functional.relu
        elif activation == "leaky_relu":
            self.activation = torch.nn.functional.leaky_relu
        elif activation == "sigmoid":
            self.activation = torch.sigmoid
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "x":
            self.activation = lambda x: x  # f(x) = x
        else:
            raise ValueError(f"{activation} is not supported")

    def forward(self, x):
        x = self.conv(x)
        x = self.layernorm(x)
        return self.activation(x)
