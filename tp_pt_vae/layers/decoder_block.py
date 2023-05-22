# -*- coding: utf-8 -*-

# Copyright 2019 Hao-Wen Dong
# Copyright 2023 KateSawada
#  MIT License (https://opensource.org/licenses/MIT)

import torch


class DecoderBlock(torch.nn.Module):
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
        self.transconv = torch.nn.ConvTranspose3d(in_dim, out_dim, kernel, stride)
        self.batchnorm = torch.nn.BatchNorm3d(out_dim)
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
        x = self.transconv(x)
        x = self.batchnorm(x)
        return self.activation(x)
