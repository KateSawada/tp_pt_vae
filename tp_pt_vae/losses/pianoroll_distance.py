# -*- coding: utf-8 -*-

# Copyright 2023 KateSawada
#  MIT License (https://opensource.org/licenses/MIT)

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

class PianorollDistanceLoss(nn.Module):
    def __init__(
        self,
        loss_type="L2"
        ):
        super(PianorollDistanceLoss, self).__init__()

        assert loss_type  in ["l2", "l1", "bce"], f"{loss_type} is not supported."

        if loss_type == "l2":
            self.distance_function = partial(F.mse_loss, reduction="sum")
        elif loss_type == "l1":
            self.distance_function = partial(F.l1_loss, reduction="sum")
        elif loss_type == "bce":
            self.distance_function = partial(F.binary_cross_entropy, reduction="sum")

    def forward(self, x1, x2):
        """reconstruction loss averaged over batch

        Args:
            x1 (Tensor): predicted tensor
            x2 (Tensor): actual tensor

        Returns:
            Tensor: loss
        """
        batch_size = x1.shape[0]
        return self.distance_function(x1, x2) / batch_size
