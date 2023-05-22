# -*- coding: utf-8 -*-
import torch


class KLDivergenceLoss(torch.nn.Module):
    def __init__(self):
        super(KLDivergenceLoss, self).__init__()

    def forward(self, log_var, mu):
        kld =  torch.mean(
            -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(),dim = 1),
            dim = 0)

        return kld

