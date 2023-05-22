# -*- coding: utf-8 -*-

# Copyright 2019 Hao-Wen Dong
# Copyright 2023 KateSawada
#  MIT License (https://opensource.org/licenses/MIT)

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from tp_pt_vae.layers import DecoderBlock

# A logger for this file
logger = getLogger(__name__)


class TP_PT_Decoder(torch.nn.Module):
    """A convolutional neural network (CNN) based decoder.
    """
    def __init__(
        self,
        d_latent,
        n_tracks,
        n_measures,
        measure_resolution,
        n_pitches,
        ):
        """_summary_

        Args:
            d_latent (int): Dimension of random noise
            n_tracks (int): Number of tracks
            n_measures (int): Number of Measures
            measure_resolution (int): time resolution per measure
            n_pitches (int): number of used pitches
        """
        super().__init__()

        self.d_latent = d_latent
        self.n_tracks = n_tracks
        self.n_measures = n_measures
        self.measure_resolution = measure_resolution
        self.n_pitches = n_pitches

        shared_conv_params = {
            "in_channel": [d_latent, 512, 256],
            "out_channel": [512, 256, 128],
            "kernel": [(4, 1, 1), (1, 4, 3), (1, 4, 3)],
            "stride": [(4, 1, 1), (1, 4, 3), (1, 4, 2)],
        }
        self.shared_conv_network = nn.ModuleList()
        for i in range(len(shared_conv_params["in_channel"])):
            self.shared_conv_network += [
                DecoderBlock(
                    shared_conv_params["in_channel"][i],
                    shared_conv_params["out_channel"][i],
                    shared_conv_params["kernel"][i],
                    shared_conv_params["stride"][i],
                    "relu",
                )
            ]

        # pitch-time private
        pitch_time_params = {
            "in_channel": [128, 32],
            "out_channel": [32, 16],
            "kernel": [(1, 1, 12), (1, 3, 1)],
            "stride": [(1, 1, 12), (1, 3, 1)],
        }
        self.pitch_time_private = nn.ModuleList()
        for i in range(len(pitch_time_params["in_channel"])):
            self.pitch_time_private += [
                nn.ModuleList(
                    [
                        DecoderBlock(
                            pitch_time_params["in_channel"][i],
                            pitch_time_params["out_channel"][i],
                            pitch_time_params["kernel"][i],
                            pitch_time_params["stride"][i],
                            "relu",
                        )
                        for _ in range(self.n_tracks)
                    ]
                )
            ]

        # time-pitch private
        time_pitch_params = {
            "in_channel": [128, 32],
            "out_channel": [32, 16],
            "kernel": [(1, 3, 1), (1, 1, 12)],
            "stride": [(1, 3, 1), (1, 1, 12)],
        }
        self.time_pitch_private = nn.ModuleList()
        for i in range(len(time_pitch_params["in_channel"])):
            self.time_pitch_private += [
                nn.ModuleList(
                    [
                        DecoderBlock(
                            time_pitch_params["in_channel"][i],
                            time_pitch_params["out_channel"][i],
                            time_pitch_params["kernel"][i],
                            time_pitch_params["stride"][i],
                            "relu",
                        )
                        for _ in range(self.n_tracks)
                    ]
                )
            ]

        assert len(self.time_pitch_private) == len(self.pitch_time_private), \
            "pitch-time and time-pitch must have same number of layers."

        # merged private
        private_conv_params = {
            "in_channel": [32],
            "out_channel": [1],
            "kernel": [(1, 1, 1)],
            "stride": [(1, 1, 1)],
        }
        self.private_conv_network = nn.ModuleList()
        # this loop will be not executed because number of private conv is 1
        for i in range(len(private_conv_params["in_channel"]) - 1):
            self.private_conv_network += [
                nn.ModuleList(
                    [
                        DecoderBlock(
                            private_conv_params["in_channel"][i],
                            private_conv_params["out_channel"][i],
                            private_conv_params["kernel"][i],
                            private_conv_params["stride"][i],
                            "relu"
                        )
                        for _ in range(self.n_tracks)
                    ]
                )
            ]

        # final layers use tanh as  activation function
        self.private_conv_network += [
            nn.ModuleList(
                [
                    DecoderBlock(
                        private_conv_params["in_channel"][-1],
                        private_conv_params["out_channel"][-1],
                        private_conv_params["kernel"][-1],
                        private_conv_params["stride"][-1],
                        "sigmoid"
                    )
                    for _ in range(self.n_tracks)
                ]
            )
        ]

    def forward(self, x):
        # if (self.conditioning):
        #     x, condition = x
        #     condition = condition.view(-1, self.conditioning_dim)
        #     shape = list(x.shape)
        #     shape[1] = self.conditioning_dim
        #     condition = condition.expand(shape)
        #     x = torch.cat([x, condition], 1)
        # x = x.view(-1, self.d_latent + self.conditioning_dim, 1, 1, 1)
        x = x.view(-1, self.d_latent, 1, 1, 1)

        # shared
        for i in range(len(self.shared_conv_network)):
            x = self.shared_conv_network[i](x)

        # time-pitch and pitch-time
        pt = [conv(x) for conv in self.pitch_time_private[0]]
        tp = [conv(x) for conv in self.time_pitch_private[0]]

        for i_layer in range(1, len(self.time_pitch_private)):
            pt = [conv(x_) for x_, conv in zip(pt, self.pitch_time_private[i_layer])]
            tp = [conv(x_) for x_, conv in zip(tp, self.time_pitch_private[i_layer])]

        # merge pitch-time and time-pitch
        x = [torch.cat((pt[i], tp[i]), 1) for i in range(self.n_tracks)]

        # merged private
        for i in range(len(self.private_conv_network)):
            x = [conv(x_) for x_, conv in zip(x, self.private_conv_network[i])]

        # reshape
        x = torch.cat(x, 1)

        # x = torch.permute(x, (0, 2, 3 ,4, 1))  # (batch, measures, beat_resolution, pitches, tracks)
        x = x.view(-1, self.n_tracks, self.n_measures,  self.measure_resolution, self.n_pitches)
        return (x,)
