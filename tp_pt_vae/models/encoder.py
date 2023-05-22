# -*- coding: utf-8 -*-

# Copyright 2019 Hao-Wen Dong
# Copyright 2023 KateSawada
#  MIT License (https://opensource.org/licenses/MIT)

from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F

from tp_pt_vae.layers import EncoderBlock


# A logger for this file
logger = getLogger(__name__)


class TP_PT_Encoder(nn.Module):
    """A convolutional neural network (CNN) based Encoder.
    """
    def __init__(
        self,
        n_tracks,
        n_measures,
        measure_resolution,
        n_beats,
        n_pitches,
        ):
        """_summary_

        Args:
            n_tracks (int): Number of tracks
            n_measures (int): Number of Measures
            measure_resolution (int): time resolution per measure
            n_beats (int): number of beats per measure
            n_pitches (int): number of used pitches
        """
        super().__init__()

        self.n_tracks = n_tracks
        self.n_measures = n_measures
        self.measure_resolution = measure_resolution
        self.n_beats = n_beats
        self.n_pitches = n_pitches

        # pitch-time private
        pitch_time_params = {
            "in_channel": [1, 16],
            "out_channel": [16, 32],
            "kernel": [(1, 1, 12), (1, 3, 1)],
            "stride": [(1, 1, 12), (1, 3, 1)],
        }
        self.pitch_time_private = nn.ModuleList()
        for i in range(len(pitch_time_params["in_channel"])):
            self.pitch_time_private += [
                nn.ModuleList(
                    [
                        EncoderBlock(
                            pitch_time_params["in_channel"][i],
                            pitch_time_params["out_channel"][i],
                            pitch_time_params["kernel"][i],
                            pitch_time_params["stride"][i],
                            "leaky_relu",
                        )
                        for _ in range(self.n_tracks)
                    ]
                )
            ]

        # time-pitch private
        time_pitch_params = {
            "in_channel": [1, 16],
            "out_channel": [16, 32],
            "kernel": [(1, 3, 1), (1, 1, 12)],
            "stride": [(1, 3, 1), (1, 1, 12)],
        }
        self.time_pitch_private = nn.ModuleList()
        for i in range(len(time_pitch_params["in_channel"])):
            self.time_pitch_private += [
                nn.ModuleList(
                    [
                        EncoderBlock(
                            time_pitch_params["in_channel"][i],
                            time_pitch_params["out_channel"][i],
                            time_pitch_params["kernel"][i],
                            time_pitch_params["stride"][i],
                            "leaky_relu",
                        )
                        for _ in range(self.n_tracks)
                    ]
                )
            ]

        # merged private
        merged_private_conv_params = {
            "in_channel": [64],
            "out_channel": [64],
            "kernel": [(1, 1, 1), (1, 1, 1)],
            "stride": [(1, 1, 1), (1, 1, 1)],
        }

        self.merged_private_conv_network = nn.ModuleList()
        for i in range(len(merged_private_conv_params["in_channel"])):
            self.merged_private_conv_network += [
                nn.ModuleList(
                    [
                        EncoderBlock(
                            merged_private_conv_params["in_channel"][i],
                            merged_private_conv_params["out_channel"][i],
                            merged_private_conv_params["kernel"][i],
                            merged_private_conv_params["stride"][i],
                            "leaky_relu",
                        )
                        for _ in range(self.n_tracks)
                    ]
                )
            ]

        # shared
        self.shared_conv_network = nn.ModuleList()
        shared_conv_params = {
            "in_channel": [64 * n_tracks, 128],
            "out_channel": [128, 256],
            "kernel": [(1, 4, 3), (1, 4, 3),],
            "stride": [(1, 4, 2), (1, 4, 2),],
        }

        for i in range(len(shared_conv_params["in_channel"])):
            self.shared_conv_network += [
                EncoderBlock(
                    shared_conv_params["in_channel"][i],
                    shared_conv_params["out_channel"][i],
                    shared_conv_params["kernel"][i],
                    shared_conv_params["stride"][i],
                    "leaky_relu",
                )
            ]

        # chroma
        self.chroma_conv_network = nn.ModuleList()
        chroma_conv_params = {
            "in_channel": [n_tracks, 32],
            "out_channel": [32, 64],
            "kernel": [(1, 1, 12), (1, 4, 1),],
            "stride": [(1, 1, 12), (1, 4, 1),],
        }

        for i in range(len(chroma_conv_params["in_channel"])):
            self.chroma_conv_network += [
                EncoderBlock(
                    chroma_conv_params["in_channel"][i],
                    chroma_conv_params["out_channel"][i],
                    chroma_conv_params["kernel"][i],
                    chroma_conv_params["stride"][i],
                    "leaky_relu",
                )
            ]

        # onset/offset
        self.of_off_conv_network = nn.ModuleList()
        of_off_conv_params = {
            "in_channel": [n_tracks, 16, 32],
            "out_channel": [16, 32, 64],
            "kernel": [(1, 3, 1), (1, 4, 1), (1, 4, 1),],
            "stride": [(1, 3, 1), (1, 4, 1), (1, 4, 1),],
        }

        for i in range(len(of_off_conv_params["in_channel"])):
            self.of_off_conv_network += [
                EncoderBlock(
                    of_off_conv_params["in_channel"][i],
                    of_off_conv_params["out_channel"][i],
                    of_off_conv_params["kernel"][i],
                    of_off_conv_params["stride"][i],
                    "leaky_relu",
                )
            ]

        # all merge
        self.all_merge_conv_network = nn.ModuleList()
        all_merge_conv_params = {
            "in_channel": [384,],
            "out_channel": [512,],
            "kernel": [(2, 1, 1),],
            "stride": [(1, 1, 1),],
        }

        for i in range(len(all_merge_conv_params["in_channel"])):
            self.all_merge_conv_network += [
                EncoderBlock(
                    all_merge_conv_params["in_channel"][i],
                    all_merge_conv_params["out_channel"][i],
                    all_merge_conv_params["kernel"][i],
                    all_merge_conv_params["stride"][i],
                    "leaky_relu",
                )
            ]


    def forward(self, x):
        x = x.view(-1, self.n_tracks, self.n_measures, self.measure_resolution, self.n_pitches)
        # (batch, 5, 4, 48, 84)

        # chroma feature
        reshaped = torch.reshape(x, (-1, self.n_tracks, self.n_measures, self.n_beats, self.measure_resolution // self.n_beats, self.n_pitches))
        # (batch, 5, 4, 4, 12, 84)
        summed = torch.sum(reshaped, 4)
        # (batch, 5, 4, 4, 84): (batch, n_tracks, n_measure, n_beats, n_pitches)
        factor = self.n_pitches // 12
        remainder = self.n_pitches % 12
        reshaped = torch.reshape(summed[..., :factor * 12], (-1, self.n_tracks, self.n_measures, self.n_beats, factor, 12))
        chroma = torch.sum(reshaped, 4)
        # (batch, 5, 4, 4, 12): (batch, n_tracks, n_measure, n_beats, 12)
        if remainder != 0:
            chroma[..., -remainder:] += summed[..., -remainder:]

        # onset/offset
        padded = torch.nn.functional.pad(x[..., :-1, :], (0, 0, 1, 0), "constant", 0)
        on_off_set = torch.sum(x - padded, 4, keepdim=True)

        x = [x[:, i].view(
                -1,
                1,
                self.n_measures,
                self.measure_resolution,
                self.n_pitches)
                for i in range(self.n_tracks)]

        pt = [conv(x_) for x_, conv in zip(x, self.pitch_time_private[0])]
        tp = [conv(x_) for x_, conv in zip(x, self.time_pitch_private[0])]
        for i_layer in range(1, len(self.time_pitch_private)):
            pt = [conv(x_) for x_, conv in zip(pt, self.pitch_time_private[i_layer])]
            tp = [conv(x_) for x_, conv in zip(tp, self.time_pitch_private[i_layer])]

        x = [torch.cat((pt[i], tp[i]), 1) for i in range(self.n_tracks)]
        for track in range(self.n_tracks):
            for layer in range(len(self.merged_private_conv_network)):
                x[track] = \
                    self.merged_private_conv_network[layer][track](x[track])

        # merge time-pitch and pitch-time
        x = torch.cat(x, 1)

        # shared
        for i in range(len(self.shared_conv_network)):
            x = self.shared_conv_network[i](x)

        # chroma stream
        for i in range(len(self.chroma_conv_network)):
            chroma = self.chroma_conv_network[i](chroma)

        # onset/offset stream
        for i in range(len(self.of_off_conv_network)):
            on_off_set = self.of_off_conv_network[i](on_off_set)

        x = torch.cat((x, chroma, on_off_set), 1)

        # all merged
        for i in range(len(self.all_merge_conv_network)):
            x = self.all_merge_conv_network[i](x)

        x = x.view(x.shape[0], -1)
        return x
