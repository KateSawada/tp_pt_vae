# -*- coding: utf-8 -*-

# Copyright 2023 KateSawada
# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Dataset modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG

"""

from logging import getLogger
from multiprocessing import Manager

import numpy as np
from hydra.utils import to_absolute_path
from tp_pt_vae.utils import read_txt
from torch.utils.data import Dataset

# A logger for this file
logger = getLogger(__name__)


class PianorollDataset(Dataset):
    """PyTorch compatible audio and acoustic feat. dataset."""

    def __init__(
        self,
        pianoroll_list,
        n_tracks=5,
        measure_resolution=48,
        n_pitches=84,
        n_measures=4,
        return_filename=False,
        allow_cache=False,
    ):
        """Initialize dataset.

        Args:
            pianoroll_list (str): Filename of the list of pianoroll npy files.
            n_tracks (int): Number of tracks.
            measure_resolution (int): timestep resolution per measure.
            n_pitches (int): Number of pitches.
            n_measures (int): Number of measures.
            return_filename (bool): Whether to return the filename with arrays.
            allow_cache (bool): Whether to allow cache of the loaded files.
        """
        # load pianoroll files & check filename
        pianoroll_files = read_txt(to_absolute_path(pianoroll_list))

        self.pianoroll_files = pianoroll_files
        self.n_tracks = n_tracks
        self.measure_resolution = measure_resolution
        self.n_pitches = n_pitches
        self.n_measures = n_measures
        self.return_filename = return_filename
        self.allow_cache = allow_cache

        if allow_cache:
            # NOTE(kan-bayashi): Manager is need to share memory in dataloader with num_workers > 0
            self.manager = Manager()
            self.caches = self.manager.list()
            self.caches += [() for _ in range(len(pianoroll_files))]


    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only in return_filename = True).
            ndarray: Pianoroll (measures, timestep, pitches, tracks).
        """
        if self.allow_cache and len(self.caches[idx]) != 0:
            return self.caches[idx]
        # load audio and features
        pianoroll = np.load(to_absolute_path(self.pianoroll_files[idx]))
        pianoroll = pianoroll.astype(np.float32)


        if self.return_filename:
            items = (self.pianoroll_files[idx], pianoroll)
        else:
            items = pianoroll

        if self.allow_cache:
            self.caches[idx] = items

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.pianoroll_files)
