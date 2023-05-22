# -*- coding: utf-8 -*-

# Copyright 2023 KateSawada
# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Decoding Script for Source-Filter HiFiGAN.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/bigpon/QPPWG

"""

import os
from logging import getLogger

import hydra
import numpy as np
import scipy.stats
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from midi2audio import FluidSynth
from pypianoroll import Multitrack, Track
import matplotlib.pyplot as plt

from tp_pt_vae.datasets import PianorollDataset
# from tp_pt_vae.utils import SignalGenerator, dilated_factor
from tp_pt_vae.utils import midi

# A logger for this file
logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="decode")
def main(config: DictConfig) -> None:
    """Run decoding process."""

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Decode on {device}.")

    # load pre-trained model from checkpoint file
    if config.checkpoint_path is None:
        checkpoint_path = os.path.join(
            config.out_dir,
            "checkpoints",
            f"checkpoint-{config.checkpoint_steps}steps.pkl",
        )
    else:
        checkpoint_path = config.checkpoint_path
    state_dict = torch.load(to_absolute_path(checkpoint_path), map_location="cpu")
    logger.info(f"Loaded model parameters from {checkpoint_path}.")
    model = hydra.utils.instantiate(config.generator)
    model.load_state_dict(state_dict["model"]["generator"])
    # model.remove_weight_norm()
    model.eval().to(device)

    # check directory existence
    out_types = ["pianoroll", "npy", "mid", "wav"]
    for i in out_types:
        out_dir = to_absolute_path(os.path.join(config.out_dir, i, str(config.checkpoint_steps)))
        os.makedirs(out_dir, exist_ok=True)

    z_grid = scipy.stats.truncnorm.rvs(-2, 2, size=(np.prod(config.sample_grid), config.generator.d_latent)).astype(np.float32)
    z_grid = torch.from_numpy(z_grid).to(device)
    with torch.no_grad():
        pianoroll = model(z_grid)
        pianoroll = pianoroll.cpu().detach().numpy().copy()
    measure_resolution = config.data.measure_resolution
    tempo_array = np.full((4 * 4 * measure_resolution, 1), config.data.tempo)

    # TODO: lpdではここの順番変わる
    pianoroll = pianoroll.transpose(1, 0, 2, 3).reshape(config.data.n_tracks, -1, config.data.n_pitches)

    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(
        zip(config.data.programs, config.data.is_drums, config.data.track_names)
    ):
        if len(pianoroll[idx]) >= measure_resolution * 4 * 4:
            pianoroll_ = np.pad(
                pianoroll[idx, :measure_resolution * 4 * 4] > 0.5,  # plot 4 samples
                ((0, 0), (config.data.lowest_pitch, 128 - config.data.lowest_pitch - config.data.n_pitches))
            )
        else:
            pianoroll_ = np.pad(
                pianoroll[idx] > 0.5,
                ((0, 0), (config.data.lowest_pitch, 128 - config.data.lowest_pitch - config.data.n_pitches))
            )
        tracks.append(
            Track(
                name=track_name,
                program=program,
                is_drum=is_drum,
                pianoroll=pianoroll_
            )
        )
    m = Multitrack(tracks=tracks, tempo=tempo_array, resolution=config.data.beat_resolution)

    # save pianoroll as png
    axs = m.plot()

    for ax in axs:
        for x in range(
            measure_resolution,
            4 * measure_resolution * config.data.n_measures,
            measure_resolution
        ):
            if x % (measure_resolution * 4) == 0:
                ax.axvline(x - 0.5, color='k')
            else:
                ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1)
    plt.gcf().set_size_inches((16, 8))

    plt.savefig(os.path.join(config.out_dir, "pianoroll", str(config.checkpoint_steps), "pianoroll.png"))
    plt.clf()
    plt.close()

    # save npy
    np.save(os.path.join(config.out_dir, "npy", str(config.checkpoint_steps), f"npy.npy"), pianoroll)


    # midi npyのsample間に1小節の空白をあける pianoroll.shape = (tracks, timestep, pitches)
    pianoroll_blank = midi.insert_blank_between_samples(pianoroll, pianoroll.shape[1] // np.prod(config.sample_grid), config.data.measure_resolution)
    tracks = []
    for idx, (program, is_drum, track_name) in enumerate(
        zip(config.data.programs, config.data.is_drums, config.data.track_names)
    ):
        pianoroll_ = np.pad(
            pianoroll_blank[idx] > 0.5,
            ((0, 0), (config.data.lowest_pitch, 128 - config.data.lowest_pitch - config.data.n_pitches))
        )
        tracks.append(
            Track(
                name=track_name,
                program=program,
                is_drum=is_drum,
                pianoroll=pianoroll_
            )
        )
    m = Multitrack(tracks=tracks, tempo=tempo_array, resolution=config.data.beat_resolution)
    mid = midi.multitrack_to_pretty_midi(m)

    mid_path = os.path.join(config.out_dir, "mid", str(config.checkpoint_steps), f"mid.mid")
    mid.write(mid_path)

    # wav
    fs = FluidSynth(sound_font=config.sf2_path)
    fs.midi_to_audio(mid_path, os.path.join(config.out_dir, "wav", str(config.checkpoint_steps), f"wav.wav"))


if __name__ == "__main__":
    main()
