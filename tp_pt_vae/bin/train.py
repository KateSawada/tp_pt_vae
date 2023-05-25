# -*- coding: utf-8 -*-

# Copyright 2023 KateSawada
# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Training Script for Source-Filter HiFiGAN.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

import os
import sys
from collections import defaultdict
from logging import getLogger

import hydra
import matplotlib
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from midi2audio import FluidSynth
import pypianoroll
from pypianoroll import Multitrack, Track
import scipy.stats

from tp_pt_vae.datasets import PianorollDataset
from tp_pt_vae.datasets.collater import Collater
from tp_pt_vae.utils import midi

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")


# A logger for this file
logger = getLogger(__name__)



class Trainer(object):
    """Customized trainer module for Source-Filter HiFiGAN training."""

    def __init__(
        self,
        config,
        steps,
        epochs,
        data_loader,
        model,
        criterion,
        optimizer,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            config (dict): Config dict loaded from yaml format configuration file.
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "vae" models.
            criterion (dict): Dict of criterions. It must contrain "kld" and "reconstruct" criterions.
            optimizer (dict): Dict of optimizers. It must contrain "vae" optimizers.
            device (torch.deive): Pytorch device instance.

        """
        self.config = config
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.finish_train = False
        self.writer = SummaryWriter(config.out_dir)
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)

        # random noise to generate sample
        self.sample_z = scipy.stats.truncnorm.rvs(
            -2,
            2,
            size=(np.prod(config.train.sample_grid), self.config.vae.d_latent))
        self.sample_z = torch.FloatTensor(self.sample_z).to(self.device)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config.train.train_max_steps, desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logger.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "vae": self.optimizer["vae"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = {
            "vae": self.model["vae"].state_dict(),
        }

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        self.model["vae"].load_state_dict(state_dict["model"]["vae"])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["vae"].load_state_dict(
                state_dict["optimizer"]["vae"]
            )


    def _train_step(self, batch):
        """Train model one step."""
        # parse batch
        y = batch
        y = y.to(self.device)

        # vae forward
        y_tuple, _, mu, log_var = self.model["vae"](y)
        y_ = y_tuple[0]

        # calculate losses
        reconstruct_loss = self.criterion["reconstruct"](y_, y)
        self.total_train_loss["train/reconstruct_loss"] += reconstruct_loss.item()

        kld_loss = self.criterion["kld"](log_var, mu)
        self.total_train_loss["train/kld_loss"] += kld_loss.item()

        vae_loss = reconstruct_loss + kld_loss


        # update vae
        self.optimizer["vae"].zero_grad()
        vae_loss.backward()
        self.optimizer["vae"].step()


    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(self.data_loader["train"], 1):
            # train one step
            self._train_step(batch)

            # update counts
            self.steps += 1
            self.tqdm.update(1)
            self._check_train_finish()

            # check interval
            self._check_log_interval()
            self._check_eval_interval()
            self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logger.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )


    @torch.no_grad()
    def _eval_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        y = batch
        y = y.to(self.device)

        # vae forward
        y_tuple, _, mu, log_var = self.model["vae"](y)
        y_ = y_tuple[0]

        reconstruct_loss = self.criterion["reconstruct"](y_, y)
        self.total_eval_loss["eval/reconstruct_loss"] += reconstruct_loss.item()

        kld_loss = self.criterion["kld"](log_var, mu)
        self.total_eval_loss["eval/kld_loss"] += kld_loss.item()

        vae_loss = reconstruct_loss + kld_loss

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logger.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["valid"], desc="[eval]"), 1
        ):
            # eval one step
            self._eval_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)
            if eval_steps_per_epoch == 3:
                break

        logger.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logger.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        y = batch
        # use only the first sample
        y = y[:1].to(self.device)

        # vae forward
        outs = self.model["vae"].decode(self.sample_z)

        measure_resolution = self.config.data.measure_resolution
        tempo_array = np.full((4 * 4 * measure_resolution, 1), self.config.data.tempo)

        for pianoroll, name in zip(outs, ["generated"]):
            if pianoroll is not None:
                pianoroll = pianoroll.cpu().detach().numpy().copy()

                pianoroll = pianoroll.reshape(self.config.data.n_tracks, -1, self.config.data.n_pitches)

                tracks = []
                for idx, (program, is_drum, track_name) in enumerate(
                    zip(self.config.data.programs, self.config.data.is_drums, self.config.data.track_names)
                ):
                    if len(pianoroll[idx]) >= measure_resolution * 4 * 4:
                        pianoroll_ = np.pad(
                            pianoroll[idx, :measure_resolution * 4 * 4] > 0.5,  # plot 4 samples
                            ((0, 0), (self.config.data.lowest_pitch, 128 - self.config.data.lowest_pitch - self.config.data.n_pitches))
                        )
                    else:
                        pianoroll_ = np.pad(
                            pianoroll[idx] > 0.5,
                            ((0, 0), (self.config.data.lowest_pitch, 128 - self.config.data.lowest_pitch - self.config.data.n_pitches))
                        )
                    tracks.append(
                        Track(
                            name=track_name,
                            program=program,
                            is_drum=is_drum,
                            pianoroll=pianoroll_
                        )
                    )
                m = Multitrack(tracks=tracks, tempo=tempo_array, resolution=self.config.data.beat_resolution)

                # save pianoroll as png
                # axs = m.plot()

                # for ax in axs:
                #     for x in range(
                #         measure_resolution,
                #         4 * measure_resolution * self.config.data.n_measures,
                #         measure_resolution
                #     ):
                #         if x % (measure_resolution * 4) == 0:
                #             ax.axvline(x - 0.5, color='k')
                #         else:
                #             ax.axvline(x - 0.5, color='k', linestyle='-', linewidth=1)
                # plt.gcf().set_size_inches((16, 8))

                # self.mkdir(os.path.join(self.config.out_dir, "pianoroll"))
                # plt.savefig(os.path.join(self.config.out_dir, "pianoroll", f"pianoroll-{self.steps}steps.png"))
                # plt.clf()
                # plt.close()

                # save npy
                self.mkdir(os.path.join(self.config.out_dir, "npy"))
                np.save(os.path.join(self.config.out_dir, "npy", f"npy-{self.steps}steps.npy"), pianoroll)


                # midi npyのsample間に1小節の空白をあける pianoroll.shape = (tracks, timestep, pitches)
                pianoroll_blank = midi.insert_blank_between_samples(pianoroll, self.config.data.measure_resolution * self.config.data.n_tracks , self.config.data.measure_resolution)
                tracks = []
                for idx, (program, is_drum, track_name) in enumerate(
                    zip(self.config.data.programs, self.config.data.is_drums, self.config.data.track_names)
                ):
                    pianoroll_ = np.pad(
                        pianoroll_blank[idx] > 0.5,
                        ((0, 0), (self.config.data.lowest_pitch, 128 - self.config.data.lowest_pitch - self.config.data.n_pitches))
                    )
                    tracks.append(
                        Track(
                            name=track_name,
                            program=program,
                            is_drum=is_drum,
                            pianoroll=pianoroll_
                        )
                    )
                m = Multitrack(tracks=tracks, tempo=tempo_array, resolution=self.config.data.beat_resolution)
                mid = midi.multitrack_to_pretty_midi(m)

                self.mkdir(os.path.join(self.config.out_dir, "mid"))
                mid_path = os.path.join(self.config.out_dir, "mid", f"mid-{self.steps}steps.mid")
                mid.write(mid_path)

                # wav
                self.mkdir(os.path.join(self.config.out_dir, "wav"))
                fs = FluidSynth(sound_font=self.config.sf2_path)
                fs.midi_to_audio(mid_path, os.path.join(self.config.out_dir, "wav", f"wav-{self.steps}steps.wav"))


    @staticmethod
    def mkdir(path):
        """make directory if it doesn't exist'

        Args:
            path (str): path to directory
        """
        if not os.path.exists(path):
            os.mkdir(path)

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config.train.save_interval_steps == 0:
            self.save_checkpoint(
                os.path.join(
                    self.config.out_dir,
                    "checkpoints",
                    f"checkpoint-{self.steps}steps.pkl",
                )
            )
            logger.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config.train.eval_interval_steps == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config.train.log_interval_steps == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config.train.log_interval_steps
                logger.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)

            # reset
            self.total_train_loss = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config.train.train_max_steps:
            self.finish_train = True


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config: DictConfig) -> None:
    """Run training process."""

    if not torch.cuda.is_available():
        print("CPU")
        device = torch.device("cpu")
    else:
        print("GPU")
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True

    # fix seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    os.environ["PYTHONHASHSEED"] = str(config.seed)

    # check directory existence
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)

    # write config to yaml file
    with open(os.path.join(config.out_dir, "config.yaml"), "w") as f:
        f.write(OmegaConf.to_yaml(config))
    logger.info(OmegaConf.to_yaml(config))

    train_dataset = PianorollDataset(
        pianoroll_list=to_absolute_path(config.data.train_pianoroll),
        n_tracks=config.data.n_tracks,
        measure_resolution=config.data.measure_resolution,
        n_pitches=config.data.n_pitches,
        n_measures=config.data.n_measures,
        allow_cache=config.data.allow_cache,
    )
    logger.info(f"The number of training files = {len(train_dataset)}.")

    valid_dataset = PianorollDataset(
        pianoroll_list=to_absolute_path(config.data.valid_pianoroll),
        n_tracks=config.data.n_tracks,
        measure_resolution=config.data.measure_resolution,
        n_pitches=config.data.n_pitches,
        n_measures=config.data.n_measures,
        allow_cache=config.data.allow_cache,
    )
    logger.info(f"The number of validation files = {len(valid_dataset)}.")

    dataset = {"train": train_dataset, "valid": valid_dataset}

    # get data loader
    collater = Collater()
    train_sampler, valid_sampler = None, None
    data_loader = {
        "train": DataLoader(
            dataset=dataset["train"],
            shuffle=True,
            collate_fn=collater,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=train_sampler,
            pin_memory=config.data.pin_memory,
        ),
        "valid": DataLoader(
            dataset=dataset["valid"],
            shuffle=True,
            collate_fn=collater,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers,
            sampler=valid_sampler,
            pin_memory=config.data.pin_memory,
        ),
    }

    # define models and optimizers
    model = {
        "vae": hydra.utils.instantiate(config.vae).to(device),
    }

    # define training criteria
    criterion = {
        "reconstruct": hydra.utils.instantiate(config.train.reconstruct_loss).to(device),
        "kld": hydra.utils.instantiate(config.train.kld_loss).to(device),
    }

    # define optimizers and schedulers
    optimizer = {
        "vae": hydra.utils.instantiate(
            config.train.vae_optimizer, params=model["vae"].parameters()
        ),
    }

    # define trainer
    trainer = Trainer(
        config=config,
        steps=0,
        epochs=0,
        data_loader=data_loader,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # load trained parameters from checkpoint
    if config.train.resume:
        resume = os.path.join(
            config.out_dir, "checkpoints", f"checkpoint-{config.train.resume}steps.pkl"
        )
        if os.path.exists(resume):
            trainer.load_checkpoint(resume)
            logger.info(f"Successfully resumed from {resume}.")
        else:
            logger.info(f"Failed to resume from {resume}.")
            sys.exit(0)
    else:
        logger.info("Start a new training process.")

    # run training loop
    try:
        trainer.run()
    except KeyboardInterrupt:
        trainer.save_checkpoint(
            os.path.join(
                config.out_dir, "checkpoints", f"checkpoint-{trainer.steps}steps.pkl"
            )
        )
        logger.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
