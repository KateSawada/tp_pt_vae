# -*- coding: utf-8 -*-
import torch
import numpy as np

from tp_pt_vae import models


class VAE(torch.nn.Module):
    def __init__(
        self,
        n_tracks,
        n_measures,
        measure_resolution,
        n_beats,
        n_pitches,
        d_latent,
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
        self.d_latent = d_latent

        self.encoder = models.TP_PT_Encoder(
            n_tracks,
            n_measures,
            measure_resolution,
            n_beats,
            n_pitches,
        )

        _sample = torch.zeros(
            (1, n_measures, measure_resolution, n_pitches, n_tracks),
            dtype = torch.float32,
        )

        # _sample = torch.FloatTensor(np.array([np.load("/home/ksawada/Documents/lab/lab_research/musegan/data/extracted/data/71843.npy").astype(np.float32), ]))
        _sample = torch.permute(_sample, (0, 4, 1, 2, 3))
        _sample = _sample.contiguous()
        _sample_out = self.encoder(_sample)

        self.fc_mu = torch.nn.Linear(_sample_out.shape[1], d_latent)
        self.fc_var = torch.nn.Linear(_sample_out.shape[1], d_latent)

        self.decoder = models.TP_PT_Decoder(
            d_latent,
            n_tracks,
            n_measures,
            measure_resolution,
            n_pitches
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, input):
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return (mu, log_var)

    def decode(self, z):
        return self.decoder(z)


    def forward(self, input):
        """forward function

        Args:
            input (Tensor): pianoroll batch.
                [B, n_tracks, n_measures, measure_resolution, n_pitches]

        Returns:
            Tuple[Tuple[Tensor], Tensor, Tensor, Tensor]:
                Tuple[reconstructed], input, mu, log_var
        """
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]
