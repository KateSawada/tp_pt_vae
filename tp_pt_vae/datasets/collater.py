import numpy as np
import torch

class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self,
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:

        """
        pass

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of tuple of the pair of audio and features.

        Returns:
            Tensor: Gaussian noise (and sine) batch (B, D, T).
            Tensor: Auxiliary feature batch (B, C, T').
            Tensor: Dilated factor batch (B, 1, T).
            Tensor: F0 sequence batch (B, 1, T').
            Tensor: Target signal batch (B, 1, T).

        """
        # y_batch, c_batch, f0_batch, cf0_batch = [], [], [], []
        pianoroll_batch = []
        for idx in range(len(batch)):
            # x, c, f0, cf0 = batch[idx]
            pianoroll = batch[idx]
            # if len(c) > self.batch_max_frames:
            #     # randomly pickup with the batch_max_length length of the part
            #     start_frame = np.random.randint(0, len(c) - self.batch_max_frames)
            #     start_step = start_frame * self.hop_size
            #     y = x[start_step : start_step + self.batch_max_length]
            #     c = c[start_frame : start_frame + self.batch_max_frames]
            #     f0 = f0[start_frame : start_frame + self.batch_max_frames]
            #     cf0 = cf0[start_frame : start_frame + self.batch_max_frames]
            #     dfs = []
            #     for df, us in zip(self.dense_factors, self.prod_upsample_scales):
            #         dfs += [
            #             np.repeat(dilated_factor(cf0, self.sample_rate, df), us)
            #             if self.df_f0_type == "cf0"
            #             else np.repeat(dilated_factor(f0, self.sample_rate, df), us)
            #         ]
            #     self._check_length(y, c, f0, cf0, dfs)
            # else:
            #     logger.warn(f"Removed short sample from batch (length={len(x)}).")
            #     continue
            # y_batch += [y.astype(np.float32).reshape(-1, 1)]  # [(T, 1), ...]
            # c_batch += [c.astype(np.float32)]  # [(T', D), ...]
            # f0_batch += [f0.astype(np.float32).reshape(-1, 1)]  # [(T', 1), ...]
            # cf0_batch += [cf0.astype(np.float32).reshape(-1, 1)]  # [(T', 1), ...]
            pianoroll_batch += [pianoroll.astype(np.float32)]

        # convert each batch to tensor, asuume that each item in batch has the same length
        # y_batch = torch.FloatTensor(np.array(y_batch)).transpose(2, 1)  # (B, 1, T)
        # c_batch = torch.FloatTensor(np.array(c_batch)).transpose(2, 1)  # (B, 1, T')
        # f0_batch = torch.FloatTensor(np.array(f0_batch)).transpose(2, 1)  # (B, 1, T')
        # cf0_batch = torch.FloatTensor(np.array(cf0_batch)).transpose(2, 1)  # (B, 1, T')
        pianoroll_batch = torch.FloatTensor(np.array(pianoroll_batch))

        # # make input signal batch tensor
        # if self.sine_f0_type == "cf0":
        #     in_batch = self.signal_generator(cf0_batch)
        # elif self.sine_f0_type == "f0":
        #     in_batch = self.signal_generator(f0_batch)

        # return (in_batch, c_batch, f0_batch), dfs_batch, y_batch
        pianoroll_batch = torch.permute(pianoroll_batch, (0, 4, 1, 2, 3))
        return pianoroll_batch
