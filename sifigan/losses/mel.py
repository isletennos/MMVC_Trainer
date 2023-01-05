# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based loss modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel


def stft(
    x, fft_size, hop_size, win_length, window, center=True, onesided=True, power=False
):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    x_stft = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window,
        center=center,
        onesided=onesided,
        return_complex=False,
    )
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    if power:
        return torch.clamp(real**2 + imag**2, min=1e-7).transpose(2, 1)
    else:
        return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class MelSpectralLoss(nn.Module):
    """Mel-spectral L1 loss module."""

    def __init__(
        self,
        fft_size=1024,
        hop_size=120,
        win_length=1024,
        window="hann_window",
        sample_rate=24000,
        n_mels=80,
        fmin=0,
        fmax=None,
    ):
        """Initialize MelSpectralLoss loss.

        Args:
            fft_size (int): FFT points.
            hop_length (int): Hop length.
            win_length (Optional[int]): Window length.
            window (str): Window type.
            sample_rate (int): Sampling rate.
            n_mels (int): Number of Mel basis.
            fmin (Optional[int]): Minimum frequency of mel-filter-bank.
            fmax (Optional[int]): Maximum frequency of mel-filter-bank.

        """
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length if win_length is not None else fft_size
        self.register_buffer("window", getattr(torch, window)(self.win_length))
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2
        melmat = librosa_mel(
            sr=sample_rate, n_fft=fft_size, n_mels=n_mels, fmin=fmin, fmax=fmax
        ).T
        self.register_buffer("melmat", torch.from_numpy(melmat).float())

    def forward(self, x, y):
        """Calculate Mel-spectral L1 loss.

        Args:
            x (Tensor): Generated waveform tensor (B, 1, T).
            y (Tensor): Groundtruth waveform tensor (B, 1, T).

        Returns:
            Tensor: Mel-spectral L1 loss value.

        """
        x = x.squeeze(1)
        y = y.squeeze(1)
        x_mag = stft(x, self.fft_size, self.hop_size, self.win_length, self.window)
        y_mag = stft(y, self.fft_size, self.hop_size, self.win_length, self.window)
        x_log_mel = torch.log(torch.clamp(torch.matmul(x_mag, self.melmat), min=1e-7))
        y_log_mel = torch.log(torch.clamp(torch.matmul(y_mag, self.melmat), min=1e-7))
        mel_loss = F.l1_loss(x_log_mel, y_log_mel)

        return mel_loss
