# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Spectral envelopes estimation module based on CheapTrick.

References:
    - https://www.sciencedirect.com/science/article/pii/S0167639314000697
    - https://github.com/mmorise/World

"""

import math

import torch
import torch.fft
import torch.nn as nn


class AdaptiveWindowing(nn.Module):
    """CheapTrick F0 adptive windowing module."""

    def __init__(
        self,
        sample_rate,
        hop_size,
        fft_size,
        f0_floor,
        f0_ceil,
    ):
        """Initilize AdaptiveWindowing module.

        Args:
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size.
            fft_size (int): FFT size.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.

        """
        super(AdaptiveWindowing, self).__init__()

        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.fft_size = fft_size
        self.register_buffer("window", torch.zeros((f0_ceil + 1, fft_size)))
        self.zero_padding = nn.ConstantPad2d((fft_size // 2, fft_size // 2, 0, 0), 0)

        # Pre-calculation of the window functions
        for f0 in range(f0_floor, f0_ceil + 1):
            half_win_len = round(1.5 * self.sample_rate / f0)
            base_index = torch.arange(
                -half_win_len, half_win_len + 1, dtype=torch.int64
            )
            position = base_index / 1.5 / self.sample_rate
            left = fft_size // 2 - half_win_len
            right = fft_size // 2 + half_win_len + 1
            window = torch.zeros(fft_size)
            window[left:right] = 0.5 * torch.cos(math.pi * position * f0) + 0.5
            average = torch.sum(window * window).pow(0.5)
            self.window[f0] = window / average

    def forward(self, x, f, power=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Waveform (B, fft_size // 2 + 1, T).
            f (Tensor): F0 sequence (B, T').
            power (boot): Whether to use power or magnitude.

        Returns:
            Tensor: Power spectrogram (B, bin_size, T').

        """
        # Get the matrix of window functions corresponding to F0
        x = self.zero_padding(x).unfold(1, self.fft_size, self.hop_size)
        windows = self.window[f]

        # Adaptive windowing and calculate power spectrogram.
        # In test, change x[:, : -1, :] to x.
        x = torch.abs(torch.fft.rfft(x[:, :-1, :] * windows))
        x = x.pow(2) if power else x

        return x


class AdaptiveLiftering(nn.Module):
    """CheapTrick F0 adptive windowing module."""

    def __init__(
        self,
        sample_rate,
        fft_size,
        f0_floor,
        f0_ceil,
        q1=-0.15,
    ):
        """Initilize AdaptiveLiftering module.

        Args:
            sample_rate (int): Sampling rate.
            fft_size (int): FFT size.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.
            q1 (float): Parameter to remove effect of adjacent harmonics.

        """
        super(AdaptiveLiftering, self).__init__()

        self.sample_rate = sample_rate
        self.bin_size = fft_size // 2 + 1
        self.q1 = q1
        self.q0 = 1.0 - 2.0 * q1
        self.register_buffer(
            "smoothing_lifter", torch.zeros((f0_ceil + 1, self.bin_size))
        )
        self.register_buffer(
            "compensation_lifter", torch.zeros((f0_ceil + 1, self.bin_size))
        )

        # Pre-calculation of the smoothing lifters and compensation lifters
        for f0 in range(f0_floor, f0_ceil + 1):
            smoothing_lifter = torch.zeros(self.bin_size)
            compensation_lifter = torch.zeros(self.bin_size)
            quefrency = torch.arange(1, self.bin_size) / sample_rate
            smoothing_lifter[0] = 1.0
            smoothing_lifter[1:] = torch.sin(math.pi * f0 * quefrency) / (
                math.pi * f0 * quefrency
            )
            compensation_lifter[0] = self.q0 + 2.0 * self.q1
            compensation_lifter[1:] = self.q0 + 2.0 * self.q1 * torch.cos(
                2.0 * math.pi * f0 * quefrency
            )
            self.smoothing_lifter[f0] = smoothing_lifter
            self.compensation_lifter[f0] = compensation_lifter

    def forward(self, x, f, elim_0th=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Power spectrogram (B, bin_size, T').
            f (Tensor): F0 sequence (B, T').
            elim_0th (bool): Whether to eliminate cepstram 0th component.

        Returns:
            Tensor: Estimated spectral envelope (B, bin_size, T').

        """
        # Setting the smoothing lifter and compensation lifter
        smoothing_lifter = self.smoothing_lifter[f]
        compensation_lifter = self.compensation_lifter[f]

        # Calculating cepstrum
        tmp = torch.cat((x, torch.flip(x[:, :, 1:-1], [2])), dim=2)
        cepstrum = torch.fft.rfft(torch.log(torch.clamp(tmp, min=1e-7))).real

        # Set the 0th cepstrum to 0
        if elim_0th:
            cepstrum[..., 0] = 0

        # Liftering cepstrum with the lifters
        liftered_cepstrum = cepstrum * smoothing_lifter * compensation_lifter

        # Return the result to the spectral domain
        x = torch.fft.irfft(liftered_cepstrum)[:, :, : self.bin_size]

        return x


class CheapTrick(nn.Module):
    """CheapTrick based spectral envelope estimation module."""

    def __init__(
        self,
        sample_rate,
        hop_size,
        fft_size,
        f0_floor=70,
        f0_ceil=340,
        uv_threshold=0,
        q1=-0.15,
    ):
        """Initilize AdaptiveLiftering module.

        Args:
            sample_rate (int): Sampling rate.
            hop_size (int): Hop size.
            fft_size (int): FFT size.
            f0_floor (int): Minimum value of F0.
            f0_ceil (int): Maximum value of F0.
            uv_threshold (float): V/UV determining threshold.
            q1 (float): Parameter to remove effect of adjacent harmonics.

        """
        super(CheapTrick, self).__init__()

        # fft_size must be larger than 3.0 * sample_rate / f0_floor
        assert fft_size > 3.0 * sample_rate / f0_floor
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.uv_threshold = uv_threshold

        self.ada_wind = AdaptiveWindowing(
            sample_rate,
            hop_size,
            fft_size,
            f0_floor,
            f0_ceil,
        )
        self.ada_lift = AdaptiveLiftering(
            sample_rate,
            fft_size,
            f0_floor,
            f0_ceil,
            q1,
        )

    def forward(self, x, f, power=False, elim_0th=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Power spectrogram (B, T).
            f (Tensor): F0 sequence (B, T').
            power (boot): Whether to use power or magnitude spectrogram.
            elim_0th (bool): Whether to eliminate cepstram 0th component.

        Returns:
            Tensor: Estimated spectral envelope (B, bin_size, T').

        """
        # Step0: Round F0 values to integers.
        voiced = (f > self.uv_threshold) * torch.ones_like(f)
        f = voiced * f + (1.0 - voiced) * self.f0_ceil
        f = torch.round(torch.clamp(f, min=self.f0_floor, max=self.f0_ceil)).to(
            torch.int64
        )

        # Step1: Adaptive windowing and calculate power or amplitude spectrogram.
        x = self.ada_wind(x, f, power)

        # Step3: Smoothing (log axis) and spectral recovery on the cepstrum domain.
        x = self.ada_lift(x, f, elim_0th)

        return x
