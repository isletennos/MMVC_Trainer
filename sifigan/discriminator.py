# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Discriminator modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN
    - https://github.com/jik876/hifi-gan
    - UnivNet: A Neural Vocoder with Multi-Resolution Spectrogram Discriminators for High-Fidelity Waveform Generation
      https://www.isca-speech.org/archive/interspeech_2021/jang21_interspeech.html

"""

import copy
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import spectrogram

# A logger for this file
logger = getLogger(__name__)


class HiFiGANPeriodDiscriminator(nn.Module):
    """HiFiGAN period discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        period=3,
        kernel_sizes=[5, 3],
        channels=32,
        downsample_scales=[3, 3, 3, 3, 1],
        max_downsample_channels=1024,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_spectral_norm=False,
    ):
        """Initialize HiFiGANPeriodDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            period (int): Period.
            kernel_sizes (list): Kernel sizes of initial conv layers and the final conv layer.
            channels (int): Number of initial channels.
            downsample_scales (list): List of downsampling scales.
            max_downsample_channels (int): Number of maximum downsampling channels.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        assert len(kernel_sizes) == 2
        assert kernel_sizes[0] % 2 == 1, "Kernel size must be odd number."
        assert kernel_sizes[1] % 2 == 1, "Kernel size must be odd number."

        self.period = period
        self.convs = nn.ModuleList()
        in_chs = in_channels
        out_chs = channels
        for downsample_scale in downsample_scales:
            self.convs += [
                nn.Sequential(
                    nn.Conv2d(
                        in_chs,
                        out_chs,
                        (kernel_sizes[0], 1),
                        (downsample_scale, 1),
                        padding=((kernel_sizes[0] - 1) // 2, 0),
                        bias=bias,
                    ),
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Use downsample_scale + 1?
            out_chs = min(out_chs * 4, max_downsample_channels)
        self.output_conv = nn.Conv2d(
            out_chs,
            out_channels,
            (kernel_sizes[1] - 1, 1),
            1,
            padding=((kernel_sizes[1] - 1) // 2, 0),
            bias=bias,
        )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            c (Tensor): Input tensor (B, in_channels, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            list: List of each layer's tensors.

        """
        # transform 1d to 2d -> (B, C, T/P, P)
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t += n_pad
        x = x.view(b, c, t // self.period, self.period)

        # forward conv
        fmap = []
        for f in self.convs:
            x = f(x)
            if return_fmaps:
                fmap.append(x)
        x = self.output_conv(x)
        out = torch.flatten(x, 1, -1)

        if return_fmaps:
            return out, fmap
        else:
            return out

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.spectral_norm(m)
                logger.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class HiFiGANMultiPeriodDiscriminator(nn.Module):
    """HiFiGAN multi-period discriminator module."""

    def __init__(
        self,
        periods=[2, 3, 5, 7, 11],
        discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initialize HiFiGANMultiPeriodDiscriminator module.

        Args:
            periods (list): List of periods.
            discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        self.discriminators = nn.ModuleList()
        for period in periods:
            params = copy.deepcopy(discriminator_params)
            params["period"] = period
            self.discriminators += [HiFiGANPeriodDiscriminator(**params)]

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs, fmaps = [], []
        for f in self.discriminators:
            if return_fmaps:
                out, fmap = f(x, return_fmaps)
                fmaps.extend(fmap)
            else:
                out = f(x)
            outs.append(out)

        if return_fmaps:
            return outs, fmaps
        else:
            return outs


class HiFiGANScaleDiscriminator(nn.Module):
    """HiFi-GAN scale discriminator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_sizes=[15, 41, 5, 3],
        channels=128,
        max_downsample_channels=1024,
        max_groups=16,
        bias=True,
        downsample_scales=[2, 2, 4, 4, 1],
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.1},
        use_weight_norm=True,
        use_spectral_norm=False,
    ):
        """Initilize HiFiGAN scale discriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_sizes (list): List of four kernel sizes. The first will be used for the first conv layer,
                and the second is for downsampling part, and the remaining two are for output layers.
            channels (int): Initial number of channels for conv layer.
            max_downsample_channels (int): Maximum number of channels for downsampling layers.
            bias (bool): Whether to add bias parameter in convolution layers.
            downsample_scales (list): List of downsampling scales.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()
        self.layers = nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == 4
        for ks in kernel_sizes:
            assert ks % 2 == 1

        # add first layer
        self.layers += [
            nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    channels,
                    # NOTE(kan-bayashi): Use always the same kernel size
                    kernel_sizes[0],
                    bias=bias,
                    padding=(kernel_sizes[0] - 1) // 2,
                ),
                getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        # add downsample layers
        in_chs = channels
        out_chs = channels
        # NOTE(kan-bayashi): Remove hard coding?
        groups = 4
        for downsample_scale in downsample_scales:
            self.layers += [
                nn.Sequential(
                    nn.Conv1d(
                        in_chs,
                        out_chs,
                        kernel_size=kernel_sizes[1],
                        stride=downsample_scale,
                        padding=(kernel_sizes[1] - 1) // 2,
                        groups=groups,
                        bias=bias,
                    ),
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]
            in_chs = out_chs
            # NOTE(kan-bayashi): Remove hard coding?
            out_chs = min(in_chs * 2, max_downsample_channels)
            # NOTE(kan-bayashi): Remove hard coding?
            groups = min(groups * 4, max_groups)

        # add final layers
        out_chs = min(in_chs * 2, max_downsample_channels)
        self.layers += [
            nn.Sequential(
                nn.Conv1d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_sizes[2],
                    stride=1,
                    padding=(kernel_sizes[2] - 1) // 2,
                    bias=bias,
                ),
                getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.last_layer = nn.Conv1d(
            out_chs,
            out_channels,
            kernel_size=kernel_sizes[3],
            stride=1,
            padding=(kernel_sizes[3] - 1) // 2,
            bias=bias,
        )

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of output tensors of each layer.

        """
        fmap = []
        for f in self.layers:
            x = f(x)
            if return_fmaps:
                fmap.append(x)
        out = self.last_layer(x)

        if return_fmaps:
            return out, fmap
        else:
            return out

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.spectral_norm(m)
                logger.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class HiFiGANMultiScaleDiscriminator(nn.Module):
    """HiFi-GAN multi-scale discriminator module."""

    def __init__(
        self,
        scales=3,
        downsample_pooling="AvgPool1d",
        # follow the official implementation setting
        downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm=False,
    ):
        """Initilize HiFiGAN multi-scale discriminator module.

        Args:
            scales (int): Number of multi-scales.
            downsample_pooling (str): Pooling module name for downsampling of the inputs.
            downsample_pooling_params (dict): Parameters for the above pooling module.
            discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.

        """
        super().__init__()
        self.discriminators = nn.ModuleList()

        # add discriminators
        for i in range(scales):
            params = copy.deepcopy(discriminator_params)
            if follow_official_norm:
                if i == 0:
                    params["use_weight_norm"] = False
                    params["use_spectral_norm"] = True
                else:
                    params["use_weight_norm"] = True
                    params["use_spectral_norm"] = False
            self.discriminators += [HiFiGANScaleDiscriminator(**params)]
        self.pooling = getattr(nn, downsample_pooling)(**downsample_pooling_params)

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs, fmaps = [], []
        for f in self.discriminators:
            if return_fmaps:
                out, fmap = f(x, return_fmaps)
                fmaps.extend(fmap)
            else:
                out = f(x)
            outs.append(out)
            x = self.pooling(x)

        if return_fmaps:
            return outs, fmaps
        else:
            return outs


class HiFiGANMultiScaleMultiPeriodDiscriminator(nn.Module):
    """HiFi-GAN multi-scale + multi-period discriminator module."""

    def __init__(
        self,
        # Multi-scale discriminator related
        scales=3,
        scale_downsample_pooling="AvgPool1d",
        scale_downsample_pooling_params={
            "kernel_size": 4,
            "stride": 2,
            "padding": 2,
        },
        scale_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [15, 41, 5, 3],
            "channels": 128,
            "max_downsample_channels": 1024,
            "max_groups": 16,
            "bias": True,
            "downsample_scales": [2, 2, 4, 4, 1],
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
        },
        follow_official_norm=True,
        # Multi-period discriminator related
        periods=[2, 3, 5, 7, 11],
        period_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initilize HiFiGAN multi-scale + multi-period discriminator module.

        Args:
            scales (int): Number of multi-scales.
            scale_downsample_pooling (str): Pooling module name for downsampling of the inputs.
            scale_downsample_pooling_params (dict): Parameters for the above pooling module.
            scale_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            follow_official_norm (bool): Whether to follow the norm setting of the official
                implementaion. The first discriminator uses spectral norm and the other
                discriminators use weight norm.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        self.msd = HiFiGANMultiScaleDiscriminator(
            scales=scales,
            downsample_pooling=scale_downsample_pooling,
            downsample_pooling_params=scale_downsample_pooling_params,
            discriminator_params=scale_discriminator_params,
            follow_official_norm=follow_official_norm,
        )
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.

        """
        if return_fmaps:
            msd_outs, msd_fmaps = self.msd(x, return_fmaps)
            mpd_outs, mpd_fmaps = self.mpd(x, return_fmaps)
            outs = msd_outs + mpd_outs
            fmaps = msd_fmaps + mpd_fmaps
            return outs, fmaps
        else:
            msd_outs = self.msd(x)
            mpd_outs = self.mpd(x)
            outs = msd_outs + mpd_outs
            return outs


class UnivNetSpectralDiscriminator(nn.Module):
    """UnivNet spectral discriminator module."""

    def __init__(
        self,
        fft_size,
        hop_size,
        win_length,
        window="hann_window",
        kernel_sizes=[(3, 9), (3, 9), (3, 9), (3, 9), (3, 3), (3, 3)],
        strides=[(1, 1), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1)],
        channels=32,
        bias=True,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        use_weight_norm=True,
    ):
        """Initilize HiFiGAN scale discriminator module.

        Args:
            fft_size (list): FFT size.
            hop_size (int): Hop size.
            win_length (int): Window length.
            window (stt): Name of window function.
            kernel_sizes (list): List of kernel sizes in down-sampling CNNs.
            strides (list): List of stride sizes in down-sampling CNNs.
            channels (int): Number of channels for conv layer.
            bias (bool): Whether to add bias parameter in convolution layers.
            nonlinear_activation (str): Activation function module name.
            nonlinear_activation_params (dict): Hyperparameters for activation function.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super().__init__()

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))

        self.layers = nn.ModuleList()

        # check kernel size is valid
        assert len(kernel_sizes) == len(strides)

        # add first layer
        self.layers += [
            nn.Sequential(
                nn.Conv2d(
                    1,
                    channels,
                    kernel_sizes[0],
                    stride=strides[0],
                    bias=bias,
                ),
                getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]

        for i in range(1, len(kernel_sizes) - 2):
            self.layers += [
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels,
                        kernel_size=kernel_sizes[i],
                        stride=strides[i],
                        bias=bias,
                    ),
                    getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
                )
            ]

        # add final layers
        self.layers += [
            nn.Sequential(
                nn.Conv2d(
                    channels,
                    channels,
                    kernel_size=kernel_sizes[-2],
                    stride=strides[-2],
                    bias=bias,
                ),
                getattr(nn, nonlinear_activation)(**nonlinear_activation_params),
            )
        ]
        self.layers += [
            nn.Conv2d(
                channels,
                1,
                kernel_size=kernel_sizes[-1],
                stride=strides[-1],
                bias=bias,
            )
        ]

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of output tensors of each layer.

        """
        x = spectrogram(
            x,
            pad=self.win_length // 2,
            window=self.window,
            n_fft=self.fft_size,
            hop_length=self.hop_size,
            win_length=self.win_length,
            power=1.0,
            normalized=False,
        ).transpose(-1, -2)

        fmap = []
        for f in self.layers:
            x = f(x)
            if return_fmaps:
                fmap.append(x)

        if return_fmaps:
            return x, fmap
        else:
            return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


class UnivNetMultiResolutionSpectralDiscriminator(nn.Module):
    """UnivNet multi-resolution spectral discriminator module."""

    def __init__(
        self,
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        discriminator_params={
            "channels": 32,
            "kernel_sizes": [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3), (3, 3)],
            "strides": [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1)],
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.2},
        },
    ):
        """Initilize UnivNetMultiResolutionSpectralDiscriminator module.

        Args:
            fft_sizes (list): FFT sizes for each spectral discriminator.
            hop_sizes (list): Hop sizes for each spectral discriminator.
            win_lengths (list): Window lengths for each spectral discriminator.
            window (stt): Name of window function.
            discriminator_params (dict): Parameters for univ-net spectral discriminator module.

        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.discriminators = nn.ModuleList()

        # add discriminators
        for i in range(len(fft_sizes)):
            params = copy.deepcopy(discriminator_params)
            self.discriminators += [
                UnivNetSpectralDiscriminator(
                    fft_size=fft_sizes[i],
                    hop_size=hop_sizes[i],
                    win_length=win_lengths[i],
                    window=window,
                    **params,
                )
            ]

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs, which consists of each layer output tensors.

        """
        outs, fmaps = [], []
        for f in self.discriminators:
            if return_fmaps:
                out, fmap = f(x, return_fmaps)
                fmaps.extend(fmap)
            else:
                out = f(x)
            outs.append(out)

        if return_fmaps:
            return outs, fmaps
        else:
            return outs


class UnivNetMultiResolutionMultiPeriodDiscriminator(nn.Module):
    """UnivNet multi-resolution + multi-period discriminator module."""

    def __init__(
        self,
        # Multi-resolution discriminator related
        fft_sizes=[1024, 2048, 512],
        hop_sizes=[120, 240, 50],
        win_lengths=[600, 1200, 240],
        window="hann_window",
        spectral_discriminator_params={
            "channels": 32,
            "kernel_sizes": [(3, 9), (3, 9), (3, 9), (3, 9), (3, 3), (3, 3)],
            "strides": [(1, 1), (1, 2), (1, 2), (1, 2), (1, 1), (1, 1)],
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.2},
        },
        # Multi-period discriminator related
        periods=[2, 3, 5, 7, 11],
        period_discriminator_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_sizes": [5, 3],
            "channels": 32,
            "downsample_scales": [3, 3, 3, 3, 1],
            "max_downsample_channels": 1024,
            "bias": True,
            "nonlinear_activation": "LeakyReLU",
            "nonlinear_activation_params": {"negative_slope": 0.1},
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initilize UnivNetMultiResolutionMultiPeriodDiscriminator module.

        Args:
            fft_sizes (list): FFT sizes for each spectral discriminator.
            hop_sizes (list): Hop sizes for each spectral discriminator.
            win_lengths (list): Window lengths for each spectral discriminator.
            window (stt): Name of window function.
            sperctral_discriminator_params (dict): Parameters for hifi-gan scale discriminator module.
            periods (list): List of periods.
            period_discriminator_params (dict): Parameters for hifi-gan period discriminator module.
                The period parameter will be overwritten.

        """
        super().__init__()
        self.mrd = UnivNetMultiResolutionSpectralDiscriminator(
            fft_sizes=fft_sizes,
            hop_sizes=hop_sizes,
            win_lengths=win_lengths,
            window=window,
            discriminator_params=spectral_discriminator_params,
        )
        self.mpd = HiFiGANMultiPeriodDiscriminator(
            periods=periods,
            discriminator_params=period_discriminator_params,
        )

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input noise signal (B, 1, T).
            return_fmaps (bool): Whether to return feature maps.

        Returns:
            List: List of list of each discriminator outputs,
                which consists of each layer output tensors.
                Multi scale and multi period ones are concatenated.

        """
        if return_fmaps:
            mrd_outs, mrd_fmaps = self.mrd(x, return_fmaps)
            mpd_outs, mpd_fmaps = self.mpd(x, return_fmaps)
            outs = mrd_outs + mpd_outs
            fmaps = mrd_fmaps + mpd_fmaps

            return outs, fmaps
        else:
            mrd_outs = self.mrd(x)
            mpd_outs = self.mpd(x)
            outs = mrd_outs + mpd_outs

            return outs
