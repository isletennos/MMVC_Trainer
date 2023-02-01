# -*- coding: utf-8 -*-

# Copyright 2022 Reo Yoneyama (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Adversarial loss modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdversarialLoss(nn.Module):
    """Adversarial loss module."""

    def __init__(
        self,
        average_by_discriminators=False,
        loss_type="mse",
    ):
        """Initialize AversarialLoss module."""
        super(AdversarialLoss, self).__init__()
        self.average_by_discriminators = average_by_discriminators

        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.adv_criterion = self._mse_adv_loss
            self.fake_criterion = self._mse_fake_loss
            self.real_criterion = self._mse_real_loss
        else:
            self.adv_criterion = self._hinge_adv_loss
            self.fake_criterion = self._hinge_fake_loss
            self.real_criterion = self._hinge_real_loss

    def forward(self, p_fakes, p_reals=None):
        """Calcualate generator/discriminator adversarial loss.

        Args:
            p_fakes (list): List of
                discriminator outputs calculated from generator outputs.
            p_reals (list): List of
                discriminator outputs calculated from groundtruth.

        Returns:
            Tensor: Generator adversarial loss value.
            Tensor: Discriminator real loss value.
            Tensor: Discriminator fake loss value.

        """
        # generator adversarial loss
        if p_reals is None:
            adv_loss = 0.0
            for p_fake in p_fakes:
                adv_loss += self.adv_criterion(p_fake)

            if self.average_by_discriminators:
                adv_loss /= len(p_fakes)

            return adv_loss

        # discriminator adversarial loss
        else:
            fake_loss = 0.0
            real_loss = 0.0
            for p_fake, p_real in zip(p_fakes, p_reals):
                fake_loss += self.fake_criterion(p_fake)
                real_loss += self.real_criterion(p_real)

            if self.average_by_discriminators:
                fake_loss /= len(p_fakes)
                real_loss /= len(p_reals)

            return fake_loss, real_loss

    def _mse_adv_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_real_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x):
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_adv_loss(self, x):
        return -x.mean()

    def _hinge_real_loss(self, x):
        return -torch.mean(torch.min(x - 1, x.new_zeros(x.size())))

    def _hinge_fake_loss(self, x):
        return -torch.mean(torch.min(-x - 1, x.new_zeros(x.size())))


class FeatureMatchLoss(nn.Module):
    # Feature matching loss module.

    def __init__(
        self,
        average_by_layers=False,
    ):
        """Initialize FeatureMatchLoss module."""
        super(FeatureMatchLoss, self).__init__()
        self.average_by_layers = average_by_layers

    def forward(self, fmaps_fake, fmaps_real):
        """Calculate forward propagation.

        Args:
            fmaps_fake (list): List of discriminator outputs
                calcuated from generater outputs.
            fmaps_real (list): List of discriminator outputs
                calcuated from groundtruth.

        Returns:
            Tensor: Feature matching loss value.

        """
        fm_loss = 0.0
        for feat_fake, feat_real in zip(fmaps_fake, fmaps_real):
            fm_loss += F.l1_loss(feat_fake, feat_real.detach())

        if self.average_by_layers:
            fm_loss /= len(fmaps_fake)

        return fm_loss
