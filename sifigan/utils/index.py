# -*- coding: utf-8 -*-

# Copyright 2020 Yi-Chiao Wu (Nagoya University)
#  MIT License (https://opensource.org/licenses/MIT)

"""Indexing-related functions."""

import torch


def pd_indexing(x, d, dilation, batch_index, ch_index):
    """Pitch-dependent indexing of past and future samples.

    Args:
        x (Tensor): Input feature map (B, C, T).
        d (Tensor): Input pitch-dependent dilated factors (B, 1, T).
        dilation (Int): Dilation size.
        batch_index (Tensor): Batch index
        ch_index (Tensor): Channel index

    Returns:
        Tensor: Past output tensor (B, out_channels, T)
        Tensor: Future output tensor (B, out_channels, T)

    """
    B, C, T = x.size()
    batch_index = torch.arange(0, B, dtype=torch.long, device=x.device).reshape(B, 1, 1)
    ch_index = torch.arange(0, C, dtype=torch.long, device=x.device).reshape(1, C, 1)
    dilations = torch.clamp((d * dilation).long(), min=1)

    # get past index (assume reflect padding)
    idx_base = torch.arange(0, T, dtype=torch.long, device=x.device).reshape(1, 1, T)
    idxP = (idx_base - dilations).abs() % T
    idxP = (batch_index, ch_index, idxP)
    
    # get future index (assume reflect padding)
    idxF = idx_base + dilations
    overflowed = idxF >= T
    idxF[overflowed] = -(idxF[overflowed] % T)
    idxF = (batch_index, ch_index, idxF)

    return x[idxP], x[idxF]


def index_initial(n_batch, n_ch, tensor=True, device="cuda"):
    """Tensor batch and channel index initialization.

    Args:
        n_batch (Int): Number of batch.
        n_ch (Int): Number of channel.
        tensor (bool): Return tensor or numpy array

    Returns:
        Tensor: Batch index
        Tensor: Channel index

    """
    batch_index = []
    for i in range(n_batch):
        batch_index.append([[i]] * n_ch)
    ch_index = []
    for i in range(n_ch):
        ch_index += [[i]]
    ch_index = [ch_index] * n_batch

    if tensor:
        batch_index = torch.tensor(batch_index)
        ch_index = torch.tensor(ch_index)
        if torch.cuda.is_available():
            batch_index = batch_index.to(device)
            ch_index = ch_index.to(device)
    return batch_index, ch_index
