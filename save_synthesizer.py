import os
import json
import argparse
import itertools
import math
import sys
from psutil import cpu_count
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
from torch.cuda.amp import autocast, GradScaler
import datetime
import pytz
import time
from tqdm import tqdm


import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols


def main():
  hps = utils.get_hparams()
  run(0, hps)


def run(rank, hps):
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)

  if not hasattr(hps.model, "use_mel_train"):
    hps.model.use_mel_train = False

  if hps.model.use_mel_train:
      channels = hps.data.n_mel_channels
  else:
      channels = hps.data.filter_length // 2 + 1

  net_g = SynthesizerTrn(
      len(symbols),
      channels,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      hps_data=hps.data,
      **hps.model)

  _, _, _, _ = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*[0-9].pth"), net_g, optimizer=None)

  net_g.save_synthesizer(os.path.join(hps.model_dir, "synthesizer.pth"))

if __name__ == "__main__":
  main()
