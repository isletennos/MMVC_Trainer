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
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from text.symbols import symbols

torch.backends.cudnn.benchmark = True
global_step = 0


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '8000'

  hps = utils.get_hparams()

  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step
  
  if hps.others.os_type == "windows":
    backend_type = "gloo"
    parallel = DP
  else: # Colab
    backend_type = "nccl"
    parallel = DDP

  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  cpu_count = os.cpu_count()
  if cpu_count > 8:
    cpu_count = 8

  if not hasattr(hps.model, "use_mel_train"):
    hps.model.use_mel_train = False

  if hasattr(hps.others, "input_filename"):
    assert os.path.isfile(hps.others.input_filename), "VC test file does not exist."
    assert hasattr(hps.others, "source_id") and hasattr(hps.others, "target_id"), "VC source_id and target_id are required."

  dist.init_process_group(backend=backend_type, init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)
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
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = parallel(net_g, device_ids=[rank])
  net_d = parallel(net_d, device_ids=[rank])

  logger.info('FineTuning : '+str(hps.fine_flag))
  if hps.fine_flag:
      logger.info('Load model : '+str(hps.fine_model_g))
      logger.info('Load model : '+str(hps.fine_model_d))
      _, _, _, global_step = utils.load_checkpoint(hps.fine_model_g, net_g, optim_g)
      _, _, _, global_step = utils.load_checkpoint(hps.fine_model_d, net_d, optim_d)
      epoch_str = 1
      global_step = 0

  else:
    try:
      _, _, _, global_step = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*[0-9].pth"), net_g, optim_g)
      _, _, _, global_step = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*[0-9].pth"), net_d, optim_d)
    except:
      epoch_str = 1
      global_step = 0

  net_g.module.save_synthesizer(os.path.join(hps.model_dir, "synthesizer.pth"))

if __name__ == "__main__":
  main()
