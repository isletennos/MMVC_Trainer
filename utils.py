import os
import glob
import sys
import argparse
import logging
import json
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch
import wave
import csv
from mel_processing import spec_to_mel_torch_data
import warnings

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging


def load_checkpoint(checkpoint_path, model, optimizer=None):
  assert os.path.isfile(checkpoint_path), f"No such file or directory: {checkpoint_path}"
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  iteration = checkpoint_dict['iteration']
  learning_rate = checkpoint_dict['learning_rate']
  if optimizer is not None:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict['model']
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  new_state_dict= {}
  for k, v in state_dict.items():
    try:
      new_state_dict[k] = saved_state_dict[k]
    except:
      logger.info("%s is not in the checkpoint" % k)
      new_state_dict[k] = v
  if hasattr(model, 'module'):
    model.module.load_state_dict(new_state_dict)
  else:
    model.load_state_dict(new_state_dict)
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
  if hasattr(model, 'module'):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  torch.save({'model': state_dict,
              'iteration': iteration,
              'optimizer': optimizer.state_dict(),
              'learning_rate': learning_rate}, checkpoint_path)


def save_vc_sample(hps, loader, collate, generator, name):
  if not hasattr(hps.others, "input_filename"):
    return

  input_filename = hps.others.input_filename
  source_id = hps.others.source_id
  target_ids = hps.others.target_id

  if type(target_ids) != list:
    target_ids = [target_ids]

  dataset = loader("", hps.data, no_use_textfile=True, disable_tqdm=True)
  data = dataset.get_audio_text_speaker_pair([input_filename, source_id, "a", "0"])
  data = collate()([data])
  x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src, note, note_lengths = [x.cuda(0) for x in data]
  mel = spec_to_mel_torch_data(spec, hps.data)
  if hps.model.use_mel_train:
    spec = mel
  for target_id in target_ids:
    with torch.no_grad():
      sid_tgt = torch.LongTensor([target_id]).cuda(0)
      audio = generator.module.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][0,0].data.cpu().float().numpy()

    audio = audio * hps.data.max_wav_value
    wav = audio.astype(np.int16).tobytes()

    output_filename = os.path.join(hps.model_dir, f"vc_{target_id}_{name}.wav")
    with wave.open(output_filename, 'wb') as fh:
      fh.setnchannels(1)
      fh.setsampwidth(2)
      fh.setframerate(hps.data.sampling_rate)
      fh.writeframes(wav)


def save_best_log(best_log_path, global_step, loss_mel_value, date):
    with open(best_log_path, "a") as f:
       writer = csv.writer(f)
       writer.writerow([global_step, loss_mel_value, date])


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
  x = f_list[-1]
  print(x)
  return x


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path):
  #音声にメタデータが含まれる際のWavFileWarning対策
  with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text


def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')
  parser.add_argument('-fg', '--fine_tuning_g', type=str, default=None,
                      help='If fine tuning, please specify model(G)')
  parser.add_argument('-fd', '--fine_tuning_d', type=str, default=None,
                      help='If fine tuning, please specify model(D)')
  
  args = parser.parse_args()
  model_dir = os.path.join("./logs", args.model)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  best_log_path = os.path.join(model_dir, "best.log")
  if not os.path.exists(best_log_path):
    with open(best_log_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss g/mel", "date"])
  with open(best_log_path, "r") as f:
    reader = csv.reader(f)
    last_row = list(reader)[-1]
  try:
    best_loss_mel = float(last_row[1])
  except:
    best_loss_mel = 9999

  config_path = args.config
  config_save_path = os.path.join(model_dir, "config.json")
  if init:
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)

  #Added about fine tuning
  if args.fine_tuning_g != None and args.fine_tuning_d != None:
    config['fine_flag'] = True
    config['fine_model_g'] = args.fine_tuning_g
    config['fine_model_d'] = args.fine_tuning_d
  else:
    config['fine_flag'] = False
  
  hparams = HParams(**config)
  hparams.model_dir = model_dir
  hparams.best_log_path = best_log_path
  hparams.best_loss_mel = best_loss_mel
  return hparams


def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams


def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warn("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      logger.warn("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
