import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import glob
import sys

import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate
)
from models import (
  SynthesizerTrn
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols

CONFIG_PATH = "./configs/train_config_zundamon.json"
NET_PATH = "./fine_model/G_180000.pth"
MY_VOICE_PATH = "./dataset/textful/00_myvoice/wav"

def mel_loss(spec, audio):
    # 学習と同じやり方でmel spectrogramの誤差を算出
    y_mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)

    y_hat = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
    y_hat = y_hat.float()
    y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1), 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate, 
        hps.data.hop_length, 
        hps.data.win_length, 
        hps.data.mel_fmin, 
        hps.data.mel_fmax)

    loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
    return loss_mel

hps = utils.get_hparams_from_file(CONFIG_PATH)

net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model)
_ = net_g.eval()
_ = utils.load_checkpoint(NET_PATH, net_g, None)

sample_voice_num = 5
dummy_source_speaker_id = 0
eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data)
wav_files = sorted(glob.glob(f"{MY_VOICE_PATH}/*.wav"))
wav_files = wav_files[:sample_voice_num] # 最大10音声
all_data = list()
for wav_file in wav_files:
    data = eval_dataset.get_audio_text_speaker_pair([wav_file, dummy_source_speaker_id, "a"])
    data = TextAudioSpeakerCollate()([data])
    all_data.append(data)

speaker_num = 100
loss_mels = np.zeros(speaker_num + 1)
for target_id in range(0, speaker_num + 1):
    sid_target = torch.LongTensor([target_id])
    print(f"target id: {target_id} / loss mel: ", end="")
    for x, x_lengths, spec, spec_lengths, y, y_lengths, sid_src in all_data:
        result = net_g.voice_conversion(spec, spec_lengths, sid_src=sid_target, sid_tgt=sid_target)
        audio = result[0][0,0].data.cpu().float().numpy()
        loss_mel = mel_loss(spec, audio)
        loss_mels[target_id] += loss_mel
        print(f"{loss_mel:.3f} ", end="")
    loss_mels[target_id] /= len(all_data)
    print(f"/ ave: {loss_mels[target_id]:.3f}")

print("--- Most likely voice ---")
top_losses = np.argsort(loss_mels)[:3]
for target_id in top_losses:
    print(f"target id: {target_id} / ave: {loss_mels[target_id]:.3f}")
