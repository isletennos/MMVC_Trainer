import logging
logging.getLogger("h5py").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)
import time
import os
import random
import numpy as np
import torch
import torch.utils.data
import tqdm
import csv
import pyworld as pw

import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text

#add
from retry import retry
import random
import torchaudio

from sifigan.utils import dilated_factor
from sifigan.utils.features import SignalGenerator

"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams, augmentation=False, augmentation_params=None, disable_tqdm = False):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.augmentation = augmentation
        if augmentation :
            self.gain_p = augmentation_params.gain_p
            self.min_gain_in_db = augmentation_params.min_gain_in_db
            self.max_gain_in_db = augmentation_params.max_gain_in_db
            self.time_stretch_p = augmentation_params.time_stretch_p
            self.min_rate = augmentation_params.min_rate
            self.max_rate = augmentation_params.max_rate
            self.pitch_shift_p = augmentation_params.pitch_shift_p
            self.min_semitones = augmentation_params.min_semitones
            self.max_semitones = augmentation_params.max_semitones
        
        self.disable_tqdm = disable_tqdm

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        
        for audiopath, sid, text, f0, cf0, _ in tqdm.tqdm(self.audiopaths_sid_text, disable=self.disable_tqdm):
            audiopaths_sid_text_new.append([audiopath, sid, text, f0, cf0])
            #ファイルのサイズをバイト単位で取得
            lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        

    def get_audio_text_speaker_pair(self, audiopath_sid_text):
        # separate filename, speaker_id and text
        audiopath, sid, text, f0, cf0 = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2], audiopath_sid_text[3], audiopath_sid_text[4]
        text = self.get_text(text)
        #DA、音量とピッチのみ機能再開、話速は使わないこと
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        f0 = self.get_f0(f0)
        cf0 = self.get_f0(cf0)
        return (text, spec, wav, sid, f0, cf0)
        
    @retry(exceptions=(PermissionError), tries=100, delay=10)
    def get_audio(self, filename):
        # 音声データは±1.0内に正規化したtorchベクトルでunsqueeze(0)で外側1次元くるんだものを扱う
        audio, sampling_rate = load_wav_to_torch(filename)
        try:
            if sampling_rate != self.sampling_rate:
                raise ValueError("[Error] Exception: source {} SR doesn't match target {} SR".format(
                    sampling_rate, self.sampling_rate))
        except ValueError as e:
            print(e)
            exit()
        audio_norm = self.get_normalized_audio(audio, self.max_wav_value)

        if self.augmentation:
            audio_augmented = self.add_augmentation(audio_norm, sampling_rate)
            # ノーマライズ後のaugmentationとnoise付加で範囲外になったところを削る
            # -1～1に丸める
            audio_augmented = torch.clamp(audio_augmented, -1, 1) 
            audio_norm = audio_augmented
            # spec(スペクトログラム)は入力信号となるのでaugmentationしてさらにノイズを付加したものを使用
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
        return spec, audio_norm

    def add_augmentation(self, audio, sampling_rate):
        gain_in_db = 0.0
        if random.random() <= self.gain_p:
            gain_in_db = random.uniform(self.min_gain_in_db, self.max_gain_in_db)
        time_stretch_rate = 1.0
        if random.random() <= self.time_stretch_p:
            time_stretch_rate = random.uniform(self.min_rate, self.max_rate)
        pitch_shift_semitones = 0
        if random.random() <= self.pitch_shift_p:
            pitch_shift_semitones = random.uniform(self.min_semitones, self.max_semitones) * 100 # 1/100 semitone 単位指定のため
        augmentation_effects = [
            ["gain",  f"{gain_in_db}"],
            ["tempo", f"{time_stretch_rate}"],
            ["pitch", f"{pitch_shift_semitones}"],
            ["rate",  f"{sampling_rate}"]
        ]
        audio_augmented, _ = torchaudio.sox_effects.apply_effects_tensor(audio, sampling_rate, augmentation_effects)
        return audio_augmented

    def get_normalized_audio(self, audio, max_wav_value):
        audio_norm = audio / max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        return audio_norm

    def get_text(self, text):
        text_norm = torch.FloatTensor(np.load(text))
        return text_norm

    def get_f0(self, f0):
        f0_tensor = torch.FloatTensor(np.load(f0))
        return f0_tensor

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index])

    def __len__(self):
        return len(self.audiopaths_sid_text)

    def get_all_sid(self):
        return list(set([int(r[1]) for r in self.audiopaths_sid_text]))

class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(
        self, 
        sample_rate,
        segment_size,
        hop_size,
        df_f0_type="cf0",
        dense_factors=[0.5, 1, 4, 8],
        upsample_scales=[8, 4, 2, 2],
        sine_amp=0.1,
        noise_amp=0.003,
        sine_f0_type="cf0",
        signal_types=["sine"],
        train = True,
        f0_factor = 1.0,
        return_ids=False,
        no_text = False):
        self.return_ids = return_ids
        self.no_text = no_text
        self.hop_size = hop_size
        self.segment_size = segment_size
        self.max_frames = segment_size // hop_size
        self.df_f0_type = df_f0_type
        self.dense_factors = dense_factors
        self.prod_upsample_scales = np.cumprod(upsample_scales)
        self.sample_rate = sample_rate
        self.sine_f0_type = sine_f0_type
        self.signal_generator = SignalGenerator(
            sample_rate=sample_rate,
            hop_size=hop_size,
            sine_amp=sine_amp,
            noise_amp=noise_amp,
            signal_types=signal_types,
        )
        self.train = train
        self.f0_factor = f0_factor


    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid, note]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        sid = torch.LongTensor(len(batch))
        f0_lengths = torch.LongTensor(len(batch))
        cf0_lengths = torch.LongTensor(len(batch))
        slice_id = torch.LongTensor(len(batch))

        text_padded = torch.FloatTensor(len(batch), max_text_len, 256)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        
        # row wav は sliceした形でしか使わない 
        wav_padded = torch.FloatTensor(len(batch), 1, self.segment_size)
        # f0 も同上
        if self.train:
            f0_padded = torch.FloatTensor(len(batch), 1, self.max_frames)
            cf0_padded = torch.FloatTensor(len(batch), 1, self.max_frames)
        # 推論時の処理
        else:
            tmp = batch[0]
            f0_padded = torch.FloatTensor(len(batch), 1, tmp[4].size(0))
            cf0_padded = torch.FloatTensor(len(batch), 1, tmp[5].size(0))
        
        #返り値の初期化
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        f0_padded.zero_()
        cf0_padded.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            #Slice
            start_frame = np.random.randint(0, (row[1].size(1) - self.max_frames))
            start_step = start_frame * self.hop_size
            slice_id[i] = start_frame

            text = row[0]
            text_padded[i, :text.shape[0], :] = text
            text_lengths[i] = text.shape[0]

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            #Slice した音声を格納
            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav[:,start_step:start_step + self.segment_size]
            wav_lengths[i] = wav.size(1)

            sid[i] = row[3]

            #学習時 f0/cf0をSliceして格納、dilated_factorを求める
            if self.train:
                f0 = row[4]
                f0_padded[i, :, :f0.size(0)] = f0[start_frame : start_frame + self.max_frames]
                f0_lengths[i] = f0.size(0)

                cf0 = row[5]
                cf0_padded[i, :, :cf0.size(0)] = cf0[start_frame : start_frame + self.max_frames]
                cf0_lengths[i] = cf0.size(0)

            #推論時 f0/cf0にf0の倍率を乗算してf0/cf0を求める
            else:
                f0 = row[4] * self.f0_factor
                f0_padded[i, :, :f0.size(0)] = f0
                f0_lengths[i] = f0.size(0)

                cf0 = row[5] * self.f0_factor
                cf0_padded[i, :, :cf0.size(0)] = cf0
                cf0_lengths[i] = cf0.size(0)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, ids_sorted_decreasing, f0_padded, f0_lengths, cf0_padded, cf0_lengths, slice_id
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sid, f0_padded, f0_lengths, cf0_padded, cf0_lengths, slice_id


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)

        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
  
        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))
  
        batches = []
        for i in range(len(self.buckets)):
            next_bucket = (i+1) % len(self.buckets)
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]

            if len_bucket == 0:
              print("[Warn] Exception: length of buckets {} is 0. ID:{} Skip.".format(i,i))
              continue

            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
    
            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]
    
            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
                batches.append(batch)
  
        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
  
        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)
    
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
