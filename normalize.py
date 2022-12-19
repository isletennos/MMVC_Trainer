import shutil
import os
import glob
from pydub import AudioSegment, silence
import sys
import tqdm
# -16 + -6
NORMALIZE_dBFS = -22
#50ms
SILENCE_THRESHOLD_MS = 50
SAMPLING_RATE = 24000
SILENCE_THRESHOLD = -45
#all_rm_silent param
MIN_SILENCE_LEN = 100
KEEP_SILENCE = 50

def all_rm_silent(sound):
    audio_chunks = silence.split_on_silence(sound
                                ,min_silence_len = MIN_SILENCE_LEN
                                ,silence_thresh = SILENCE_THRESHOLD
                                ,keep_silence = KEEP_SILENCE
                            )

    # Putting the file back together
    combined = AudioSegment.empty()
    for chunk in audio_chunks:
        combined += chunk
    return combined

def normalize_audio(file_name):
    sound = AudioSegment.from_file(file_name, "wav")
    #To monaural
    sound = sound.set_channels(1)
    #To 24000Hz
    sound = sound.set_frame_rate(SAMPLING_RATE)

    # silence[[0, T]]
    start_silent = silence.detect_leading_silence(sound, silence_threshold=SILENCE_THRESHOLD)
    # 開始の無音は0.05秒以下に
    start_silent = start_silent - SILENCE_THRESHOLD_MS
    if start_silent > 0:
        sound = sound[start_silent:]
    # 終わりの無音も0.05秒以下に
    # silence[[0, T]]
    end_silent = silence.detect_leading_silence(sound.reverse(), silence_threshold=SILENCE_THRESHOLD)
    end_silent = end_silent - SILENCE_THRESHOLD_MS
    if end_silent > 0:
        sound = sound.reverse()[end_silent:]
        sound = sound.reverse()

    #無音を削除して、dbFSを計算
    source_dBFS = all_rm_silent(sound).dBFS
    #normalize dBFS -6
    change_in_dBFS = (NORMALIZE_dBFS) - source_dBFS
    normalized_sound = sound.apply_gain(change_in_dBFS)
    #16bitへ
    normalized_sound = normalized_sound.set_sample_width(2)
    #output
    normalized_sound.export(file_name, format="wav")

def convert_audio_mmvc_format(back_up = True):
    dataset_dir_list = glob.glob("dataset/**/")
    for d in tqdm.tqdm(dataset_dir_list):
        d = d[:-1]
        wav_file_list = glob.glob(d+"/wav/*.wav")
        if len(wav_file_list) == 0:
            print("{} dir is 0 wav files".format(d))
            continue
        if back_up:
            os.makedirs(d+"/wav/back_up", exist_ok=True)
        for wav in tqdm.tqdm(wav_file_list):
            if back_up:
                shutil.copyfile(wav, "{}/back_up/{}".format(os.path.dirname(wav), os.path.basename(wav)))
            normalize_audio(wav)

if __name__ == '__main__':
    args = sys.argv
    print(args[1])
    convert_audio_mmvc_format(args[1])