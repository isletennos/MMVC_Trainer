import glob
import os
import argparse
import json
import librosa
import numpy as np
import csv
import torch, torchaudio
import requests

#Global
NUM_MAX_SID = 127
#JVS+Zun+Sora+Me+Tsum+Tsuk+NICT*2 = 107
NUM_TRAINED_SPEAKER = 1
ZUNDAMON_SID = 101
MY_SID = 0
VAL_PER = 20

def get_f0(wav_path, frame_length=2048, win_length=None, hop_length=None):
    y, sr = librosa.load(wav_path)
    #Get f0
    #win_length : If None, defaults to frame_length // 2 hop_length : If None, defaults to frame_length // 4
    #https://librosa.org/doc/main/generated/librosa.pyin.html
    f0, _, _ = librosa.pyin(y, sr = sr, frame_length=frame_length, win_length=win_length, hop_length=hop_length, fmin = librosa.note_to_hz('C2'), fmax= librosa.note_to_hz('C7'))
    f0 = np.nan_to_num(f0)
    return f0

#f0を連続値>カテゴリ(note)
def f0_to_note(f0, borders):
    note = np.zeros(f0.shape)
    borders = borders[1:]

    for i in range(f0.shape[0]):
        for j, border in enumerate(borders):
            if f0[i] < border:
                note[i] = j
                break
            else:
              pass
    return note

#音階リストの読み込み
def get_note_list(border_path):
    note_border = list()
    with open(border_path) as f:
        reader = csv.reader(f)
        for row in reader:
            note_border.append(float(row[0]))
    return note_border

#既存のテキストエンコーダーと同様に読み込みができる所に処理
def note2text(note):
    note_text = '-'.join(map(str, map(int, note)))
    return note_text

#音階を記載
def get_note_text(wav_path, note_list_path):
    f0 = get_f0(wav_path)
    note_list = get_note_list(note_list_path)
    note = f0_to_note(f0, note_list)
    text = note2text(note)
    return text

#textを音素に
def mozi2phone(hubert, wav_path, d_):
    source, sr = torchaudio.load(wav_path)
    source = torchaudio.functional.resample(source, sr, 16000)
    source = source.unsqueeze(0)
    units = hubert.units(source).numpy().squeeze(0)
    filename = os.path.basename(wav_path)
    d = os.path.basename(d_)
    save_path = "units/{}/{}".format(d , filename[:-4])
    os.makedirs("units", exist_ok=True)
    os.makedirs("units/{}".format(d), exist_ok=True)
    np.save(save_path, units)
    return save_path + ".npy"

#filelistの1行を作成する
def create_data_line(wav, note_list_path, speaker_id, d, hubert):
    units = mozi2phone(hubert, wav, d)
    note = get_note_text(wav, note_list_path)
    one_line = wav + "|"+ str(speaker_id) + "|"+ units + "|"+ note + "\n"
    print(one_line)
    return one_line

def create_json(filename, num_speakers, sr, config_path):
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    data['data']['training_files'] = 'filelists/' + filename + '_textful.txt'
    data['data']['validation_files'] = 'filelists/' + filename + '_textful_val.txt'
    data['data']['training_files_notext'] = 'filelists/' + filename + '_textless.txt'
    data['data']['validation_files_notext'] = 'filelists/' + filename + '_val_textless.txt'
    data['data']['sampling_rate'] = sr
    data['data']['n_speakers'] = num_speakers

    with open("./configs/" + filename + ".json", 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def write_configs(lists, filename):
    #推論用の空リスト　そのうち消したい
    output_file_list_textless = list()
    with open('filelists/' + filename + '_textful.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lists[0])
    with open('filelists/' + filename + '_textful_val.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lists[1])
    with open('filelists/' + filename + '_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_textless)
    with open('filelists/' + filename + '_Correspondence.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(lists[2])

def create_dataset(filename, note_list_path = "note_correspondence.csv"):
    #空白話者の先頭からスタート
    speaker_id = NUM_TRAINED_SPEAKER
    #list回りの宣言とソート
    textful_dir_list = glob.glob("dataset/**/")
    textful_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")

    for d in textful_dir_list:
        wav_file_list = glob.glob(d+"/wav/*.wav")
        wav_file_list.sort()
        if len(wav_file_list) == 0:
            continue
        counter = 0
        for wav in wav_file_list:
            one_line = create_data_line(wav, note_list_path, speaker_id, d, hubert)
            if counter % VAL_PER != 0:
                output_file_list.append(one_line)
            else:
                output_file_list_val.append(one_line)
            counter = counter +1
        Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")
        speaker_id = speaker_id + 1
        if speaker_id > NUM_MAX_SID:
            break
    
    lists = [output_file_list, output_file_list_val, Correspondence_list]
    write_configs(lists, filename)
    return NUM_MAX_SID

def create_dataset_zundamon(filename, note_list_path = "note_correspondence.csv"):
    #list回りの宣言とソート
    textful_dir_list = glob.glob("dataset/**/")
    textful_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")
    #paths
    my_path = "dataset/00_myvoice"
    zundamon_path = "dataset/1205_zundamon"

    #set list wav and text
    #myvoice
    speaker_id = MY_SID
    d = my_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    wav_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    for wav in wav_file_list:
        one_line = create_data_line(wav, note_list_path, speaker_id, d, hubert)
        if counter % VAL_PER != 0:
            output_file_list.append(one_line)
        else:
            output_file_list_val.append(one_line)
        counter = counter +1
    Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")

    speaker_id = ZUNDAMON_SID
    d = zundamon_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    wav_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    for wav in wav_file_list:
        one_line = create_data_line(wav, note_list_path, speaker_id, d, hubert)
        if counter % VAL_PER != 0:
            output_file_list.append(one_line)
        else:
            output_file_list_val.append(one_line)
        counter = counter +1
    Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")

    lists = [output_file_list, output_file_list_val, Correspondence_list]
    write_configs(lists, filename)
    return NUM_MAX_SID

def create_dataset_character(filename, tid, note_list_path = "note_correspondence.csv"):
    #list回りの宣言とソート
    textful_dir_list = glob.glob("dataset/**/")
    textful_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")
    #paths
    my_path = "dataset/textful/00_myvoice"
    target_path = "dataset/textful/01_target"

    #set list wav and text
    #myvoice
    speaker_id = MY_SID
    d = my_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    wav_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    for wav in wav_file_list:
        one_line = create_data_line(wav, note_list_path, speaker_id, d, hubert)
        if counter % VAL_PER != 0:
            output_file_list.append(one_line)
        else:
            output_file_list_val.append(one_line)
        counter = counter +1
    Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")

    speaker_id = tid
    d = target_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    wav_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    for wav in wav_file_list:
        one_line = create_data_line(wav, note_list_path, speaker_id, d, hubert)
        if counter % VAL_PER != 0:
            output_file_list.append(one_line)
        else:
            output_file_list_val.append(one_line)
        counter = counter +1
    Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")

    lists = [output_file_list, output_file_list_val, Correspondence_list]
    write_configs(lists, filename)
    return NUM_MAX_SID

def create_dataset_multi_character(filename, file_path, note_list_path = "note_correspondence.csv"):
    #list回りの宣言とソート
    textful_dir_list = glob.glob("dataset/**/")
    textful_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")

    with open(file_path, "r") as f:
        for line in f.readlines():
            target_dir , sid = line.split("|")
            sid = sid.rstrip('\n')
            wav_file_list = glob.glob("dataset/" + target_dir + "/wav/*.wav")
            wav_file_list.sort()
            if len(wav_file_list) == 0:
                print("Error" + target_dir + "/wav に音声データがありません")
                exit()
            counter = 0
            for wav in wav_file_list:
                one_line = create_data_line(wav, note_list_path, sid, target_dir, hubert)
                if counter % VAL_PER != 0:
                    output_file_list.append(one_line)
                else:
                    output_file_list_val.append(one_line)
                counter = counter +1
            Correspondence_list.append(str(sid)+"|"+ target_dir + "\n")

    lists = [output_file_list, output_file_list_val, Correspondence_list]
    write_configs(lists, filename)
    return NUM_MAX_SID

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True,
                        help='filelist for configuration')
    parser.add_argument('-s', '--sr', type=int, default=24000,
                        help='sampling rate (default = 24000)')
    parser.add_argument('-t', '--target', type=int, default=9999,
                        help='pre_traind targetid (zundamon = 101, sora = 102, methane = 103, tsumugi = 104)')
    parser.add_argument('-m', '--multi_target', type=str, default=None,
                        help='pre_traind targetid (zundamon = 101, sora = 102, methane = 103, tsumugi = 104)')
    parser.add_argument('-c', '--config', type=str, default="./configs/baseconfig.json",
                        help='JSON file for configuration')
    args = parser.parse_args()
    filename = args.filename
    print(filename)
    if args.multi_target != None:
        n_spk = create_dataset_multi_character(filename, args.multi_target)
    elif args.target != 9999 and args.target == ZUNDAMON_SID:
        n_spk = create_dataset_zundamon(filename)
    elif args.target != 9999:
        n_spk = create_dataset_character(filename, args.target)
    else:
        n_spk = create_dataset(filename)
    
    create_json(filename, n_spk, args.sr, args.config)

if __name__ == '__main__':
    main()