import glob
import sys
import os
import argparse
import pyopenjtalk
import json
import pyworld as pw
import numpy as np
import csv
from scipy.io import wavfile

def get_f0(wav_path, fs = 24000, hop = 128):
    sr, x = wavfile.read(wav_path, fs)
    x = x.astype(np.float64)
    _f0, _time = pw.dio(x, fs)    # 基本周波数の抽出
    f0 = pw.stonemask(x, _f0, _time, fs)  # 基本周波数の修正
    for index, pitch in enumerate(f0):
        #0.
        f0[index] = round(pitch, 1)
    return f0

def f0_to_note(f0, borders):
    print(f0.shape)
    f0 = f0[0::2]
    f0 = f0[0::2]
    print(f0.shape)
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

def get_note_list(border_path):
    note_border = list()
    with open(border_path) as f:
        reader = csv.reader(f)
        for row in reader:
            note_border.append(float(row[0]))
    return note_border

def note2text(note):
    note_text = '-'.join(map(str, map(int, note)))
    return note_text

def get_note_text(wav_path, note_list_path):
    f0 = get_f0(wav_path)
    note_list = get_note_list(note_list_path)
    note = f0_to_note(f0, note_list)
    text = note2text(note)
    return text

def mozi2phone(mozi):
    text = pyopenjtalk.g2p(mozi)
    text = "sil " + text + " sil"
    text = text.replace(' ', '-')
    return text

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

def create_dataset(filename, note_list_path = "note_correspondence.csv"):
    speaker_id = 0
    textful_dir_list = glob.glob("dataset/textful/*")
    textless_dir_list = glob.glob("dataset/textless/*")
    textful_dir_list.sort()
    textless_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    output_file_list_textless = list()
    output_file_list_val_textless = list()
    max_wav_files = 0
    for d in textful_dir_list:
        wav_file_list = glob.glob(d+"/wav/*.wav")
        tmp = len(wav_file_list)
        if max_wav_files < tmp:
            max_wav_files = tmp

    print(max_wav_files)

    for d in textful_dir_list:
        wav_file_list = glob.glob(d+"/wav/*.wav")
        lab_file_list = glob.glob(d + "/text/*.txt")
        wav_file_list.sort()
        lab_file_list.sort()
        if len(wav_file_list) == 0:
            continue
        counter = 0
        end_counter = 0
        val_flag = True
        while True:
            for lab, wav in zip(lab_file_list, wav_file_list):
                with open(lab, 'r', encoding="utf-8") as f:
                    mozi = f.read().split("\n")
                print(str(mozi))
                test = mozi2phone(str(mozi))
                print(test)
                note = get_note_text(wav, note_list_path)
                print(wav + "|"+ str(speaker_id) + "|"+ test + note)
                if counter % 10 != 0:
                    output_file_list.append(wav + "|"+ str(speaker_id) + "|"+ test + "|"+ note + "\n")
                else:
                    if val_flag:
                        output_file_list_val.append(wav + "|"+ str(speaker_id) + "|"+ test + "|"+ note + "\n")
                    else:
                        pass
                
                counter = counter +1
                end_counter = end_counter + 1
                if end_counter == max_wav_files:
                    break
            val_flag = False
            if end_counter == max_wav_files:
                break
        Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")
        speaker_id = speaker_id + 1
        if speaker_id > 108:
            break

    for d in textless_dir_list:
        wav_file_list = glob.glob(d+"/*.wav")
        wav_file_list.sort()
        counter = 0
        for wav in wav_file_list:
            print(wav + "|"+ str(speaker_id) + "|a")
            if counter % 10 != 0:
                output_file_list_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            else:
                output_file_list_val_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            counter = counter +1
        Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")
        speaker_id = speaker_id + 1

    with open('filelists/' + filename + '_textful.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list)
    with open('filelists/' + filename + '_textful_val.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_val)
    with open('filelists/' + filename + '_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_textless)
    with open('filelists/' + filename + '_val_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_val_textless)
    with open('filelists/' + filename + '_Correspondence.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(Correspondence_list)
    return speaker_id

def create_dataset_zundamon(filename, note_list_path = "note_correspondence.csv"):
    textful_dir_list = glob.glob("dataset/textful/*")
    textless_dir_list = glob.glob("dataset/textless/*")
    textful_dir_list.sort()
    textless_dir_list.sort()
    Correspondence_list = list()
    output_file_list = list()
    output_file_list_val = list()
    output_file_list_textless = list()
    output_file_list_val_textless = list()
    #paths
    my_path = "dataset/textful/00_myvoice"
    zundamon_path = "dataset/textful/1205_zundamon"

    #set list wav and text
    #myvoice
    speaker_id = 107
    d = my_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    lab_file_list = glob.glob(d + "/text/*.txt")
    wav_file_list.sort()
    lab_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    for lab, wav in zip(lab_file_list, wav_file_list):
        with open(lab, 'r', encoding="utf-8") as f:
            mozi = f.read().split("\n")
        #print(str(mozi))
        test = mozi2phone(str(mozi))
        #print(test)
        note = get_note_text(wav, note_list_path)
        print(wav + "|"+ str(speaker_id) + "|"+ test + "|"+ note)
        if counter % 10 != 0:
            output_file_list.append(wav + "|"+ str(speaker_id) + "|"+ test + "|"+ note + "\n")
        else:
            output_file_list_val.append(wav + "|"+ str(speaker_id) + "|"+ test + "|"+ note + "\n")
        counter = counter +1
    Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")

    speaker_id = 100
    d = zundamon_path
    wav_file_list = glob.glob(d + "/wav/*.wav")
    lab_file_list = glob.glob(d + "/text/*.txt")
    wav_file_list.sort()
    lab_file_list.sort()
    if len(wav_file_list) == 0:
        print("Error" + d + "/wav に音声データがありません")
        exit()
    counter = 0
    for lab, wav in zip(lab_file_list, wav_file_list):
        with open(lab, 'r', encoding="utf-8") as f:
            mozi = f.read().split("\n")
        #print(str(mozi))
        test = mozi2phone(str(mozi))
        #print(test)
        note = get_note_text(wav, note_list_path)
        print(wav + "|"+ str(speaker_id) + "|"+ test + "|"+ note)
        if counter % 10 != 0:
            output_file_list.append(wav + "|"+ str(speaker_id) + "|"+ test + "|"+ note + "\n")
        else:
            output_file_list_val.append(wav + "|"+ str(speaker_id) + "|"+ test + "|"+ note + "\n")
        counter = counter +1
    Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")

    for d in textless_dir_list:
        wav_file_list = glob.glob(d+"/*.wav")
        wav_file_list.sort()
        counter = 0
        for wav in wav_file_list:
            print(wav + "|"+ str(speaker_id) + "|a")
            if counter % 10 != 0:
                output_file_list_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            else:
                output_file_list_val_textless.append(wav + "|"+ str(speaker_id) + "|a" + "\n")
            counter = counter +1
        Correspondence_list.append(str(speaker_id)+"|"+os.path.basename(d) + "\n")
        speaker_id = speaker_id + 1

    with open('filelists/' + filename + '_textful.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list)
    with open('filelists/' + filename + '_textful_val.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_val)
    with open('filelists/' + filename + '_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_textless)
    with open('filelists/' + filename + '_val_textless.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(output_file_list_val_textless)
    with open('filelists/' + filename + '_Correspondence.txt', 'w', encoding='utf-8', newline='\n') as f:
        f.writelines(Correspondence_list)
    return 109

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, required=True,
                        help='filelist for configuration')
    parser.add_argument('-s', '--sr', type=int, default=24000,
                        help='sampling rate (default = 24000)')
    parser.add_argument('-z', '--zundamon', type=bool, default=False,
                        help='U.N. zundamon Was Her? (default = False)')
    parser.add_argument('-c', '--config', type=str, default="./configs/baseconfig.json",
                        help='JSON file for configuration')
    args = parser.parse_args()
    filename = args.filename
    print(filename)
    if args.zundamon:
        n_spk = create_dataset_zundamon(filename)
    else:
        n_spk = create_dataset(filename)
    
    create_json(filename, n_spk, args.sr, args.config)

if __name__ == '__main__':
    main()
