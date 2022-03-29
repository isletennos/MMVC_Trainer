MMVC_Trainer
====

AIを使ったリアルタイムボイスチェンジャーのモデル学習用ツール

## Description
AIを使ったリアルタイムボイスチェンジャー「MMVC(RealTime-Many to Many Voice Conversion)」  
で使用するモデルを学習するためのリポジトリです。  
google colaboratoryを用いることで、個人の環境に依存せず、かつ簡単に機械学習の学習フェーズを実行可能です。  
## MMVC_Client
MMVCを実際に動かすClient software  
https://github.com/isletennos/MMVC_Client
## concept
「簡単」「だれでも」「好きな声に」「リアルタイムで」
## Demo
作成中
## Requirement
・Google アカウント
## Install
このリポジトリをダウンロードして、展開、展開したディレクトリをgoogle drive上にアップロードしてください。
## Usage
### 学習用のデータセットの作成および配置
1. 自分の声の音声データとその音声データに対応するテキスト、変換したい声の音声データとその音声データに対応するテキストを用意します。  
2. 下記のようなディレクトリ構成になるように音声データとテキストデータを配置します。  
    textfulの直下には2ディレクトリになります。  
```
dataset
├── textful
│   ├── 000_myvoice
│   │   ├── text
│   │   │   ├── s_voice_001.txt
│   │   │   ├── s_voice_002.txt
│   │   │   ├── ...
│   │   └── wav
│   │        ├── s_voice_001.wav
│   │        ├── s_voice_002.wav
│   │        ├── ...
│   └── 001_target
│       ├── text
│       │   ├── t_voice_001.txt
│       │   ├── t_voice_002.txt
│       │   ├── ...
│       └── wav
│            ├── t_voice_001.wav
│            ├── t_voice_002.wav
│            ├── ...      
│        
└── textless
```
### モデルの学習方法
1. notebookディレクトリにある「Create_Configfile.ipynb」をgoogle colab 上で実行、学習に必要なconfigファイルを作成
2. 学習したコンフィグファイル(json)の
 
      - "eval_interval"   
        modelを保存する間隔です。
      - "batch_size"   
        colabで割り当てたGPUに合わせて調整してください。

    上記2項目を環境に応じて設定ください。

3. notebookディレクトリにある「Train_MMVC.ipynb」をgoogle colab 上で実行してください。  
    logs/にモデルが生成されます。
### 学習したモデルの性能検証
1. notebookディレクトリにある「MMVC_Interface.ipynb」をgoogle colab 上で実行
## Q&A
Q1. 下記のようなエラーが発生しました。どうしたらよいですか？  
```
Traceback (most recent call last):
  File "train_ms.py", line 302, in <module>
    main()
  File "train_ms.py", line 50, in main
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
  File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 200, in spawn
    return start_processes(fn, args, nprocs, join, daemon, start_method='spawn')
  File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 158, in start_processes
    while not context.join():
  File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 119, in join
    raise Exception(msg)
Exception: 

-- Process 0 terminated with the following error:
Traceback (most recent call last):
  File "/usr/local/lib/python3.7/dist-packages/torch/multiprocessing/spawn.py", line 20, in _wrap
    fn(i, *args)
  File "/content/drive/MyDrive/MMVC_Trainer/train_ms.py", line 126, in run
    train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
  File "/content/drive/MyDrive/MMVC_Trainer/train_ms.py", line 172, in train_and_evaluate
    hps.data.mel_fmax
  File "/content/drive/MyDrive/MMVC_Trainer/mel_processing.py", line 105, in mel_spectrogram_torch
    center=center, pad_mode='reflect', normalized=False, onesided=True)
  File "/usr/local/lib/python3.7/dist-packages/torch/functional.py", line 465, in stft
    return _VF.stft(input, n_fft, hop_length, win_length, window, normalized, onesided)
RuntimeError: cuFFT doesn't support signals of half type with compute capability less than SM_53, but the device containing input half tensor only has SM_37
```
A1. お手数ですがコンフィグファイル(json)の  
"fp16_run": true,  
を  
"fp16_run": false,  
に変更ください。


順次更新
## Note
なにか不明点があればお気軽にご連絡ください。
## Reference
https://arxiv.org/abs/2106.06103  
https://github.com/jaywalnut310/vits

## Special thanks
- JVS (Japanese versatile speech) corpus  
  contributors : 高道 慎之介様/三井 健太郎様/齋藤 佑樹様/郡山 知樹様/丹治 尚子様/猿渡 洋様  
  https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus  

- ITAコーパス マルチモーダルデータベース  
  contributors : 金井郁也様/千葉隆壱様/齊藤剛史様/森勢将雅様/小口純矢様/能勢隆様/尾上真惟子様/小田恭央様  
  CharacterVoice : 東北イタコ(木戸衣吹様)/ずんだもん(伊藤ゆいな様)/四国めたん(田中小雪様)  
  https://zunko.jp/multimodal_dev/login.php  

- つくよみちゃんコーパス  
  contributor : 夢前黎様  
  CharacterVoice : つくよみちゃん(夢前黎様)  
  https://tyc.rei-yumesaki.net/material/corpus/  

## Author
Isle Tennos  
Twitter : https://twitter.com/IsleTennos

