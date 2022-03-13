RT-MMVC_Trainer
====

AIを使ったリアルタイムボイスチェンジャーのモデル学習用ツール

## Description
AIを使ったリアルタイムボイスチェンジャー「RT-MMVC(RealTime-Many to Many Voice Conversion)」  
で使用するモデルを学習するためのリポジトリです。  
google colaboratoryを用いることで、個人の環境に依存せず、かつ簡単に機械学習の学習フェーズを実行可能です。
## concept
「簡単」「だれでも」
## Demo
作成中
## Requirement
・Google アカウント
## Install
このリポジトリをダウンロードして、展開、展開したディレクトリをgoogle drive上にアップロードしてください。
## Usage
### 学習用のデータセットの作成および配置
1. 学習用の話者の音声データとその音声データに対応するテキストを用意します。
    下記の公開コーパス等を利用することを推奨します。
    - JVS (Japanese versatile speech) corpus  
      contributors : 高道 慎之介様/三井 健太郎様/齋藤 佑樹様/郡山 知樹様/丹治 尚子様/猿渡 洋様  
      https://sites.google.com/site/shinnosuketakamichi/research-topics/jvs_corpus  


   	- つくよみちゃんコーパス  
      contributor : 夢前黎様  
      https://tyc.rei-yumesaki.net/material/corpus/  

2.下記のようなディレクトリ構成になるように音声データとテキストデータを配置します。
```
dataset
├── textful
│   ├── 000_jvs001
│   │   ├── text
│   │   │   ├── VOICEACTRESS100_001.txt
│   │   │   ├── VOICEACTRESS100_002.txt
│   │   │   ├── ...
│   │   └── wav
│   │        ├── VOICEACTRESS100_001.wav
│   │        ├── VOICEACTRESS100_002.wav
│   │        ├── ...
│   ├── 001_jvs002
│   │   ├── text
│   │   │   ├── VOICEACTRESS100_001.txt
│   │   │   ├── VOICEACTRESS100_002.txt
│   │   │   ├── ...
│   │   └── wav
│   │        ├── VOICEACTRESS100_001.wav
│   │        ├── VOICEACTRESS100_002.wav
│   │        ├── ...
│   ├── 002_jvs003
│   │   ├── text
│   │   │   ├── VOICEACTRESS100_001.txt
│   │   │   ├── VOICEACTRESS100_002.txt
│   │   │   ├── ...
│   │   └── wav
│   │        ├── VOICEACTRESS100_001.wav
│   │        ├── VOICEACTRESS100_002.wav
│   │        ├── ...
│   ├── ...
│   │        
│   │        
│   │        
│ 
│ 
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

3. notebookディレクトリにある「Train_RT-MMVC.ipynb」をgoogle colab 上で実行してください。  
    logs/にモデルが生成されます。
### 学習したモデルの性能検証
1. notebookディレクトリにある「RT-MMVC_Interface.ipynb」をgoogle colab 上で実行
### Q&A
順次更新
## Note
なにか不明点があればお気軽にご連絡ください。
## Reference
https://arxiv.org/abs/2106.06103  
https://github.com/jaywalnut310/vits
## Author
Isle Tennos  
Twitter : https://twitter.com/IsleTennos

