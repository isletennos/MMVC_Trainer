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
https://www.nicovideo.jp/watch/sm40292061

## 利用規約(2022/04/20)
本ソフトウェアの利用規約は基本的にMITライセンスに準拠します。  
VRCでの利用などライセンス記載が不可の場合、記載は不要です。  
ライセンスの記載が可能なプラットフォームでの利用の場合、下記クレジットどちらかををご利用ください。  
(可能であればパターン2を使ってくれると製作者はうれしいです)  

ライセンスパターン 1　
```
Copyright (c) 2021 Isle.Tennos　
Released under the MIT license　
https://opensource.org/licenses/mit-license.php
```

ライセンスパターン 2　
```
MMVCv1.x.x(使用バージョン)　
Copyright (c) 2021 Isle.Tennos　
Released under the MIT license　
https://opensource.org/licenses/mit-license.php
git:https://github.com/isletennos/MMVC_Trainer
community(discord):https://discord.gg/PgspuDSTEc
```
## Requirement
・Google アカウント
## Install
このリポジトリをダウンロードして、展開、展開したディレクトリをgoogle drive上にアップロードしてください。
## Usage
### チュートリアル : ずんだもんになる
#### Ph1. 自分の音声の録音と音声データの配置
1. 自分の声の音声データを録音します。  
JVSコーパスやITAコーパス等を台本にし、100文程度読み上げます。  
音声の録音ツールは  
Audacity  
https://forest.watch.impress.co.jp/library/software/audacity/  
OREMO  
http://nwp8861.web.fc2.com/soft/oremo/  
等があります。  
また、録音した音声は**24000Hz 16bit 1ch**である必要があります。  
※MMVC用にテキストを分割したITAコーパスです。ご利用ください。  
https://drive.google.com/file/d/14oXoQqLxRkP8NJK8qMYGee1_q2uEED1z/view?usp=sharing

2. dataset/textful/000_myvoice に音声データとテキストデータを配置します。 
最終的に下記のようなディレクトリ構成になります。  
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
│   │── 001_target
│   │   ├── text
│   │   └── wav
│   │
│   └── 1205_zundamon
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

#### Ph2. モデルの学習方法
1. 下記リンクより、「G_180000.pth」「D_180000.pth」をダウンロード。 
https://drive.google.com/drive/folders/1XGpW0loNr1KjMtXVVG3WRd47R_ne6_X2?usp=sharing
2. 「G_180000.pth」「D_180000.pth」をfine_modelに配置します。(良く忘れるポイントなので要注意！)  
3. notebookディレクトリにある「Create_Configfile_zundamon.ipynb」をgoogle colab 上で実行、学習に必要なconfigファイルを作成します  
4. configsに作成されたtrain_config_zundamon.jsonの  
 
      - "eval_interval"   
        modelを保存する間隔です。
      - "batch_size"   
        colabで割り当てたGPUに合わせて調整してください。

    上記2項目を環境に応じて最適化してください。わからない方はそのままで大丈夫です。  

5. notebookディレクトリにある「Train_MMVC.ipynb」をgoogle colab 上で実行してください。  
    logs/にモデルが生成されます。

#### Ph3. 学習したモデルの性能検証
1. notebookディレクトリにある「MMVC_Interface.ipynb」をgoogle colab 上で実行してください。
### 好きなキャラクターの声になる
#### Ph1. 自分の音声の録音と音声データの配置 及びターゲット音声データの配置
1. 自分の声の音声データとその音声データに対応するテキスト、変換したい声の音声データとその音声データに対応するテキストを用意します。    
この時、用意する音声(自分の声の音声データ/変換したい声の音声データ共に)は**24000Hz 16bit 1ch**を強く推奨しております。  

    九州そらと四国めたんのMMVC用のデータは下記リンクからダウンロードください。  
    ダウンロード後、2節のように音声データとテキストデータを配置してください。   
    https://drive.google.com/drive/folders/1ClIUx_2Wv-uNnuW2LlfG7aTHrUaZ2Asx?usp=sharing

2. 下記のようなディレクトリ構成になるように音声データとテキストデータを配置します。  
    textfulの直下には2ディレクトリになります。  
    (1205_zundamonディレクトリは無くても問題ありません) 

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
│   │── 001_target
│   │   ├── text
│   │   │   ├── t_voice_001.txt
│   │   │   ├── t_voice_002.txt
│   │   │   ├── ...
│   │   └── wav
│   │        ├── t_voice_001.wav
│   │        ├── t_voice_002.wav
│   │        ├── ... 
│   └── 1205_zundamon
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
#### Ph2. モデルの学習方法
以降、「チュートリアル : ずんだもんになる Ph2.」と同様のため割愛  
#### Ph3. 学習したモデルの性能検証
以降、「チュートリアル : ずんだもんになる Ph3.」と同様のため割愛  
## Q&A
下記サイトをご参考ください。  
https://mmvc.readthedocs.io/ja/latest/index.html
## Note
なにか不明点があればお気軽にご連絡ください。
## MMVCコミュニティサーバ(discord)
開発の最新情報や、不明点のお問合せ、MMVCの活用法などMMVCに関するコミュニティサーバです。  
https://discord.gg/PgspuDSTEc

## ITAコーパス マルチモーダルデータベースについて 
本ソフトウェアで再配布されている  
・ずんだもん(伊藤ゆいな)  
・四国めたん(田中小雪)  
・九州そら(西田望見)  
の音声データの著作物の権利はSSS合同会社様にあります。  
本ソフトウェアではSSS合同会社様に許可を頂き、音声データを本ソフトウェア用に改変、再配布を行っております。  
上記キャラクターの音声を利用する際にはSSS合同会社様の利用規約に同意する必要があります。  
https://zunko.jp/multimodal_dev/login.php

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

## Reference
https://arxiv.org/abs/2106.06103  
https://github.com/jaywalnut310/vits

## Author
Isle Tennos  
Twitter : https://twitter.com/IsleTennos

