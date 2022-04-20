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
## Requirement
・Google アカウント
## Install
このリポジトリをダウンロードして、展開、展開したディレクトリをgoogle drive上にアップロードしてください。
## Usage
### 学習用のデータセットの作成および配置
1. 自分の声の音声データとその音声データに対応するテキスト、変換したい声の音声データとその音声データに対応するテキストを用意します。  
この時、用意する音声(自分の声の音声データ/変換したい声の音声データ共に)は**24000Hz 16bit 1ch**を強く推奨しております。  

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
1. 下記リンクより、「G_232000.pth」「D_232000.pth」をダウンロード。
https://drive.google.com/drive/u/8/folders/1ZZ1tTPuXtwWZugJiMCAjvlz-xPdLQV6M
2. 「G_232000.pth」「D_232000.pth」をfine_modelに移動。
3. notebookディレクトリにある「Create_Configfile.ipynb」をgoogle colab 上で実行、学習に必要なconfigファイルを作成
4. 学習したコンフィグファイル(json)の
 
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
下記のリンクをご参考ください。
それでも解決しない場合はお気軽にコミュニティ or 製作者にお問い合わせください。
https://mmvc.readthedocs.io/ja/latest/index.html

順次更新
## Note
なにか不明点があればお気軽にご連絡ください。
## MMVCコミュニティサーバ(discord)
開発の最新情報や、不明点のお問合せ、MMVCの活用法などMMVCに関するコミュニティサーバです。  
https://mmvc.readthedocs.io/ja/latest/index.html

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

