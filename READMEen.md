# MMVC_Trainer

AI-based Real-Time Voice Changer Model Training Tool

## Description

This repository provides the training environment for models used in the AI-based real-time voice changer, "MMVC" (RealTime-Many to Many Voice Conversion).
By utilizing Google Colaboratory (Google Colab), users can easily execute the machine learning training phase regardless of their local hardware environment.

## Concept

"Simple", "For Everyone", "Any Voice", "In Real-Time"

## Demo

* Currently in production (v1.3.0.0)
* [https://www.nicovideo.jp/watch/sm40386035](https://www.nicovideo.jp/watch/sm40386035) (v1.2.0.0)

---

## Terms of Use & Audio Distribution (As of August 10, 2022)

### MMVC Terms of Use

The terms of use for MMVC (hereafter referred to as "the Software") generally comply with the MIT License.

1. Anyone is free to use the Software, including copying, distributing, modifying, redistributing modified versions, commercial use, and paid sales.
2. If the Software is used on a platform where displaying a license is possible, please include one of the license notations provided below. If displaying a license is difficult (e.g., usage within VRChat), the notation is not required.
3. The developers provide no warranty regarding the Software. Furthermore, the developers assume no responsibility for any issues or damages arising from the use of the Software.
4. When using audio data as training material, you must obtain permission from the copyright holder of the data prior to use. You must also adhere to the terms of service provided by the original distributor of the audio data.

### Official MMVC Audio Data Terms and Downloads

In addition to the Software's terms, using the official audio data below requires agreement to the respective providers' terms of service.
*Note: We have received special permission from the following companies and organizations to modify and redistribute their audio data specifically for this Software.*

**SSS LLC.**

* [Terms of Use] | [Zundamon Audio Data] (Same data bundled with the Software)
* [Terms of Use] | [Kyushu Sora Audio Data]
* [Terms of Use] | [Shikoku Metan Audio Data]
* [Terms of Use] | [Tohoku Kiritan Audio Data]

**Kasukabe Tsumugi Project**

* [Terms of Use] | [Kasukabe Tsumugi Audio Data]

*(Note: Replace bracketed text with original URLs if maintaining hyperlink structures).*

### License Notation

When using the characters Zundamon, Shikoku Metan, Kyushu Sora, Kasukabe Tsumugi, or Tohoku Kiritan, please specify the tool used by adding a format like `MMVC: Zundamon` or `MMVC: Zundamon/Shikoku Metan` alongside the license pattern below. If displaying a license is difficult, notation is not required.

**License Pattern 1**

```text
Copyright (c) 2022 Isle.Tennos
Released under the MIT license
https://opensource.org/licenses/mit-license.php

```

**License Pattern 2 (Recommended)**

```text
MMVCv1.x.x.x (Version Used)
Copyright (c) 2022 Isle.Tennos
Released under the MIT license
https://opensource.org/licenses/mit-license.php
git: https://github.com/isletennos/MMVC_Trainer
community (discord): https://discord.gg/2MGysH3QpD

```

---

## Requirements

* A Google Account

## Installation

Click the button below to install the repository to your Google Drive via Google Colab.

Once completed, open your Google Drive, navigate to `My Drive > MMVC_Trainer > notebook`, and execute the respective notebooks.

---

## Usage

### Tutorial: Becoming Zundamon

This tutorial uses the audio data for Zundamon (SSS LLC.). You must comply with the Zundamon Terms of Use independently of the MMVC terms.

**Phase 1: Recording and Organizing Audio Data**

1. Record your voice audio data. Use `00_Rec_Voice.ipynb` in the notebook directory, or record locally and upload to Google Drive. Read approximately 100 sentences using scripts like the JVS or ITA corpus. The recorded audio must strictly be **24000Hz 16bit 1ch**.
2. Place your audio and text data into `dataset/textful/00_myvoice`. If you used `00_Rec_Voice.ipynb`, this step is handled automatically. Ensure your final directory structure matches the following:

```text
dataset
├── textful
│   ├── 00_myvoice
│   │   ├── text
│   │   │   ├── emoNormal_001.txt
│   │   │   ├── ...
│   │   └── wav
│   │       ├── emoNormal_001.wav
│   │       ├── ...
│   ├── 01_target
│   │   ├── text
│   │   └── wav
│   └── 1205_zundamon
│       ├── text
│       └── wav
└── textless

```

**Phase 2: Training the Model**

1. Ensure pre-trained data is placed. If installed via `00_Clone_Repo.ipynb`, this is already done. Otherwise, download `G_v13_20231020.pth` and `D_v13_20231020.pth` from the Hugging Face repository and place them in the `fine_model` directory.
2. Run `01_Create_Configfile.ipynb` in Google Colab to generate the necessary configuration files.
3. Open the generated `train_config.json` in the `configs` folder. Optimize `eval_interval` (model save frequency) and `batch_size` (based on assigned GPU). If unsure, leave default values.
4. Execute `02_Train_MMVC.ipynb` in Google Colab. The trained models will be generated in the `logs/` directory.

**Phase 3: Model Verification**

1. Execute `03_MMVC_Interface.ipynb` in Google Colab to test the performance of your trained model.

### Becoming a Custom Character

**Phase 1: Organizing Source and Target Data**

1. Prepare your voice data/text and the target voice data/text. Both datasets are highly recommended to be formatted as **24000Hz 16bit 1ch**.
2. Place your voice data in `00_myvoice` and the target character's voice data in `01_target` following the exact directory structure outlined in the tutorial section above.

**Phases 2 & 3: Training and Verification**

1. Follow the exact same procedures outlined in Phase 2 and Phase 3 of the Zundamon tutorial.

---

## MMVC_Client

### Official Client

The client software to operate MMVC locally:
[https://github.com/isletennos/MMVC_Client](https://github.com/isletennos/MMVC_Client)

### Community Client

**Voice Changer Trainer and Player**
A client software built to run MMVC across various environments.
[https://github.com/w-okada/voice-changer](https://github.com/w-okada/voice-changer)

**Operating Status**

| # | OS | Middleware | Training App | Voice Changer |
| --- | --- | --- | --- | --- |
| 1 | Windows | Anaconda | Not Tested | Not Tested |
| 2 | Windows (WSL2) | Docker | Verified on WSL2+Ubuntu | Verified on WSL2+Ubuntu |
| 3 | Windows (WSL2) | Anaconda | Not Tested | Verified on Ubuntu |
| 4 | Mac (Intel) | Anaconda | Not Tested | Works but very slow (2019, i5) |
| 5 | Mac (M1) | Anaconda | Not Tested | Verified on M1 MBA, M1 MBP |
| 6 | Linux | Docker | Verified on Debian | Verified on Debian |
| 7 | Linux | Anaconda | Not Tested | Not Tested |
| 8 | Colab | Notebook | Verified | Verified |

*Note: CPU operation is possible on relatively modern hardware (proven on i7-9700K).*

---

## Community Tutorial Videos (v1.2.1.x)

* **Preparation Part 1:** [NicoNico] | [YouTube]
* **Audio Correction:** [NicoNico] | [YouTube]
* **Preparation Part 2:** [NicoNico] | [YouTube]
* **Training Part 1:** [NicoNico] | [YouTube]
* **Training Part 2:** [NicoNico] | [YouTube]
* **Training (Post):** [NicoNico] | [YouTube]
* **Real-Time Usage:** [NicoNico] | [YouTube]
* **Q&A Section:** [NicoNico] | [YouTube]
* **Advanced (Kyushu Sora):** [NicoNico] | [YouTube]
* **Advanced (Otomachi Una):** [NicoNico] | [YouTube]

*(Please refer to the original document for exact video URLs).*

---

## Support & Community

**Q&A / FAQ**
Please refer to the official FAQ wiki:
[https://github.com/isletennos/MMVC_Trainer/wiki/FAQ](https://github.com/isletennos/MMVC_Trainer/wiki/FAQ)

**Discord Community Server**
Join the community server for the latest development news, technical support, and usage tips:
[https://discord.gg/2MGysH3QpD](https://discord.gg/2MGysH3QpD)

**Developer Contact (PIXIV FANBOX)**
For direct inquiries or questions for the developer:
[https://mmvc.fanbox.cc/posts/6858033](https://mmvc.fanbox.cc/posts/6858033)

---

## Special Thanks

* **JVS (Japanese versatile speech) corpus**
Contributors: Shinnosuke Takamichi, Kentaro Mitsui, Yuki Saito, Tomoki Koriyama, Naoko Tanji, Hiroshi Saruwatari
* **ITA Corpus Multimodal Database**
Contributors: Fumiya Kanai, Ryuichi Chiba, Takeshi Saito, Masanori Morise, Junya Oguchi, Takashi Nose, Maiko Ono, Yasuo Oda
* **Tsukuyomi-chan Corpus**
Contributor: Rei Yumesaki

## Reference

* [https://arxiv.org/abs/2106.06103](https://arxiv.org/abs/2106.06103)
* [https://github.com/jaywalnut310/vits](https://github.com/jaywalnut310/vits)

## Author

**Isle Tennos**
Twitter: [https://twitter.com/IsleTennos](https://twitter.com/IsleTennos)
