# SepCommander: Sound Source Separation using text commands

## Introduction
This repository contains our IWAENC2024 submission

## Installation
Create a Conda virtual environment by running:
```bash
conda env create -f environment.yml
```

Alternatively, you can look at the `environment.yml` file and install the dependencies manually.

### Dataset creation

1. Download AudioCaps dataset running the command:
```bash
python -m data.downloaders.audiocaps --out_wav_dir ~/datasets/audiocaps --out_csv_dir ~/config/datafiles/csvs
```

The dataset has around 80GB of audio files

2. Create the mixture files

The mixture files are created by mixing the audio files from the AudioCaps dataset. Each sample on AudioCaps is mixed with one other sample from the same dataset. This is done in 4 steps:

2.1 Create the mixture .csv files
```bash
python -m data.mixing.audiocaps_csv_mixer --in_csv_dir config/datafiles/csvs --out_csv_dir config/datafiles/csvs
```

2.2 Create the mixture .wav files
```bash
python -m data.mixing.mix_audiocaps --in_csv_dir config/datafiles/csvs/ --out_json_dir config/datafiles/ --in_wav_dir ~/datasets/audiocaps/ --out_wav_dir ~/datasets/audiocaps/mix
```

2.3 Create the command embeddings
```bash
@python -m data.mixing.create_audiocaps_commands --in_csv_dir config/datafiles/csvs --out_dir ~/datasets/audiocaps/embeddings
```

2.4 Create the .json file which is loaded by the dataloader
```bash
@python data/json/create_template_json_audiocaps.py --wav_dir ~/datasets/audiocaps/ --csv_dir config/datafiles/csvs/ --output_json config/datafiles/
```


### Training
python train.py --workspace workspace/ --config_yaml config/audiosep_base.yaml --resume_checkpoint_path checkpoint/audiosep_base_4M_steps.ckpt
