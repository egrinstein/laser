import argparse
import os
import wget

from audiocaps_download import Downloader


def main(out_csv_dir: str, out_wav_dir: str, n_jobs: int = 5):
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_wav_dir, exist_ok=True)

    print("Downloading the AudioCaps dataset")
    print("1. Downloading the captions")
    caption_links = [
        'https://github.com/cdjkim/audiocaps/raw/master/dataset/train.csv',
        'https://github.com/cdjkim/audiocaps/raw/master/dataset/val.csv',
        'https://github.com/cdjkim/audiocaps/raw/master/dataset/test.csv',
    ]

    for link in caption_links:
        wget.download(link, out=out_csv_dir)

    print("2. Downloading the audio files")

    d = Downloader(root_path=out_wav_dir, n_jobs=n_jobs)
    d.download(format = 'wav') # it will cross-check the files with the csv files in the original repository


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download the AudioCaps dataset')
    parser.add_argument('--out_csv_dir', type=str, help='Path to the root directory where the datasets will be saved')
    parser.add_argument('--out_wav_dir', type=str, help='Path to the root directory where the datasets will be saved')
    parser.add_argument('--n_jobs', type=int, default=5, help='Number of parallel jobs to download the dataset')

    args = parser.parse_args()

    main(args.out_csv_dir, args.out_wav_dir, args.n_jobs)
