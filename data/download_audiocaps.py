import wget

from audiocaps_download import Downloader

def main():

    print("Downloading the AudioCaps dataset")
    print("1. Downloading the captions")
    caption_links = [
        'https://github.com/cdjkim/audiocaps/raw/master/dataset/train.csv',
        'https://github.com/cdjkim/audiocaps/raw/master/dataset/val.csv',
        'https://github.com/cdjkim/audiocaps/raw/master/dataset/test.csv',
    ]

    audioset_links = [
        'http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/class_labels_indices.csv'
    ]

    for link in caption_links:
        wget.download(link, out='config/datafiles/')

    print("2. Downloading the audio files")

    d = Downloader(root_path='/Users/ezajlerg/datasets/audiocaps/', n_jobs=1)
    d.download(format = 'wav') # it will cross-check the files with the csv files in the original repository


if __name__ == '__main__':
    main()