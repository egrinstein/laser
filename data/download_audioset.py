from audioset_download import Downloader

d = Downloader(root_path='~/datasets/audioset_balanced/', labels=None, n_jobs=1, download_type='balanced_train', copy_and_replicate=False)
d.download(format = 'wav')
