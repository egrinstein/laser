from audioset_download import Downloader

d = Downloader(root_path='~/datasets/audioset/', labels=None, n_jobs=1, download_type='unbalanced_train', copy_and_replicate=False)
d.download(format = 'wav')
