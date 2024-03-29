from typing import Dict, List, Optional, NoReturn
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataloader: object,
        batch_size: int,
        num_workers: int,
        test_dataloader: object = None,
    ):
        r"""Data module. To get one batch of data:

        code-block:: python

            data_module.setup()

            for batch_data_dict in data_module.train_dataloader():
                print(batch_data_dict.keys())
                break

        Args:
            train_sampler: Sampler object
            train_dataloader: torch.utils.data.DataLoader object
            num_workers: int
            distributed: bool
        """
        super().__init__()
        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader

        self.num_workers = num_workers
        self.batch_size = batch_size


    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        r"""called on every device."""

        # make assignments here (val/train/test split)
        # called on every process in DDP

        # SegmentSampler is used for selecting segments for training.
        # On multiple devices, each SegmentSampler samples a part of mini-batch
        # data.
        # self.train_dataloader = self._train_dataloader
        # self.test_dataloader = self._test_dataloader


    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader."""
        
        return self._train_dataloader

    def val_dataloader(self):
        # val_split = Dataset(...)
        # return DataLoader(val_split)
        pass

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get test loader."""
        
        return self._test_dataloader

    def teardown(self):
        # clean up after fit or test
        # called on every process in DDP
        pass
