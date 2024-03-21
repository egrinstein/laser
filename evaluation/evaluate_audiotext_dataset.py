import os
import sys
import re

import numpy as np
import torch
import lightning.pytorch as pl

from typing import Dict, List
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../AudioSep/')
from models.metrics import (
    calculate_sdr,
    calculate_sisdr,
)
from data.waveform_mixer import WaveformMixer

from utils import get_mean_from_dict_values, get_data_module, load_ss_model


class AudioTextDatasetEvaluator:
    def __init__(
        self,
        config_yaml: str = 'config/audiosep_base.yaml',
        checkpoint_path: str = 'checkpoint/audiosep_base_4M_steps.ckpt',
    ) -> None:
        r"""AudioSet evaluator.

        Args:
            audios_dir (str): directory of evaluation segments
            classes_num (int): the number of sound classes
            number_per_class (int), the number of samples to evaluate for each sound class

        Returns:
            None
        """

        self.datamodule = get_data_module(config_yaml)
        self.waveform_mixer = WaveformMixer()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model
        self.pl_model = load_ss_model(
            config_yaml,
            checkpoint_path=checkpoint_path,
        ).to(device)

    @torch.no_grad()
    def __call__(
        self,
        average: bool = False,
    ) -> Dict:
        r"""Evalute."""

        pl_model = self.pl_model
        pl_model.eval()

        sisdrs_dict = {}
        sdris_dict = {}
        
        print(f'Evaluation on {self.dataset.datafiles} with [text label] queries.')
        device = pl_model.device
        
        n_samples = len(self.dataset)
        pbar = tqdm(enumerate(self.dataset), total=n_samples)
        for idx, data in pbar:
            source = data['waveform']
            text = data['text']

            noise_track_idxs = self.waveform_mixer.get_noise_track_idxs(self.dataset, idx)
            mixture_texts = [self.dataset[i]['text'] for i in noise_track_idxs]
            mixture = self.waveform_mixer.mix_waveforms(source, [
                self.dataset[i]['waveform'] for i in noise_track_idxs]
            )
            sdr_no_sep = calculate_sdr(ref=source, est=mixture)

            conditions = pl_model.query_encoder(
                modality='text',
                text=text,
            )

            input_dict = {
                "mixture": torch.Tensor(mixture)[None, None, :].to(device),
                "condition": conditions,
            }

            sep_segment = pl_model.ss_model(input_dict)["waveform"]
            # sep_segment: (batch_size=1, channels_num=1, segment_samples)

            sep_segment = sep_segment.squeeze(0).squeeze(0).data.cpu().numpy()
            # sep_segment: (segment_samples,)

            sdr = calculate_sdr(ref=source, est=sep_segment)
            sdri = sdr - sdr_no_sep
            sisdr = calculate_sisdr(ref=source, est=sep_segment)

            sisdrs_dict[class_id].append(sisdr)
            sdris_dict[class_id].append(sdri)

            # Update tqdm progress bar
            pbar.set_postfix(
                {
                    "class_id": class_id,
                    "SI-SDR": np.nanmedian(sisdrs_dict[class_id]),
                    "SDR": np.nanmedian(sdris_dict[class_id]),
                }
            )

        stats_dict = {
            "sisdrs_dict": sisdrs_dict,
            "sdris_dict": sdris_dict,
        }

        if average:
            median_sdris = {}
            median_sisdrs = {}

            for class_id in range(527):
                median_sdris[class_id] = np.nanmedian(stats_dict["sdris_dict"][class_id])
                median_sisdrs[class_id] = np.nanmedian(stats_dict["sisdrs_dict"][class_id])
            stats_dict["sdris_dict"] = get_mean_from_dict_values(median_sdris)
            stats_dict["sisdrs_dict"] = get_mean_from_dict_values(median_sisdrs)

        return stats_dict

    def _get_audio_names(self, audios_dir: str) -> List[str]:
        r"""Get evaluation audio names."""
        audio_names = sorted(os.listdir(audios_dir))

        audio_names = [audio_name for audio_name in audio_names if '.wav' in audio_name]
        
        audio_names = [
            re.search(
                "(.*),(mixture|source).wav",
                audio_name).group(1) for audio_name in audio_names]

        audio_names = sorted(list(set(audio_names)))
        return audio_names

    @staticmethod
    def get_median_metrics(stats_dict, metric_type):
        class_ids = stats_dict[metric_type].keys()
        median_stats_dict = {
            class_id: np.nanmedian(
                stats_dict[metric_type][class_id]) for class_id in class_ids}
        return median_stats_dict


if __name__ == "__main__":
    
    config_yaml = 'config/audiosep_base.yaml'
    checkpoint_path = 'checkpoint/audiosep_base_4M_steps.ckpt'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load PytorchLightning model
    pl_model = load_ss_model(
        config_yaml,
        checkpoint_path=checkpoint_path,
    ).to(device)

    # Load data module
    data_module = get_data_module(config_yaml)

    # Directory to save logs
    logs_dir = os.path.join(
        "workspace/AudioSep",
        "logs",
        "evaluate",
    )
    os.makedirs(logs_dir, exist_ok=True)
    callbacks = [SummaryWriter(log_dir=logs_dir)]

    # Test evaluation
    trainer = pl.Trainer(
        accelerator=device,
        devices='auto',
        #strategy='ddp_find_unused_parameters_true',
        # num_nodes=num_nodes,
        precision="32-true",
        logger=None,
        # callbacks=callbacks,
        fast_dev_run=False,
        max_epochs=-1,
        log_every_n_steps=50,
        use_distributed_sampler=True,
        # sync_batchnorm=sync_batchnorm,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
        enable_model_summary=True,
    )
    trainer.test(model=pl_model, datamodule=data_module)