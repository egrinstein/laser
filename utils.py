import os
import json
import logging
from typing import Dict
import numpy as np
from data.mixing.waveform_mixer import WaveformMixer
import torch
import torch.nn as nn
import yaml

from losses import get_loss_function
from models.clap_encoder import CLAP_Encoder
from models.audiosep import AudioSep, get_model_class
from data.audiotext_dataset import AudioTextDataLoader
from data.datamodules import DataModule


def ignore_warnings():
    import warnings
    # Ignore UserWarning from torch.meshgrid
    warnings.filterwarnings('ignore', category=UserWarning, module='torch.functional')

    # Refined regex pattern to capture variations in the warning message
    pattern = r"Some weights of the model checkpoint at roberta-base were not used when initializing RobertaModel: \['lm_head\..*'\].*"
    warnings.filterwarnings('ignore', message=pattern)



def create_logging(log_dir, filemode):
    os.makedirs(log_dir, exist_ok=True)
    i1 = 0

    while os.path.isfile(os.path.join(log_dir, "{:04d}.log".format(i1))):
        i1 += 1

    log_path = os.path.join(log_dir, "{:04d}.log".format(i1))
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
        datefmt="%a, %d %b %Y %H:%M:%S",
        filename=log_path,
        filemode=filemode,
    )

    # Print to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(name)-12s: %(levelname)-8s %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)

    return logging


def float32_to_int16(x: float) -> int:
    x = np.clip(x, a_min=-1, a_max=1)
    return (x * 32767.0).astype(np.int16)


def int16_to_float32(x: int) -> float:
    return (x / 32767.0).astype(np.float32)


def parse_yaml(config_yaml: str) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


def ids_to_hots(ids, classes_num, device):
    hots = torch.zeros(classes_num).to(device)
    for id in ids:
        hots[id] = 1
    return hots


def get_audioset632_id_to_lb(ontology_path: str) -> Dict:
    r"""Get AudioSet 632 classes ID to label mapping."""
    
    audioset632_id_to_lb = {}

    with open(ontology_path) as f:
        data_list = json.load(f)

    for e in data_list:
        audioset632_id_to_lb[e["id"]] = e["name"]

    return audioset632_id_to_lb


def get_mean_from_dict_values(d):
    mean = np.nanmean(list(d.values()))
    return mean


def repeat_to_length(audio: np.ndarray, segment_samples: int) -> np.ndarray:
    r"""Repeat audio to length."""
    
    repeats_num = (segment_samples // audio.shape[-1]) + 1
    audio = np.tile(audio, repeats_num)[0 : segment_samples]

    return audio


def get_ss_model(config_yaml) -> nn.Module:
    r"""Load trained universal source separation model.

    Args:
        configs (Dict)
        checkpoint_path (str): path of the checkpoint to load
        device (str): e.g., "cpu" | "cuda"

    Returns:
        pl_model: pl.LightningModule
    """
    configs = parse_yaml(config_yaml)

    ss_model_type = configs["model"]["model_type"]
    input_channels = configs["model"]["input_channels"]
    output_channels = configs["model"]["output_channels"]
    condition_size = configs["model"]["condition_size"]
    
    # Initialize separation model
    SsModel = get_model_class(model_type=ss_model_type)

    ss_model = SsModel(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    return ss_model


def load_ss_model(
    config_yaml: str,
    checkpoint_path: str,
) -> nn.Module:
    r"""Load trained universal source separation model.

    Args:
        config_yaml: str
        checkpoint_path (str): path of the checkpoint to load
        device (str): e.g., "cpu" | "cuda"

    Returns:
        pl_model: pl.LightningModule
    """

    configs = parse_yaml(config_yaml)

    ss_model_type = configs["model"]["model_type"]
    input_channels = configs["model"]["input_channels"]
    output_channels = configs["model"]["output_channels"]
    condition_size = configs["model"]["condition_size"]
    
    # Initialize separation model
    SsModel = get_model_class(model_type=ss_model_type)

    ss_model = SsModel(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    query_encoder = CLAP_Encoder().eval()

    waveform_mixer = WaveformMixer(
        max_mix_num = configs['data']['max_mix_num'],
        lower_db = configs['data']['loudness_norm']['lower_db'],
        higher_db = configs['data']['loudness_norm']['higher_db']
    )

    # Load PyTorch Lightning model
    pl_model = AudioSep.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        strict=False,
        ss_model=ss_model,
        waveform_mixer=waveform_mixer,
        query_encoder=query_encoder,
        loss_function=get_loss_function(configs['train']['loss_type']),
        optimizer_type=None,
        learning_rate=None,
        lr_lambda_func=None,
        map_location=torch.device('cpu'),
    )

    return pl_model


def parse_yaml(config_yaml: str) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)
    

def get_data_module(
    config_yaml: str,
) -> DataModule:
    r"""Create data_module. Mini-batch data can be obtained by:

    code-block:: python

        data_module.setup()

        for batch_data_dict in data_module.train_dataloader():
            print(batch_data_dict.keys())
            break

    Args:
        workspace: str
        config_yaml: str
        num_workers: int, e.g., 0 for non-parallel and 8 for using cpu cores
            for preparing data in parallel
        distributed: bool

    Returns:
        data_module: DataModule
    """

    # read configurations
    configs = parse_yaml(config_yaml)
    sampling_rate = configs['data']['sampling_rate']
    segment_seconds = configs['data']['segment_seconds']
    batch_size = configs['train']['batch_size_per_device']
    num_workers = configs['train']['num_workers']
    
    # audio-text datasets
    datafiles = configs['data']['train_datafiles']
    
    # dataset
    dataloader = AudioTextDataLoader(
        datafiles=datafiles, 
        sampling_rate=sampling_rate, 
        max_clip_len=segment_seconds,
        batch_size=batch_size,
    )

    test_dataloader = None
    if 'test_datafiles' in configs['data']:
        test_datafiles = configs['data']['test_datafiles']
        if test_datafiles:
            test_dataloader = AudioTextDataLoader(
                datafiles=test_datafiles, 
                sampling_rate=sampling_rate, 
                max_clip_len=segment_seconds,
                batch_size=batch_size,
                shuffle=False,
            )
    
    # data module
    data_module = DataModule(
        train_dataloader=dataloader,
        num_workers=num_workers,
        batch_size=batch_size,
        test_dataloader=test_dataloader,
    )

    return data_module