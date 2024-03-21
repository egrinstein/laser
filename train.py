import argparse
import logging
import os
import pathlib
import lightning.pytorch as pl

from typing import List, NoReturn
from torch.utils.tensorboard import SummaryWriter
from models.resunet import *
from losses import get_loss_function
from models.audiosep import AudioSep, get_model_class
from data.waveform_mixer import WaveformMixer
from models.clap_encoder import CLAP_Encoder
from models.tinyclip_encoder import TinyCLIP_Encoder
from callbacks.base import CheckpointEveryNSteps
from optimizers.lr_schedulers import get_lr_lambda

from utils import create_logging, parse_yaml, get_data_module


def get_dirs(
    workspace: str, 
    filename: str, 
    config_yaml: str, 
    devices_num: int
) -> List[str]:
    r"""Get directories and paths.

    Args:
        workspace (str): directory of workspace
        filename (str): filename of current .py file.
        config_yaml (str): config yaml path
        devices_num (int): 0 for cpu and 8 for training with 8 GPUs

    Returns:
        checkpoints_dir (str): directory to save checkpoints
        logs_dir (str), directory to save logs
        tf_logs_dir (str), directory to save TensorBoard logs
        statistics_path (str), directory to save statistics
    """
    
    os.makedirs(workspace, exist_ok=True)

    yaml_name = pathlib.Path(config_yaml).stem

    # Directory to save checkpoints
    checkpoints_dir = os.path.join(
        workspace,
        "checkpoints",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
    )
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Directory to save logs
    logs_dir = os.path.join(
        workspace,
        "logs",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
    )
    os.makedirs(logs_dir, exist_ok=True)

    # Directory to save TensorBoard logs
    create_logging(logs_dir, filemode="w")
    logging.info(args)

    tf_logs_dir = os.path.join(
        workspace,
        "tf_logs",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
    )

    # Directory to save statistics
    statistics_path = os.path.join(
        workspace,
        "statistics",
        filename,
        "{},devices={}".format(yaml_name, devices_num),
        "statistics.pkl",
    )
    os.makedirs(os.path.dirname(statistics_path), exist_ok=True)

    return checkpoints_dir, logs_dir, tf_logs_dir, statistics_path


def train(args) -> NoReturn:
    r"""Train, evaluate, and save checkpoints.

    Args:
        workspace: str, directory of workspace
        gpus: int, number of GPUs to train
        config_yaml: str
    """

    # arguments & parameters
    workspace = args.workspace
    config_yaml = args.config_yaml
    filename = args.filename

    devices_num = torch.cuda.device_count()
    # Read config file.
    configs = parse_yaml(config_yaml)

    # Configuration of data
    max_mix_num = configs['data']['max_mix_num']
    lower_db = configs['data']['loudness_norm']['lower_db']
    higher_db = configs['data']['loudness_norm']['higher_db']

    # Configuration of the separation model
    query_net = configs['model']['query_net']
    model_type = configs['model']['model_type']
    input_channels = configs['model']['input_channels']
    output_channels = configs['model']['output_channels']
    condition_size = configs['model']['condition_size']
    use_text_ratio = configs['model']['use_text_ratio']
    
    # Configuration of the trainer
    num_nodes = configs['train']['num_nodes']
    sync_batchnorm = configs['train']['sync_batchnorm'] 
    optimizer_type = configs["train"]["optimizer"]["optimizer_type"]
    learning_rate = float(configs['train']["optimizer"]['learning_rate'])
    lr_lambda_type = configs['train']["optimizer"]['lr_lambda_type']
    warm_up_steps = configs['train']["optimizer"]['warm_up_steps']
    reduce_lr_steps = configs['train']["optimizer"]['reduce_lr_steps']
    save_step_frequency = configs['train']['save_step_frequency']
    resume_checkpoint_path = args.resume_checkpoint_path
    if resume_checkpoint_path == "":
        resume_checkpoint_path = None
    else:
        logging.info(f'Finetuning AudioSep with checkpoint [{resume_checkpoint_path}]')

    # Get directories and paths
    checkpoints_dir, logs_dir, tf_logs_dir, statistics_path = get_dirs(
        workspace, filename, config_yaml, devices_num,
    )

    logging.info(configs)

    # data module
    data_module = get_data_module(config_yaml)
    
    # model
    Model = get_model_class(model_type=model_type)

    ss_model = Model(
        input_channels=input_channels,
        output_channels=output_channels,
        condition_size=condition_size,
    )

    waveform_mixer = WaveformMixer(
        max_mix_num=max_mix_num,
        lower_db=lower_db, 
        higher_db=higher_db
    )

    if query_net == 'CLAP':
        query_encoder = CLAP_Encoder()
    elif query_net == 'TinyCLIP':
        query_encoder = TinyCLIP_Encoder()
    else:
        raise NotImplementedError

    lr_lambda_func = get_lr_lambda(
        lr_lambda_type=lr_lambda_type,
        warm_up_steps=warm_up_steps,
        reduce_lr_steps=reduce_lr_steps,
    )

    # pytorch-lightning model
    pl_model = AudioSep(
        query_encoder=query_encoder,
        waveform_mixer=waveform_mixer,
        ss_model=ss_model,
        loss_function=get_loss_function(
            configs['train']['loss_type']),
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        lr_lambda_func=lr_lambda_func,
        use_text_ratio=use_text_ratio
    )

    checkpoint_every_n_steps = CheckpointEveryNSteps(
        checkpoints_dir=checkpoints_dir,
        save_step_frequency=save_step_frequency,
    )

    summary_writer = SummaryWriter(log_dir=tf_logs_dir)

    callbacks = [checkpoint_every_n_steps]

    device = configs['device'] if configs['device'] else 'auto'
    trainer = pl.Trainer(
        accelerator=device,
        devices='auto',
        #strategy='ddp_find_unused_parameters_true',
        num_nodes=num_nodes,
        precision="32-true",
        logger=None,
        callbacks=callbacks,
        fast_dev_run=False,
        max_epochs=-1,
        log_every_n_steps=50,
        use_distributed_sampler=True,
        sync_batchnorm=sync_batchnorm,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
        enable_model_summary=True,
    )

    # Fit, evaluate, and save checkpoints.

    # Load checkpoint resume_checkpoint_path
    if resume_checkpoint_path is not None:
        weights = torch.load(resume_checkpoint_path, map_location=torch.device('cpu'))
        pl_model.load_state_dict(weights['state_dict'], strict=False)
        logging.info(f'Loaded checkpoint from [{resume_checkpoint_path}]')

    trainer.fit(
        model=pl_model, 
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=data_module,
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workspace", type=str, required=True, help="Directory of workspace."
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path of config file for training.",
    )

    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        required=True,
        default='',
        help="Path of pretrained checkpoint for finetuning.",
    )

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem

    train(args)
