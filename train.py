import argparse
import lightning.pytorch as pl
import logging
import torch
import os
import pathlib

from typing import List, NoReturn
from torch.utils.tensorboard import SummaryWriter
from models.resunet import ResUNet30
from models.lassnet import UNetRes_FiLM
from models.metrics import get_loss_function
from models.sepcommander import AudioSep
from callbacks.base import CheckpointEveryNSteps
from models.lr_schedulers import get_lr_lambda

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

    # Configuration of the separation model
    backbone = configs['model']['backbone']
    input_channels = configs['model']['input_channels']
    output_channels = configs['model']['output_channels']
    condition_size = configs['model']['condition_size']
    only_train_film = configs["model"]["only_train_film"]
    
    # Configuration of the trainer
    num_nodes = configs['train']['num_nodes']
    sync_batchnorm = configs['train']['sync_batchnorm'] 
    optimizer_type = configs["train"]["optimizer"]["optimizer_type"]
    learning_rate = float(configs['train']["optimizer"]['learning_rate'])
    lr_lambda_type = configs['train']["optimizer"]['lr_lambda_type']
    warm_up_steps = configs['train']["optimizer"]['warm_up_steps']
    reduce_lr_steps = configs['train']["optimizer"]['reduce_lr_steps']
    save_step_frequency = configs['train']['save_step_frequency']
    resume_checkpoint_path = configs["train"]["checkpoint_path"]
    n_epochs = configs["train"]["n_epochs"]

    # Get directories and paths
    checkpoints_dir, logs_dir, tf_logs_dir, statistics_path = get_dirs(
        workspace, filename, config_yaml, devices_num,
    )

    logging.info(configs)

    # data module
    data_module = get_data_module(config_yaml)
    
    # model
    if backbone == 'resunet30':
        ss_model = ResUNet30(
            input_channels=input_channels,
            output_channels=output_channels,
            condition_size=condition_size,
            only_train_film=only_train_film
        )
    elif backbone == 'lassnet':
        ss_model = UNetRes_FiLM(
            channels=input_channels,
            cond_embedding_dim=condition_size,
            nsrc=output_channels,
            only_train_film=only_train_film)
    else:
        raise ValueError(f"Unknown backbone [{backbone}]")

    lr_lambda_func = get_lr_lambda(
        lr_lambda_type=lr_lambda_type,
        warm_up_steps=warm_up_steps,
        reduce_lr_steps=reduce_lr_steps,
    )

    # pytorch-lightning model
    pl_model = AudioSep(
        ss_model=ss_model,
        loss_function=get_loss_function(
            configs['train']['loss_type']),
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        lr_lambda_func=lr_lambda_func,
    )

    checkpoint_every_n_steps = CheckpointEveryNSteps(
        checkpoints_dir=checkpoints_dir,
        save_step_frequency=save_step_frequency,
    )

    summary_writer = SummaryWriter(log_dir=tf_logs_dir)

    callbacks = [checkpoint_every_n_steps]

    accelerator = configs['accelerator'] if configs['accelerator'] else 'auto'
    device = configs['device'] if configs['device'] else 'auto'
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=device,
        #strategy='ddp_find_unused_parameters_true',
        num_nodes=num_nodes,
        precision="32-true",
        logger=None,
        callbacks=callbacks,
        fast_dev_run=False,
        max_epochs=n_epochs,
        log_every_n_steps=50,
        use_distributed_sampler=True,
        sync_batchnorm=sync_batchnorm,
        num_sanity_val_steps=2,
        enable_checkpointing=False,
        enable_model_summary=True,
    )

    # Fit, evaluate, and save checkpoints.

    # Load checkpoint resume_checkpoint_path
    if resume_checkpoint_path:
        logging.info(f'Finetuning with checkpoint [{resume_checkpoint_path}]')
        weights = torch.load(
            resume_checkpoint_path, map_location=torch.device('cpu'))
        new_weights = {}
        
        if 'model' in weights.keys():
            weights = weights['model']
        elif 'state_dict' in weights.keys():
            weights = weights['state_dict']

        for k, v in weights.items():
            if 'text_embedder.' in k:
                continue
            new_weights[k.replace('module.', 'ss_model.').replace('.UNet', '')] = v
        weights = new_weights
        pl_model.load_state_dict(weights, strict=True)
        logging.info(f'Loaded checkpoint from [{resume_checkpoint_path}]')

    trainer.fit(
        model=pl_model,
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

    args = parser.parse_args()
    args.filename = pathlib.Path(__file__).stem

    train(args)
