import numpy as np
import os
import torch

import argparse

# from evaluation.evaluate_audioset import AudioSetEvaluator
from evaluation.evaluate_audiocaps import AudioCapsEvaluator
# from evaluation.evaluate_vggsound import VGGSoundEvaluator
# from evaluation.evaluate_music import MUSICEvaluator
# from evaluation.evaluate_esc50 import ESC50Evaluator
# from evaluation.evaluate_clotho import ClothoEvaluator
from models.clap_encoder import ClapEncoder
from models.resunet import ResUNet30

from utils import (
    load_ss_model,
    parse_yaml,
)


def eval(checkpoint_path, log_dir='eval_logs', config_yaml='config/audiosep_base.yaml'):

    os.makedirs(log_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # # Configuration of the separation model
    # configs = parse_yaml(config_yaml)
    # backbone = configs['model']['backbone']
    # input_channels = configs['model']['input_channels']
    # output_channels = configs['model']['output_channels']
    # condition_size = configs['model']['condition_size']
    # only_train_film = configs["model"]["only_train_film"]

    # if backbone == 'resunet30':
    #     pl_model = ResUNet30(
    #         input_channels=input_channels,
    #         output_channels=output_channels,
    #         condition_size=condition_size,
    #         only_train_film=only_train_film
    #     ).to(device)
    # else:
    #     raise ValueError(f"Unknown backbone [{backbone}]")
    
    # Load model
    query_encoder = ClapEncoder().eval()

    pl_model = load_ss_model(
        config_yaml,
        checkpoint_path=checkpoint_path,
        #query_encoder=query_encoder
    )#.to(device)
    pl_model.query_encoder = query_encoder
    pl_model = pl_model.to(device)

    print(f'-------  Start Evaluation  -------')

    msgs = []

    # evaluation on AudioCaps
    if os.path.exists('evaluation/data/audiocaps'):
        audiocaps_evaluator = AudioCapsEvaluator()
        SISDR, SDRi = audiocaps_evaluator(pl_model)
        msg_audiocaps = "AudioCaps Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
        print(msg_audiocaps)
        msgs.append(msg_audiocaps)

    # # evaluation on Clotho
    # if os.path.exists('evaluation/data/clotho'):
    #     clotho_evaluator = ClothoEvaluator()
    #     SISDR, SDRi = clotho_evaluator(pl_model)
    #     msg_clotho = "Clotho Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    #     print(msg_clotho)
    #     msgs.append(msg_clotho)
    
    # # evaluation on VGGSound+ (YAN)
    # if os.path.exists('evaluation/data/vggsound'):
    #     vggsound_evaluator = VGGSoundEvaluator()
    #     SISDR, SDRi = vggsound_evaluator(pl_model)
    #     msg_vgg = "VGGSound Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    #     print(msg_vgg)
    #     msgs.append(msg_vgg)

    # # evaluation on MUSIC
    # if os.path.exists('evaluation/data/music'):
    #     music_evaluator = MUSICEvaluator()
    #     SISDR, SDRi = music_evaluator(pl_model)
    #     msg_music = "MUSIC Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    #     print(msg_music)
    #     msgs.append(msg_music)

    # # evaluation on ESC-50
    # if os.path.exists('evaluation/data/esc50'):
    #     esc50_evaluator = ESC50Evaluator()
    #     SISDR, SDRi = esc50_evaluator(pl_model)
    #     msg_esc50 = "ESC-50 Avg SDRi: {:.3f}, SISDR: {:.3f}".format(SDRi, SISDR)
    #     print(msg_esc50)
    #     msgs.append(msg_esc50)

    # # evaluation on AudioSet
    # if os.path.exists('evaluation/data/audioset'):
    #     audioset_evaluator = AudioSetEvaluator()
    #     stats_dict = audioset_evaluator(pl_model=pl_model, average=True)
        
    #     SDRi, SISDR = stats_dict["sdris_dict"], stats_dict["sisdrs_dict"]
    #     msg_audioset = "AudioSet Avg SDRi: {:.3f}, SISDR: {:.3f}".format(
    #         SDRi, SISDR)
    #     print(msg_audioset)
    #     msgs.append(msg_audioset)

    # open file in write mode
    log_path = os.path.join(log_dir, 'eval_results.txt')
    with open(log_path, 'w') as fp:
        for msg in msgs:
            fp.write(msg + '\n')
    print(f'Eval log is written to {log_path} ...')
    print('-------------------------  Done  ---------------------------')


if __name__ == '__main__':
    # argparse parameter for checkpoint path:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--config_yaml', type=str, default='config/audiosep_base.yaml')
    parser.add_argument('--log_dir', type=str, default='eval_logs')
    args = parser.parse_args()


    eval(checkpoint_path=args.checkpoint_path, log_dir=args.log_dir, config_yaml=args.config_yaml)

