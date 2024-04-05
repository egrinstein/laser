import random
import lightning.pytorch as pl
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional.audio import signal_distortion_ratio, scale_invariant_signal_distortion_ratio

from huggingface_hub import PyTorchModelHubMixin
from torch.optim.lr_scheduler import LambdaLR

# from commander import random_template_command


class AudioSep(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        ss_model: nn.Module = None,
        loss_function = None,
        optimizer_type: str = None,
        learning_rate: float = None,
        lr_lambda_func = None,
        use_text_ratio: float = 1.0,
    ):
        r"""Pytorch Lightning wrapper of PyTorch model, including forward,
        optimization of model, etc.

        Args:
            ss_model: nn.Module
            anchor_segment_detector: nn.Module
            loss_function: function or object
            learning_rate: float
            lr_lambda: function
        """

        super().__init__()
        self.ss_model = ss_model
        self.use_text_ratio = use_text_ratio
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func

        # Average will be computed using exponential moving average
        self.avg_sisdr = 0
        self.avg_qsdr = 0
        self.avg_loss = 0
        self.avg_smoothing = 0.01

        for name, param in self.ss_model.named_parameters():
            if param.requires_grad == False:
                print(name, np.prod(list(param.shape)))

    def forward(self, x):
        pass

    def _step(self, batch_audio_text_dict, batch_idx, compute_sdr=True, sdr_freq=1, prefix='train'):
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_audio_text_dict: e.g. 
            {
                    'input': {
                        'mixture': torch.Tensor,
                        'condition': torch.Tensor
                    }
                    'target': {
                        'interferers': torch.Tensor,
                        'segment': torch.Tensor
                    }
            }
            
        Returns:
            loss: float, loss function of this mini-batch
        """
        # Fix random seeds across devices
        random.seed(batch_idx)

        input_dict = batch_audio_text_dict['input']
        target_dict = batch_audio_text_dict['target']

        if prefix == 'train':
            self.ss_model.train()
        model_output = self.ss_model(input_dict)['waveform']
        model_output = model_output.squeeze(1)
        # (batch_size, 1, segment_samples)

        output_dict = {'segment': model_output}

        # Calculate loss
        loss = self.loss_function(output_dict, target_dict)
        self.avg_loss = (1 - self.avg_smoothing) * self.avg_loss + \
                self.avg_smoothing * loss.item()
        
        log_dict = {f"{prefix}_loss": self.avg_loss}
        
        if compute_sdr and batch_idx % sdr_freq == 0: # Modify this to control the frequency of metrics computation            
            interferers = target_dict['interferers']
            segments = target_dict['segment']
            if interferers.shape[1] != 1:
                raise ValueError("Only a single interferer is currently supported.")
            else:
                interferers = interferers[:, 0, 0]
            if model_output.device.type == "mps":
                # SDR calculation is not supported on mps
                model_output = model_output.cpu()
                segments = segments.cpu()
                interferers = interferers.cpu()

            # sdr = signal_distortion_ratio(model_output, segments)
            # log_dict[f"{prefix}_sdr"] = sdr.mean()
            
            # Compute si-sdr for target
            target_sisdr = scale_invariant_signal_distortion_ratio(model_output, segments)
            self.avg_sisdr = self._batch_moving_average(self.avg_sisdr, target_sisdr)
            log_dict[f"{prefix}_sisdr"] = self.avg_sisdr

            # Compute q-sdr
            ## Compute si-sdr for interferer
            interferer_sisdr = scale_invariant_signal_distortion_ratio(model_output, interferers)
            qsdr = (target_sisdr > interferer_sisdr).float()
            self.avg_qsdr = self._batch_moving_average(self.avg_qsdr, qsdr)
            log_dict[f"{prefix}_qsdr"] = self.avg_qsdr

        self.log_dict(log_dict, prog_bar=True)
        
        return loss

    def training_step(self, batch_data_dict, batch_idx):
        return self._step(batch_data_dict, batch_idx, prefix='train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix='val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix='test')
    
    def configure_optimizers(self):

        if self.optimizer_type == "AdamW":
            optimizer = optim.AdamW(
                params=self.ss_model.parameters(),
                lr=self.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=0.0,
                amsgrad=True,
            )
        else:
            raise NotImplementedError

        scheduler = LambdaLR(optimizer, self.lr_lambda_func)

        output_dict = {
            "optimizer": optimizer,
            "lr_scheduler": {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

        return output_dict
    
    def _batch_moving_average(self, current_avg, new_batch_values):
        for new_value in new_batch_values:
            current_avg = (1 - self.avg_smoothing) * current_avg + \
                self.avg_smoothing * new_value.item()

        return current_avg