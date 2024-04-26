import random
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio

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
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func

        # Average will be computed using exponential moving average
        self.avg_sisdr = 0
        self.avg_qsdr = 0
        self.avg_loss = 0
        self.avg_sisdr_tp = 0
        self.avg_smoothing = 0.01

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
        model_output = self.ss_model(input_dict)

        waveform = model_output['waveform'].squeeze(1)
        # (batch_size, 1, segment_samples)

        output_dict = {'segment': waveform}
        if 'magnitude' in model_output.keys():
            output_dict['magnitude'] = model_output['magnitude']
            
        # Calculate loss
        loss = self.loss_function(output_dict, target_dict)
        if batch_idx == 0:
            self.avg_loss = loss.item()
        else:
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
            if waveform.device.type == "mps":
                # SDR calculation is not supported on mps
                waveform = waveform.cpu()
                segments = segments.cpu()
                interferers = interferers.cpu()

            # sdr = signal_distortion_ratio(waveform, segments)
            # log_dict[f"{prefix}_sdr"] = sdr.mean()
            
            # Compute si-sdr for target
            target_sisdr = scale_invariant_signal_distortion_ratio(waveform, segments)
            self.avg_sisdr = self._batch_moving_average(self.avg_sisdr, target_sisdr, batch_idx == 0)
            log_dict[f"{prefix}_sisdr"] = self.avg_sisdr

            # Compute q-sdr
            ## Compute si-sdr for interferer
            interferer_sisdr = scale_invariant_signal_distortion_ratio(waveform, interferers)
            qsdr = (target_sisdr > interferer_sisdr).float()
            self.avg_qsdr = self._batch_moving_average(self.avg_qsdr, qsdr, batch_idx == 0)
            log_dict[f"{prefix}_qsdr"] = self.avg_qsdr

            # Compute tp-sdr: SDR for samples x where q-sdr(x) == 1
            tp_idxs = qsdr == 1
            tp_sisdr = target_sisdr[tp_idxs]
            if len(tp_sisdr) != 0:
                self.avg_sisdr_tp = self._batch_moving_average(self.avg_sisdr_tp, tp_sisdr, batch_idx == 0)
                log_dict[f"{prefix}_sisdr_tp"] = self.avg_sisdr_tp


        self.log_dict(log_dict, prog_bar=True)
        
        return loss

    def training_step(self, batch_data_dict, batch_idx):
        return self._step(batch_data_dict, batch_idx, prefix='train')
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix='val')

    def test_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, prefix='test')
    
    def configure_optimizers(self):
        optimizer = optim.Adam(
            params=self.ss_model.parameters(),
        )

        output_dict = {
            "optimizer": optimizer,
        }

        return output_dict
    
    def _batch_moving_average(self, current_avg, new_batch_values, is_first_batch=False):
        if is_first_batch:
            return new_batch_values.mean()

        for new_value in new_batch_values:
            current_avg = (1 - self.avg_smoothing) * current_avg + \
                self.avg_smoothing * new_value.item()

        return current_avg