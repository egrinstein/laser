import random
import torch
import lightning.pytorch as pl
import torch.nn as nn
import torch.optim as optim
from torchmetrics.functional.audio import signal_distortion_ratio, scale_invariant_signal_distortion_ratio

from huggingface_hub import PyTorchModelHubMixin
from torch.optim.lr_scheduler import LambdaLR

from commander import random_template_command


class AudioSep(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        query_encoder: nn.Module,
        waveform_mixer: nn.Module,
        ss_model: nn.Module = None,
        loss_function = None,
        optimizer_type: str = None,
        learning_rate: float = None,
        lr_lambda_func = None,
        use_text_ratio: float = 1.0,
        query_augmentation = True
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
        self.waveform_mixer = waveform_mixer
        self.query_encoder = query_encoder
        self.use_text_ratio = use_text_ratio
        self.loss_function = loss_function
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.lr_lambda_func = lr_lambda_func
        self.query_augmentation = query_augmentation

    def forward(self, x):
        pass

    def _step(self, batch_audio_text_dict, batch_idx, compute_sdr=True, sdr_freq=1, prefix='train'):
        r"""Forward a mini-batch data to model, calculate loss function, and
        train for one step. A mini-batch data is evenly distributed to multiple
        devices (if there are) for parallel training.

        Args:
            batch_audio_text_dict: e.g. 
            {
                    'text': ['a sound of dog', ...]
                    'waveform': (batch_size, 1, samples)
            }
            
        Returns:
            loss: float, loss function of this mini-batch
        """
        # [important] fix random seeds across devices
        random.seed(batch_idx)

        batch_text = batch_audio_text_dict['text']
        batch_audio = batch_audio_text_dict['waveform']
        
        mixtures, segments, interferers, mixture_texts = self.waveform_mixer(
            waveforms=batch_audio, texts=batch_text
        )

        # augment text data (convert caption such as "sound of dog" to "enhance sound of dog")
        if self.query_augmentation:
            z = list(zip(batch_text, mixture_texts))
            batch_text = [
                random_template_command(t, mt)
                for t, mt in z
            ]

        # calculate text embed for audio-text data
        conditions = self.query_encoder(
            modality='text',
            text=batch_text,
            audio=segments.squeeze(1),
            use_text_ratio=self.use_text_ratio,
        )

        input_dict = {
            'mixture': mixtures[:, None, :].squeeze(1),
            'condition': conditions,
        }

        target_dict = {
            'segment': segments.squeeze(1),
        }

        self.ss_model.train()
        model_output = self.ss_model(input_dict)['waveform']
        model_output = model_output.squeeze()
        # (batch_size, 1, segment_samples)

        output_dict = {
            'segment': model_output,
        }

        # Calculate loss
        loss = self.loss_function(output_dict, target_dict)

        log_dict = {f"{prefix}_loss": loss}
        
        if compute_sdr and batch_idx % sdr_freq == 0: # Modify this to control the frequency of metrics computation            
            if interferers.shape[1] != 1:
                raise ValueError("Only a single interferer is currently supported.")
            else:
                interferers = interferers[:, 0]

            if model_output.device.type == "mps":
                # SDR calculation is not supported on mps
                model_output = model_output.cpu()
                segments = segments.cpu()
                interferers = interferers.cpu()

            sdr = signal_distortion_ratio(model_output, segments.squeeze(1))
            sisdr = scale_invariant_signal_distortion_ratio(model_output, segments.squeeze(1))

            model_output_tensor = model_output.unsqueeze(1).repeat(1, interferers.size(1), 1)
            interferer_sisdr = scale_invariant_signal_distortion_ratio(model_output_tensor, interferers)
            pum = (interferer_sisdr > sisdr.unsqueeze(1)).any(dim=1).float()

            log_dict[f"{prefix}_sdr"] = sdr.mean()
            log_dict[f"{prefix}_sisdr"] = sisdr.mean()
            log_dict[f"{prefix}_pum"] = pum.mean()

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
    

def get_model_class(model_type):
    if model_type == 'ResUNet30':
        from models.resunet import ResUNet30
        return ResUNet30

    else:
        raise NotImplementedError
