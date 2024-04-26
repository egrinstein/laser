from .modules import *
from .stft import STFT, TorchStft

import numpy as np
import torch


class UNetRes_FiLM(nn.Module):
    def __init__(self, channels=1, cond_embedding_dim=256, nsrc=1, only_train_film=False,
                 n_layers_embedding_mlp=2):
        super().__init__()
        activation = 'relu'
        momentum = 0.01

        self.n_fft = 1024
        self.hop_length = 512
        self.win_length = self.n_fft

        self.nsrc = nsrc
        self.channels = channels
        self.downsample_ratio = 2 ** 6  # This number equals 2^{#encoder_blocks}

        # Specialize the bert embeddings for improving conditioning
        self.text_embedding_mlp = Mlp(
            cond_embedding_dim, cond_embedding_dim, cond_embedding_dim,
                 n_layers_embedding_mlp, activation=activation, dropout=0, batch_norm=True,
                 output_activation=None)

        # Normalize magnitude input
        self.bn_mag = torch.nn.BatchNorm2d(1, momentum=momentum)

        self.encoder_block1 = EncoderBlockRes2BCond(in_channels=channels * nsrc, out_channels=32,
                                                    downsample=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)
        self.encoder_block2 = EncoderBlockRes2BCond(in_channels=32, out_channels=64,
                                                    downsample=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)
        self.encoder_block3 = EncoderBlockRes2BCond(in_channels=64, out_channels=128,
                                                    downsample=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)
        self.encoder_block4 = EncoderBlockRes2BCond(in_channels=128, out_channels=256,
                                                    downsample=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)
        self.encoder_block5 = EncoderBlockRes2BCond(in_channels=256, out_channels=384,
                                                    downsample=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)
        self.encoder_block6 = EncoderBlockRes2BCond(in_channels=384, out_channels=384,
                                                    downsample=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)
        self.conv_block7 = ConvBlockResCond(in_channels=384, out_channels=384,
                                            kernel_size=(3, 3), activation=activation, momentum=momentum,
                                            cond_embedding_dim=cond_embedding_dim)
        self.decoder_block1 = DecoderBlockRes2BCond(in_channels=384, out_channels=384,
                                                    stride=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)
        self.decoder_block2 = DecoderBlockRes2BCond(in_channels=384, out_channels=384,
                                                    stride=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)
        self.decoder_block3 = DecoderBlockRes2BCond(in_channels=384, out_channels=256,
                                                    stride=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)
        self.decoder_block4 = DecoderBlockRes2BCond(in_channels=256, out_channels=128,
                                                    stride=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)
        self.decoder_block5 = DecoderBlockRes2BCond(in_channels=128, out_channels=64,
                                                    stride=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)
        self.decoder_block6 = DecoderBlockRes2BCond(in_channels=64, out_channels=32,
                                                    stride=(2, 2), activation=activation, momentum=momentum,
                                                    cond_embedding_dim=cond_embedding_dim)

        self.after_conv_block1 = ConvBlockResCond(in_channels=32, out_channels=32,
                                                  kernel_size=(3, 3), activation=activation, momentum=momentum,
                                                  cond_embedding_dim=cond_embedding_dim)

        self.after_conv2 = nn.Conv2d(in_channels=32, out_channels=1,
                                     kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)

        self.init_weights()

        self.stft = TorchStft() #STFT()

        if only_train_film:
            for name, param in self.named_parameters():
                if "film" not in name:
                    param.requires_grad = False

    def init_weights(self):
        init_layer(self.after_conv2)

    def forward(self, input_dict):
        """
        Args:
          input: sp: (batch_size, channels_num, segment_samples)
        Outputs:
          output_dict: {
            'wav': (batch_size, channels_num, segment_samples),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """

        waveform = input_dict["mixture"][:, 0, :]
        if input_dict["condition"].dim() == 2:
            cond_vec = input_dict["condition"]
        else:
            cond_vec = input_dict["condition"][:, 0, :]

        #mag, phase = self.stft(waveform)
        mag, phase = self.stft.transform(waveform)
        mag = mag.unsqueeze(1)  # (bs, 1, F, T)
        mag = mag.permute(0, 1, 3, 2)  # (bs, 1, T, F)

        mag_norm = self.bn_mag(mag)

        #cond_vec = self.text_embedding_mlp(cond_vec)

        dec_cond_vec = cond_vec
        x = mag_norm

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]  # time_steps
        pad_len = int(np.ceil(x.shape[2] / self.downsample_ratio)) * self.downsample_ratio - origin_len
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        x = x[..., 0: x.shape[-1] - 2]  # (bs, channels, T, F)

        # UNet
        (x1_pool, x1) = self.encoder_block1(x, cond_vec)  # x1_pool: (bs, 32, T / 2, F / 2)
        (x2_pool, x2) = self.encoder_block2(x1_pool, cond_vec)  # x2_pool: (bs, 64, T / 4, F / 4)
        (x3_pool, x3) = self.encoder_block3(x2_pool, cond_vec)  # x3_pool: (bs, 128, T / 8, F / 8)
        (x4_pool, x4) = self.encoder_block4(x3_pool, dec_cond_vec)  # x4_pool: (bs, 256, T / 16, F / 16)
        (x5_pool, x5) = self.encoder_block5(x4_pool, dec_cond_vec)  # x5_pool: (bs, 512, T / 32, F / 32)
        (x6_pool, x6) = self.encoder_block6(x5_pool, dec_cond_vec)  # x6_pool: (bs, 1024, T / 64, F / 64)
        x_center = self.conv_block7(x6_pool, dec_cond_vec)  # (bs, 2048, T / 64, F / 64)
        x7 = self.decoder_block1(x_center, x6, dec_cond_vec)  # (bs, 1024, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, dec_cond_vec)  # (bs, 512, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, cond_vec)  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, cond_vec)  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, cond_vec)  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, cond_vec)  # (bs, 32, T, F)
        x = self.after_conv_block1(x12, cond_vec)  # (bs, 32, T, F)
        x = self.after_conv2(x)  # (bs, channels, T, F)

        # Recover shape
        x = F.pad(x, pad=(0, 2))
        x = x[:, :, 0: origin_len, :]

        x = torch.sigmoid(x)

        # Apply mask
        mag = x*mag
        mag = mag[:, 0].permute(0, 2, 1)

        # Recover waveform
        x = self.stft.inverse(mag, phase)

        return {"waveform": x, "magnitude": mag}


if __name__ == "__main__":
    model = UNetRes_FiLM(channels=1, cond_embedding_dim=16)
    cond_vec = torch.randn((1, 16))
    dec_vec = cond_vec
    print(model(torch.randn((1, 1, 1001, 513)), cond_vec, dec_vec).size())
