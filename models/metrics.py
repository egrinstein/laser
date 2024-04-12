import numpy as np
import torch

from .lassnet.stft import STFT

def energy(x):
    return torch.mean(x ** 2)


def magnitude_to_db(x):
    eps = 1e-10
    return 20. * np.log10(max(x, eps))


def db_to_magnitude(x):
    return 10. ** (x / 20)


def calculate_sdr(
    ref: np.ndarray,
    est: np.ndarray,
    eps=1e-10
) -> float:
    r"""Calculate SDR between reference and estimation.

    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """
    reference = ref
    noise = est - reference

    numerator = np.clip(a=np.mean(reference ** 2), a_min=eps, a_max=None)

    denominator = np.clip(a=np.mean(noise ** 2), a_min=eps, a_max=None)

    sdr = 10. * np.log10(numerator / denominator)

    return sdr


def calculate_sisdr(ref, est):
    r"""Calculate SDR between reference and estimation.

    Args:
        ref (np.ndarray), reference signal
        est (np.ndarray), estimated signal
    """

    eps = np.finfo(ref.dtype).eps

    reference = ref.copy()
    estimate = est.copy()
    
    reference = reference.reshape(reference.size, 1)
    estimate = estimate.reshape(estimate.size, 1)

    Rss = np.dot(reference.T, reference)
    # get the scaling factor for clean sources
    a = (eps + np.dot(reference.T, estimate)) / (Rss + eps)

    e_true = a * reference
    e_res = estimate - e_true

    Sss = (e_true**2).sum()
    Snn = (e_res**2).sum()

    sisdr = 10 * np.log10((eps+ Sss)/(eps + Snn))

    return sisdr 


class Loss(torch.nn.Module):
    def __init__(self, loss_type):
        super().__init__()
        self.type = loss_type

        if loss_type == 'l1_mag':
            self.stft = STFT()

    def forward(self, output_dict, target_dict):
        if self.type == 'l1_wav':
            return l1_wav(output_dict, target_dict)
        elif self.type == 'l1_mag':
            return self.l1_mag(output_dict, target_dict)

    def l1_mag(self, output_dict, target_dict):
        wav_target = target_dict['segment']
        mag_target = self.stft.transform(wav_target)[0]
        return l1(output_dict['magnitude'], mag_target)


def l1(output, target):
    return torch.mean(torch.abs(output - target))


def l1_wav(output_dict, target_dict):
    return l1(output_dict['segment'], target_dict['segment'])
