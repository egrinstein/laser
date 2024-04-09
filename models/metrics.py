import numpy as np
import torch


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


def l1(output, target):
    return torch.mean(torch.abs(output - target))


def l1_wav(output_dict, target_dict):
    return l1(output_dict['segment'], target_dict['segment'])


def l1_mag(output_dict, target_dict):
    wav_target = target_dict['segment']
    window = torch.hann_window(1024).to(wav_target.device)
    mag_target = torch.stft(
        wav_target, n_fft=1024,
        hop_length=512, win_length=1024,
        window=window,
        return_complex=True)
    
    return l1(output_dict['magnitude'], mag_target)


def get_loss_function(loss_type):
    if loss_type == "l1_wav":
        return l1_wav
    elif loss_type == "l1_mag":
        return l1_mag
    else:
        raise NotImplementedError("Error!")
