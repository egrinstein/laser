import random
import numpy as np
import torch
import torch.nn as nn
import pyloudnorm as pyln

from typing import List


class WaveformMixer(nn.Module):
    def __init__(self, max_mix_num=2, lower_db=-10, higher_db=10,
                 sampling_rate=32000, max_clip_len=5,):
        super().__init__()

        self.max_mix_num = max_mix_num
        self.loudness_param = {
            'lower_db': lower_db,
            'higher_db': higher_db,
        }

        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate

    def __call__(self, waveforms, texts=None):
        batch_size = waveforms.shape[0]

        data_dict = {
            'segment': [],
            'mixture': [],
            'interferers': [],
        }

        mixed_texts = []

        for n in range(0, batch_size):
            segment = waveforms[n].clone()

            mix_num = random.randint(2, self.max_mix_num)

            noise_track_idxs = self.get_noise_track_idxs(texts, n, mix_num)
            noise_waveforms = [waveforms[i] for i in noise_track_idxs]
            mixed_texts_n = [texts[i] for i in noise_track_idxs]

            mixture = self.mix(segment, noise_waveforms)

            data_dict['segment'].append(segment)
            data_dict['mixture'].append(mixture)
            data_dict['interferers'].append(torch.stack(noise_waveforms))
            if texts is not None:
                mixed_texts.append(mixed_texts_n)

        for key in data_dict.keys():
            data_dict[key] = torch.stack(data_dict[key], dim=0)

        # return data_dict
        output = (data_dict['mixture'], data_dict['segment'], data_dict['interferers'])
        if texts is not None:
            output += (mixed_texts,)
        return output
    
    def get_noise_track_idxs(self, texts: List[str], n_target, mix_num: int):
        """
        Get noise tracks to mix with the target track.
        This method guarantees that the noise tracks have different classes from the target track.

        The first candidate noise track is the succeeding track of n_target in the batch.
        If the class of the succeeding track is the same as the target track, the next track is selected.
        This process is repeated until a noise track with a different class is found.
        """

        n_available_tracks = len(texts)
        
        n_tracks_added = 1
        n_track_to_add = (n_target + 1) % n_available_tracks
        mixed_texts_n = []
        track_ids_to_add = []

        # Find track ids to add in the batch
        while n_tracks_added < mix_num:
            if texts is not None:
                # Make sure the track text (i.e., class) being added
                # is different from the class of the target track
                if texts[n_target] == texts[n_track_to_add]:
                    n_track_to_add = (n_track_to_add + 1) % n_available_tracks
                    if n_track_to_add == n_target:
                        # If all the tracks have the same class, raise an exception (to avoid infinite loop)
                        raise ValueError("All the tracks in the batch have the same class.")
                    continue

                mixed_texts_n.append(texts[n_track_to_add])

            track_ids_to_add.append(n_track_to_add)
            n_tracks_added += 1

        return track_ids_to_add

    def mix(self, target_track: torch.Tensor, noise_tracks: List[torch.Tensor], return_tracks=False):
        # crop target and noise waveforms, and convert to mono

        target_track = self._cut_or_randomcrop(self._to_mono(target_track))
        noise_tracks = [self._cut_or_randomcrop(self._to_mono(noise)) for noise in noise_tracks]

        # create zero tensors as the background template
        noise = torch.zeros_like(target_track)
        # add tracks to the mixture
        for next_segment in noise_tracks:
            rescaled_next_segment = dynamic_loudnorm(audio=next_segment, reference=target_track, **self.loudness_param)
            noise += rescaled_next_segment

        # randomly normalize background noise
        noise = dynamic_loudnorm(audio=noise, reference=target_track, **self.loudness_param)

        # create audio mixture
        mixture = target_track + noise

        # apply declipping, if needed
        max_value = torch.max(torch.abs(mixture))
        if max_value > 1:
            target_track *= 0.9 / max_value
            mixture *= 0.9 / max_value
        
        if return_tracks:
            return mixture, target_track, noise_tracks
        else:
            return mixture

    def _cut_or_randomcrop(self, waveform):
        # waveform: [1, samples]
        # random crop
        if waveform.size(1) > self.max_length:
            random_idx = random.randint(0, waveform.size(1)-self.max_length)
            waveform = waveform[:, random_idx:random_idx+self.max_length]
        else:
            temp_wav = torch.zeros(1, self.max_length)
            temp_wav[:, 0:waveform.size(1)] = waveform
            waveform = temp_wav

        assert waveform.size(1) == self.max_length, \
            f"number of audio samples is {waveform.size(1)}"

        return waveform
    
    def _to_mono(self, audio: torch.Tensor):
        # convert stereo to single channel
        if len(audio.shape) == 1:
            return audio.unsqueeze(0)
        elif audio.shape[0] > 1:
            # audio: [samples]
            audio = (audio[0] + audio[1]) / 2
            audio = audio.unsqueeze(0)

        return audio


def rescale_to_match_energy(segment1, segment2):
    ratio = get_energy_ratio(segment1, segment2)
    rescaled_segment1 = segment1 / ratio
    return rescaled_segment1 


def get_energy(x):
    return torch.mean(x ** 2)


def get_energy_ratio(segment1, segment2):
    energy1 = get_energy(segment1)
    energy2 = max(get_energy(segment2), 1e-10)
    ratio = (energy1 / energy2) ** 0.5
    ratio = torch.clamp(ratio, 0.02, 50)
    return ratio


def dynamic_loudnorm(audio, reference, lower_db=-10, higher_db=10): 
    rescaled_audio = rescale_to_match_energy(audio, reference)
    
    delta_loudness = random.randint(lower_db, higher_db)

    gain = np.power(10.0, delta_loudness / 20.0)

    return gain * rescaled_audio


def torch_to_numpy(tensor):
    """Convert a PyTorch tensor to a NumPy array."""
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise ValueError("Input must be a PyTorch tensor.")


def numpy_to_torch(array):
    """Convert a NumPy array to a PyTorch tensor."""
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    else:
        raise ValueError("Input must be a NumPy array.")


# decayed
def random_loudness_norm(audio, lower_db=-35, higher_db=-15, sr=32000):
    device = audio.device
    audio = torch_to_numpy(audio.squeeze(0))
    # randomly select a norm volume
    norm_vol = random.randint(lower_db, higher_db)

    # measure the loudness first 
    meter = pyln.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(audio)
    # loudness normalize audio
    normalized_audio = pyln.normalize.loudness(audio, loudness, norm_vol)

    normalized_audio = numpy_to_torch(normalized_audio).unsqueeze(0)
    
    return normalized_audio.to(device)
    
