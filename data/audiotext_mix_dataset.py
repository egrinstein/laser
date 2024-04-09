import json
import os
import torchaudio

from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file


class AudioTextMixDataset(Dataset):
    """Can sample data from audio-text-mix databases"""
    def __init__(
        self,
        datafiles, sampling_rate=32000,
    ):
        all_data_json = []
        for datafile in datafiles:
            with open(datafile, 'r') as fp:
                data_json = json.load(fp)['data']
                all_data_json.extend(data_json)
        self.all_data_json = _filter_missing_files(all_data_json, mode='all')

        self.sampling_rate = sampling_rate

    def __len__(self):
        return len(self.all_data_json)

    def _read_audio(self, audio_path):
        audio_data, audio_rate = torchaudio.load(audio_path, channels_first=True)

        # resample audio clip
        if audio_rate != self.sampling_rate:
            audio_data = torchaudio.functional.resample(
                audio_data, orig_freq=audio_rate, new_freq=self.sampling_rate)
        
        audio_data = audio_data.unsqueeze(0)
        
        return audio_data, audio_rate

    def __getitem__(self, index):
        # create a audio tensor
        sample = self.all_data_json[index]
        mix_audio_data, audio_rate = self._read_audio(sample['wav_mixture'])
        target_audio_data, audio_rate = self._read_audio(sample['wav_target'])
        interferer_audio_data, audio_rate = self._read_audio(sample['wav_interferer'])
        
        out_dict = {
            'input': {
                'mixture': mix_audio_data.squeeze(1),
                'caption_target': sample['caption_target'],
            },
            'target': {
                'interferers': interferer_audio_data,
                'segment': target_audio_data.squeeze(),
            }
        }

        if 'command_embedding' in sample:
            out_dict['input']['condition'] = load_file(
                sample['command_embedding'])['command']

        return out_dict


class AudioTextMixDataLoader(DataLoader):
    def __init__(self, datafiles, 
        sampling_rate=32000, 
        *args, **kwargs):
        
        self._dataset = AudioTextMixDataset(
            datafiles=datafiles, 
            sampling_rate=sampling_rate,
        )

        super().__init__(self._dataset, *args, **kwargs)


def _filter_missing_files(data_json, mode='all'):
    filtered_data_json = []

    for sample in data_json:
        if 'wav_mixture' not in sample: 
            continue
        else:
            filtered_data_json.append(sample)

    print("Missing samples:", len(data_json) - len(filtered_data_json),
          "out of", len(data_json))
    
    # Filter by mode
    if mode == 'positive':
        return [data for data in filtered_data_json if data['command_type'] == 'positive']
    elif mode == 'negative':
        return [data for data in filtered_data_json if data['command_type'] == 'negative']
    elif mode == 'remove_mixed':
        return [data for data in filtered_data_json if data['command_type'] != 'mixed']
    else:
        return filtered_data_json
