import json
import os
import random
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file

from commander import CommandCreator
from models.clap_encoder import CLAP_Encoder


class AudioTextMixDataset(Dataset):
    """Can sample data from audio-text-mix databases
    """
    def __init__(
        self,
        datafiles, embeddings_dir, sampling_rate=32000, max_clip_len=5,
    ):
        all_data_json = []
        for datafile in datafiles:
            with open(datafile, 'r') as fp:
                data_json = json.load(fp)['data']
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json

        self.embeddings_dir = embeddings_dir

        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate

    def __len__(self):
        return len(self.all_data_json)

    def _read_audio(self, index):
        audio_path = self.all_data_json[index]['wav_mix']
        audio_data, audio_rate = torchaudio.load(audio_path, channels_first=True)

        # resample audio clip
        if audio_rate != self.sampling_rate:
            audio_data = torchaudio.functional.resample(
                audio_data, orig_freq=audio_rate, new_freq=self.sampling_rate)
        
        audio_data = audio_data.unsqueeze(0)
        
        return audio_data, audio_rate

    def __getitem__(self, index):
        # create a audio tensor  
        audio_data, audio_rate = self._read_audio(index)

        data_dict = {
            'waveform_mix': audio_data,
        }

        audio_path = self.all_data_json[index]['wav_mix']
        filename = os.path.basename(audio_path).split('.')[0] + '.safetensors'
        embeddings_path = os.path.join(self.embeddings_dir, filename)
        data_dict['command_embedding'] = load_file(embeddings_path)['command']
     
        return data_dict


class AudioTextMixDataLoader(DataLoader):
    def __init__(self, datafiles, 
        sampling_rate=32000, 
        max_clip_len=5, *args, **kwargs):
        
        self._dataset = AudioTextMixDataset(
            datafiles=datafiles, 
            sampling_rate=sampling_rate, 
            max_clip_len=max_clip_len
        )

        super().__init__(self._dataset, *args, **kwargs)
