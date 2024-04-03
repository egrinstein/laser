import json
import os
import torchaudio

from torch.utils.data import Dataset, DataLoader
from safetensors.torch import load_file


class AudioTextMixDataset(Dataset):
    """Can sample data from audio-text-mix databases
    """
    def __init__(
        self,
        datafiles, sampling_rate=32000,
    ):
        all_data_json = []
        for datafile in datafiles:
            with open(datafile, 'r') as fp:
                data_json = json.load(fp)['data']
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json

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
        mix_audio_data, audio_rate = self._read_audio(self.all_data_json[index]['wav_mixture'])
        target_audio_data, audio_rate = self._read_audio(self.all_data_json[index]['wav_target'])
        interferer_audio_data, audio_rate = self._read_audio(self.all_data_json[index]['wav_interferer'])

        embeddings_path = self.all_data_json[index]['command_embedding']

        out_dict = {
            'input': {
                'mixture': mix_audio_data.squeeze(1),
                'condition': load_file(embeddings_path)['command'],
            },
            'target': {
                'interferers': interferer_audio_data,
                'segment': target_audio_data.squeeze(),
            }
        }

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
