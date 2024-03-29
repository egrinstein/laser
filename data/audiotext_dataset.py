import json
import random
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader


class AudioTextDataset(Dataset):
    """Can sample data from audio-text databases
    Params:
    sampling_rate: audio sampling rate
    max_clip_len: max length (seconds) of audio clip to be sampled
    """
    def __init__(
        self,
        datafiles=[''], 
        sampling_rate=32000, 
        max_clip_len=5,
    ):
        all_data_json = []
        for datafile in datafiles:
            with open(datafile, 'r') as fp:
                data_json = json.load(fp)['data']
                all_data_json.extend(data_json)
        self.all_data_json = all_data_json

        self.sampling_rate = sampling_rate
        self.max_length = max_clip_len * sampling_rate

    def __len__(self):
        return len(self.all_data_json)

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

    def _read_audio(self, index):
        try:
            audio_path = self.all_data_json[index]['wav']
            audio_data, audio_rate = torchaudio.load(audio_path, channels_first=True)
            text = self.all_data_json[index]['caption']

            # drop short utterance
            if audio_data.size(1) < self.sampling_rate * 1:
                raise Exception(f'{audio_path} is too short, drop it ...') 
            
            return text, audio_data, audio_rate
        
        except Exception as e:
            print(f'error: {e} occurs, when loading {audio_path}')
            random_index = random.randint(0, len(self.all_data_json)-1)
            return self._read_audio(index=random_index)

    def __getitem__(self, index):
        # create a audio tensor  
        text, audio_data, audio_rate = self._read_audio(index)
        audio_len = audio_data.shape[1] / audio_rate
        # convert stero to single channel
        if audio_data.shape[0] > 1:
            # audio_data: [samples]
            audio_data = (audio_data[0] + audio_data[1]) / 2
        else:
            audio_data = audio_data.squeeze(0)
        
        # resample audio clip
        if audio_rate != self.sampling_rate:
            audio_data = torchaudio.functional.resample(audio_data, orig_freq=audio_rate, new_freq=self.sampling_rate)
        
        audio_data = audio_data.unsqueeze(0)
        
        audio_data = self._cut_or_randomcrop(audio_data)            

        data_dict = {
            'text': text, 
            'waveform': audio_data,  
            'modality': 'audio_text'
        }

        return data_dict


def collate_fn(list_data_dict):
    r"""Collate mini-batch data to inputs and targets for training.

    Args:
        list_data_dict: e.g., [
            {
                'text': 'a sound of dog',
                'waveform': (1, samples),
                'modality': 'audio_text'
            }
            ...
            ]
    Returns:
        data_dict: e.g. 
            'audio_text': {
                'text': ['a sound of dog', ...]
                'waveform': (batch_size, 1, samples)
        }
    """
    
    at_list_data_dict = [
        data_dict for data_dict in list_data_dict if data_dict['modality'] == 'audio_text']

    at_data_dict = {}
    
    if len(at_list_data_dict) > 0:
        for key in at_list_data_dict[0].keys():
            at_data_dict[key] = [at_data_dict[key] for at_data_dict in at_list_data_dict]
            if key == 'waveform':
                at_data_dict[key] = torch.stack(at_data_dict[key])
            elif key == 'text':
                at_data_dict[key] = [text for text in at_data_dict[key]]

    
    return at_data_dict


class AudioTextDataLoader(DataLoader):
    def __init__(self, datafiles=[''], 
        sampling_rate=32000, 
        max_clip_len=5, *args, **kwargs):
        
        self._dataset = AudioTextDataset(
            datafiles=datafiles, 
            sampling_rate=sampling_rate, 
            max_clip_len=max_clip_len
        )

        super().__init__(self._dataset, collate_fn=collate_fn, *args, **kwargs)
