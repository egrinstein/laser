import json
import random
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader

from commander import random_template_command
from data.mixing.waveform_mixer import WaveformMixer
from models.clap_encoder import CLAP_Encoder


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


class AudioTextDataLoader(DataLoader):
    def __init__(self, datafiles=[''], 
        sampling_rate=32000, 
        max_clip_len=5, 
        max_mix_num=2,
        lower_db=-40,
        higher_db=0,
        query_augmentation=True,
        device=None, *args, **kwargs):
        
        self._dataset = AudioTextDataset(
            datafiles=datafiles, 
            sampling_rate=sampling_rate, 
            max_clip_len=max_clip_len
        )

        self.waveform_mixer = WaveformMixer(
            max_mix_num=max_mix_num,
            lower_db=lower_db, 
            higher_db=higher_db
        )

        self.query_encoder = CLAP_Encoder().eval().to(device)


        self.device = device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.query_augmentation = query_augmentation
        super().__init__(self._dataset, collate_fn=self.collate_fn, *args, **kwargs)


    def collate_fn(self, list_data_dict):
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
        
        # Move data to device (gpu/mps/cpu)
        at_data_dict = _dict_to_device(at_data_dict, self.device)
        
        mixtures, segments, interferers, mixture_texts = self.waveform_mixer(
            waveforms=at_data_dict['waveform'], texts=at_data_dict['text']
        )

        # augment text data (convert caption such as "sound of dog" to "enhance sound of dog")
        if self.query_augmentation:
            z = list(zip(at_data_dict['text'], mixture_texts))
            batch_text = [
                random_template_command(t, mt)
                for t, mt in z
            ]

        # calculate text embed for audio-text data
        conditions = self.query_encoder(
            modality='text',
            text=batch_text,
            audio=segments.squeeze(1),
        )

        return {
            'input': {
                'mixture': mixtures[:, None, :].squeeze(1),
                'condition': conditions,
            },
            'target': {
                'interferers': interferers,
                'segment': segments.squeeze(1)
            }
        }
    

def _dict_to_device(data_dict, device):
    r"""Move data_dict to device.
    """
    
    for key, value in data_dict.items():
        if isinstance(value, torch.Tensor):
            data_dict[key] = value.to(device)
        elif isinstance(value, list):
            if isinstance(value[0], torch.Tensor):
                data_dict[key] = [v.to(device) for v in value]
            else:
                data_dict[key] = value
        else:
            data_dict[key] = value
    
    return data_dict
