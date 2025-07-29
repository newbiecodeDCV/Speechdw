import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
import os

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, transform=None,augment=False):
        self.data = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform
        self.label_map =  {label: idx for idx, label in enumerate(self.data['sex'].unique())}
        self.augment = augment
        if self.augment:
                self.augment_transform = torch.nn.Sequential(
                    T.Vol(gain=(0.5, 1.5), gain_type="amplitude"),
                    T.FrequencyMasking(freq_mask_param=15),
                    T.TimeMasking(time_mask_param=35),
                )
        else:
                self.augment_transform = None
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx]['audio_path'].lstrip('/')
        label = self.label_map[self.data.iloc[idx]['sex']]
        file_path = os.path.join(self.audio_dir, file_path)
        waveform, sample_rate = torchaudio.load(file_path)
        if self.augment_transform:
            waveform = self.augment_transform(waveform)

        if self.transform:
            waveform = self.transform(waveform)

        return waveform.squeeze(0).T, label  # (Time, Feature)
    def __getinfo__(self):
        return self.label_map

def get_collate_fn():
    def collate_fn(batch):
        waveforms, labels = zip(*batch)
        waveforms = pad_sequence(waveforms, batch_first=True)  # (B, T, F)
        labels = torch.tensor(labels)
        return waveforms.permute(0, 2, 1), labels  # (B, F, T)
    return collate_fn
