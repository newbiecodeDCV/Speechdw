import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import os

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform
        self.label_map =  {label: idx for idx, label in enumerate(self.data['gender'].unique())}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx]['file_path']
        label = self.label_map[self.data.iloc[idx]['gender']]
        file_path = os.path.join(self.audio_dir, file_path)
        waveform, sample_rate = torchaudio.load(file_path)
        if self.transform:
            waveform = self.transform(waveform)

        return waveform.squeeze(0).T, label  # (Time, Feature)

def get_collate_fn():
    def collate_fn(batch):
        waveforms, labels = zip(*batch)
        waveforms = pad_sequence(waveforms, batch_first=True)  # (B, T, F)
        labels = torch.tensor(labels)
        return waveforms.permute(0, 2, 1), labels  # (B, F, T)
    return collate_fn
