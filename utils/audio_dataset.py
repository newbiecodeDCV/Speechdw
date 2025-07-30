import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio.transforms as T
import os

class AudioDataset(Dataset):
    def __init__(self, csv_path, audio_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.transform = transform
        self.label_map =  {label: idx for idx, label in enumerate(self.data['sex'].unique())}
        print(self.label_map)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        file_path = self.data.iloc[idx]['audio_path'].lstrip('/')
        label = self.label_map[self.data.iloc[idx]['sex']]
        file_path = os.path.join(self.audio_dir, file_path)
        waveform, sample_rate = torchaudio.load(file_path)
        if self.transform:
            features = self.transform(waveform)  # (1, 64, T')
        else:
            features = waveform

        return features, label


def get_collate_fn():
    def collate_fn(batch):
        features, labels = zip(*batch)
        max_len = max(f.shape[-1] for f in features)

        padded = [
            torch.nn.functional.pad(f, (0, max_len - f.shape[-1]))  # Pad bên phải (time axis)
            for f in features
        ]

        x = torch.stack(padded)  # (B, 1, 64, T')
        y = torch.tensor(labels)

        return x, y
    return collate_fn

