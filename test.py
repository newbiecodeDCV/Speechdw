import os
import torch
import torchaudio
from model import AudioClassifier
from torchaudio.transforms import Resample, MelSpectrogram
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AudioClassifier(num_classes=2).to(device)
checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Mel spectrogram settings
target_sample_rate = 16000
mel_transform = MelSpectrogram(
    sample_rate=target_sample_rate,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)

speaker_info_path = "/data/SPEAKERS.TXT"
speaker_info = {}
with open(speaker_info_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line.startswith(';') or not line:
            continue
        parts = line.split('|')
        if len(parts) >= 2:
            speaker_id = parts[0].strip()
            sex = parts[1].strip()
            label = 0 if sex == 'F' else 1
            speaker_info[speaker_id] = label

def preprocess_audio(path):
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != target_sample_rate:
        resample = Resample(orig_freq=sample_rate, new_freq=target_sample_rate)
        waveform = resample(waveform)
    mel_spec = mel_transform(waveform)  # (1, 64, T)

    max_len = 128
    if mel_spec.size(-1) < max_len:
        mel_spec = F.pad(mel_spec, (0, max_len - mel_spec.size(-1)))
    else:
        mel_spec = mel_spec[:, :, :max_len]

    mel_spec = mel_spec.unsqueeze(0)  # (1, 1, 64, 128)
    return mel_spec.to(device)

root_dir = "/data/test-clean"
correct = 0
total = 0

for speaker_id in os.listdir(root_dir):
    speaker_path = os.path.join(root_dir, speaker_id)
    if not os.path.isdir(speaker_path):
        continue
    if speaker_id not in speaker_info:
        continue
    label = speaker_info[speaker_id]

    for chapter in os.listdir(speaker_path):
        chapter_path = os.path.join(speaker_path, chapter)
        for file in os.listdir(chapter_path):
            if file.endswith(".flac"):
                file_path = os.path.join(chapter_path, file)
                try:
                    input_tensor = preprocess_audio(file_path)
                    with torch.no_grad():
                        output = model(input_tensor)
                        pred = output.argmax(dim=1).item()
                    total += 1
                    if pred == label:
                        correct += 1
                except Exception as e:
                    print(f"Lỗi {file_path}: {e}")

acc = correct / total * 100 if total > 0 else 0
print(f"\n✅ Test Accuracy: {acc:.2f}% ({correct}/{total})")
