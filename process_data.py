import os
import csv

speakers_txt = '/data/SPEAKERS.TXT'
root_dir = '/data/train-clean-100'
output_csv = '/data//train_dataset.csv'


speaker_info = {}
with open(speakers_txt, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line.startswith(';') or not line:
            continue
        parts = line.split('|')
        if len(parts) >= 2:
            speaker_id = parts[0].strip()
            sex = parts[1].strip()
            speaker_info[speaker_id] = sex


samples = []

for speaker_id in speaker_info.keys():
    speaker_dir = os.path.join(root_dir, speaker_id)
    if not os.path.exists(speaker_dir):
        continue

    flac_files = []
    for root, _, files in os.walk(speaker_dir):
        for file in files:
            if file.endswith('.flac'):
                flac_files.append(os.path.join(root, file))

    flac_files.sort()
    selected_files = flac_files[:30]

    for audio_path in selected_files:
        relative_path = audio_path.replace('data/', '', 1)
        samples.append([speaker_id, speaker_info[speaker_id], relative_path])

# . Ghi vào CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['speaker_id', 'sex', 'audio_path'])
    writer.writerows(samples)

print(f"Đã tạo file: {output_csv} với {len(samples)} dòng.")
