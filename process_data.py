import os
import csv

speakers_txt = '/storage/asr/data/LibriSpeech/SPEAKERS.TXT'
root_dir = '/storage/asr/data/LibriSpeech/dev-clean'
output_csv = '/storage/asr/data/LibriSpeech/dev_dataset.csv'

# 1. Đọc speaker_id và sex từ SPEAKERS.TXT
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

# 2. Duyệt thư mục audio và lấy 3 file đầu tiên cho mỗi speaker
samples = []

for speaker_id in speaker_info.keys():
    speaker_dir = os.path.join(root_dir, speaker_id)
    if not os.path.exists(speaker_dir):
        continue

    # Lấy tất cả các file .flac dưới thư mục speaker_id
    flac_files = []
    for root, _, files in os.walk(speaker_dir):
        for file in files:
            if file.endswith('.flac'):
                flac_files.append(os.path.join(root, file))

    # Sắp xếp và chọn 3 file đầu tiên (nếu có)
    flac_files.sort()
    selected_files = flac_files[:8]

    for audio_path in selected_files:
        relative_path = audio_path.replace('data/', '', 1)
        samples.append([speaker_id, speaker_info[speaker_id], relative_path])

# 3. Ghi vào CSV
with open(output_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['speaker_id', 'sex', 'audio_path'])
    writer.writerows(samples)

print(f"Đã tạo file: {output_csv} với {len(samples)} dòng.")
