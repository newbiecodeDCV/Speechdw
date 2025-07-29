import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import AudioClassifier
from utils.audio_dataset import AudioDataset,get_collate_fn
import torchaudio


# 1. Khởi tạo thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Khởi tạo model
model = AudioClassifier(num_classes=2).to(device)

# 3. Loss và Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)

# 4. Dataloader (giả sử bạn đã có train_loader và val_loader)
# Ví dụ: train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
mel_spec = torchaudio.transforms.MelSpectrogram(
    sample_rate=16000,
    n_fft=1024,
    hop_length=512,
    n_mels=64
)
dataset = AudioDataset(csv_path=r'/data/train_dataset.csv', audio_dir='/data', transform=mel_spec)
vaild_set = AudioDataset(csv_path=r'/data/dev_dataset.csv',audio_dir='/data', transform=mel_spec)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True,collate_fn=get_collate_fn(),num_workers=4)
valid_loader = DataLoader(vaild_set, batch_size=32, shuffle=True,collate_fn=get_collate_fn(),num_workers=4)


best_val_acc = 0.0

# 5. Vòng lặp huấn luyện
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Thống kê
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total

    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}")

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_loss /= len(valid_loader.dataset)
    val_acc = val_correct / val_total
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
