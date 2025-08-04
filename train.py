import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from tqdm import tqdm
import numpy as np
import os
from model import AudioClassifier
from utils.audio_dataset import AudioDataset, get_collate_fn
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import json
from datetime import datetime


# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    # Device
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # Model
    num_classes = 2
    dropout_rate = 0.15

    # Training
    num_epochs = 25
    batch_size = 64
    learning_rate = 0.001
    weight_decay = 0.005

    # Data
    train_csv = '/data/train_dataset.csv'
    dev_csv = '/data/dev_dataset.csv'
    audio_dir = '/data'

    # Transforms
    sample_rate = 16000
    n_fft = 1024
    hop_length = 256  #
    n_mels = 128

    # Training settings
    patience = 8
    save_dir = 'checkpoints'
    log_dir = 'logs'


class AudioAugmentation:
    def __init__(self, sample_rate=16000, training=True):
        self.training = training
        if training:
            self.time_mask = T.TimeMasking(time_mask_param=15)
            self.freq_mask = T.FrequencyMasking(freq_mask_param=8)

    def __call__(self, mel_spec):
        if not self.training:
            return mel_spec

        # Time masking
        if torch.rand(1) > 0.3:
            mel_spec = self.time_mask(mel_spec)

        # Frequency masking
        if torch.rand(1) > 0.3:
            mel_spec = self.freq_mask(mel_spec)

        # Add noise
        if torch.rand(1) > 0.8:
            noise = torch.randn_like(mel_spec) * 0.01
            mel_spec = mel_spec + noise

        # Amplitude scaling
        if torch.rand(1) > 0.7:
            scale = torch.empty(1).uniform_(0.85, 1.15)
            mel_spec = mel_spec * scale

        return mel_spec


def get_transforms(training=True):
    mel_transform = T.MelSpectrogram(
        sample_rate=Config.sample_rate,
        n_fft=Config.n_fft,
        hop_length=Config.hop_length,
        n_mels=Config.n_mels,
        normalized=True,
        power=2.0
    )

    augmentation = AudioAugmentation(training=training)

    def transform_fn(waveform):
        mel_spec = mel_transform(waveform)
        mel_spec = augmentation(mel_spec)
        return mel_spec

    return transform_fn


# ==============================================================================
# FOCAL LOSS FOR BETTER PERFORMANCE
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def restore_weights(self, model):
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.learning_rates = []

    def update(self, train_loss, train_acc, val_loss, val_acc, lr):
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        self.learning_rates.append(lr)

    def save_plots(self, save_dir):
        epochs = range(1, len(self.train_losses) + 1)

        # Loss plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, self.val_losses, 'r-', label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accs, 'b-', label='Train Acc')
        plt.plot(epochs, self.val_accs, 'r-', label='Val Acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()


# ==============================================================================
# MAIN TRAINING FUNCTION
# ==============================================================================
def train_model():
    # Create directories
    os.makedirs(Config.save_dir, exist_ok=True)
    os.makedirs(Config.log_dir, exist_ok=True)

    print(f"🚀 Training on device: {Config.device}")
    print(f"📊 Training for {Config.num_epochs} epochs")

    # Model setup
    model = AudioClassifier(num_classes=Config.num_classes).to(Config.device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Cho balanced data

    optimizer = optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
    )

    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=True
    )

    # Data loaders
    train_transform = get_transforms(training=True)
    val_transform = get_transforms(training=False)

    train_dataset = AudioDataset(
        csv_path=Config.train_csv,
        audio_dir=Config.audio_dir,
        transform=train_transform
    )
    val_dataset = AudioDataset(
        csv_path=Config.dev_csv,
        audio_dir=Config.audio_dir,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        collate_fn=get_collate_fn(),
        num_workers=8,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size * 2,
        shuffle=False,
        collate_fn=get_collate_fn(),
        num_workers=4,
        pin_memory=True
    )

    print(f"📚 Train samples: {len(train_dataset)}")
    print(f"📝 Validation samples: {len(val_dataset)}")

    # Training utilities
    early_stopping = EarlyStopping(patience=Config.patience, min_delta=0.001)
    metrics_tracker = MetricsTracker()
    scaler = torch.cuda.amp.GradScaler()  # Mixed precision

    best_val_acc = 0.0
    start_time = datetime.now()

    # Training loop
    for epoch in range(Config.num_epochs):
        print(f"\n{'=' * 60}")
        print(f"Epoch {epoch + 1}/{Config.num_epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # ==================== TRAINING ====================
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_pbar = tqdm(train_loader, desc="Training", leave=False)
        for batch_idx, (inputs, labels) in enumerate(train_pbar):
            inputs, labels = inputs.to(Config.device, non_blocking=True), labels.to(Config.device, non_blocking=True)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total

        # ==================== VALIDATION ====================
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=False)
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(Config.device, non_blocking=True), labels.to(Config.device,
                                                                                        non_blocking=True)

                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress bar
                current_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.2f}%'
                })

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total

        # Update learning rate
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Track metrics
        metrics_tracker.update(avg_train_loss, train_acc, avg_val_loss, val_acc, current_lr)

        # Print epoch results
        print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': avg_val_loss,
            }, os.path.join(Config.save_dir, 'best_model.pth'))
            print(f"✅ New best model saved! Val Acc: {best_val_acc:.2f}%")

        # Regular checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(Config.save_dir, f'checkpoint_epoch_{epoch + 1}.pth'))

        # Early stopping
        if early_stopping(avg_val_loss, model):
            print(f"🛑 Early stopping triggered at epoch {epoch + 1}")
            early_stopping.restore_weights(model)
            break

    # Training completed
    training_time = datetime.now() - start_time
    print(f"\n🎉 Training completed!")
    print(f"⏱️ Total training time: {training_time}")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")

    # Save training history
    history = {
        'train_losses': metrics_tracker.train_losses,
        'train_accs': metrics_tracker.train_accs,
        'val_losses': metrics_tracker.val_losses,
        'val_accs': metrics_tracker.val_accs,
        'learning_rates': metrics_tracker.learning_rates,
        'best_val_acc': best_val_acc,
        'training_time': str(training_time),
        'config': {
            'num_epochs': Config.num_epochs,
            'batch_size': Config.batch_size,
            'learning_rate': Config.learning_rate,
            'weight_decay': Config.weight_decay,
        }
    }

    with open(os.path.join(Config.log_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # Save training curves
    metrics_tracker.save_plots(Config.log_dir)

    return model


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        model = train_model()
        print("✅ Training script completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"❌ Training failed with error: {str(e)}")
        raise

