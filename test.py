import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os
from model import AudioClassifier
from utils.audio_dataset import AudioDataset, get_collate_fn
import torchaudio.transforms as T
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

    # Data
    test_csv = '/data/test_dataset.csv'
    audio_dir = '/data'

    # Transforms
    sample_rate = 16000
    n_fft = 1024
    hop_length = 256
    n_mels = 128

    # Testing settings
    batch_size = 128  # Larger batch size for testing
    save_dir = 'checkpoints'
    log_dir = 'logs'
    model_path = os.path.join(save_dir, 'best_model.pth')


def get_transforms():
    mel_transform = T.MelSpectrogram(
        sample_rate=Config.sample_rate,
        n_fft=Config.n_fft,
        hop_length=Config.hop_length,
        n_mels=Config.n_mels,
        normalized=True,
        power=2.0
    )

    def transform_fn(waveform):
        mel_spec = mel_transform(waveform)
        return mel_spec

    return transform_fn


# ==============================================================================
# EVALUATION FUNCTION
# ==============================================================================
def evaluate_model():
    # Create directories
    os.makedirs(Config.log_dir, exist_ok=True)

    print(f"🚀 Evaluating on device: {Config.device}")

    # Model setup
    model = AudioClassifier(num_classes=Config.num_classes).to(Config.device)

    # Load best model
    if not os.path.exists(Config.model_path):
        raise FileNotFoundError(f"Model checkpoint not found at {Config.model_path}")

    checkpoint = torch.load(Config.model_path, map_location=Config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(
        f"✅ Loaded best model from epoch {checkpoint['epoch']} with validation accuracy: {checkpoint['val_acc']:.2f}%")

    # Data loader
    test_transform = get_transforms()
    test_dataset = AudioDataset(
        csv_path=Config.test_csv,
        audio_dir=Config.audio_dir,
        transform=test_transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        collate_fn=get_collate_fn(),
        num_workers=4,
        pin_memory=True
    )

    print(f"📝 Test samples: {len(test_dataset)}")

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Evaluation
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []

    start_time = datetime.now()

    with torch.no_grad():
        test_pbar = tqdm(test_loader, desc="Testing", leave=False)
        for inputs, labels in test_pbar:
            inputs, labels = inputs.to(Config.device, non_blocking=True), labels.to(Config.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            current_acc = 100. * test_correct / test_total
            test_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.2f}%'
            })

    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100. * test_correct / test_total
    test_time = datetime.now() - start_time

    # Print results
    print(f"\n🎉 Testing completed!")
    print(f"⏱️ Total testing time: {test_time}")
    print(f"📊 Test Loss: {avg_test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

    # Save results
    results = {
        'test_loss': avg_test_loss,
        'test_accuracy': test_acc,
        'testing_time': str(test_time),
        'model_checkpoint': Config.model_path,
        'test_samples': len(test_dataset),
        'config': {
            'batch_size': Config.batch_size,
            'num_classes': Config.num_classes,
        }
    }

    with open(os.path.join(Config.log_dir, 'test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    return test_acc, avg_test_loss


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    try:
        test_acc, test_loss = evaluate_model()
        print(f"✅ Testing script completed successfully!")
        print(f"🏆 Final test accuracy: {test_acc:.2f}%")
    except KeyboardInterrupt:
        print("\n⚠️ Testing interrupted by user")
    except Exception as e:
        print(f"❌ Testing failed with error: {str(e)}")
        raise