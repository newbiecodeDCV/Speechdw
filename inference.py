import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
import json
from datetime import datetime
import argparse
from pathlib import Path
from model import AudioClassifier


class InferenceConfig:

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


    num_classes = 2
    class_names = ['F', 'M  ']

    # Audio processing
    sample_rate = 16000
    n_fft = 1024
    hop_length = 256
    n_mels = 128

    # Model path
    save_dir = 'checkpoints'
    model_path = os.path.join(save_dir, 'best_model.pth')


class AudioInference:
    def __init__(self, config=InferenceConfig()):
        self.config = config
        self.device = config.device

        # Initialize transforms
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.n_mels,
            normalized=True,
            power=2.0
        )

        # Load model
        self.model = self.load_model()

    def load_model(self):
        """Load the trained model"""
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.config.model_path}")

        model = AudioClassifier(num_classes=self.config.num_classes).to(self.device)

        checkpoint = torch.load(self.config.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model

    def preprocess_audio(self, audio_path):
        """Preprocess audio file for inference"""
        try:
            waveform, orig_sample_rate = torchaudio.load(audio_path)

            # Resample if necessary
            if orig_sample_rate != self.config.sample_rate:
                resampler = T.Resample(orig_freq=orig_sample_rate, new_freq=self.config.sample_rate)
                waveform = resampler(waveform)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Apply mel spectrogram transform
            mel_spec = self.mel_transform(waveform)

            # Add batch dimension
            mel_spec = mel_spec.unsqueeze(0)  # Shape: [1, 1, n_mels, time]

            return mel_spec

        except Exception as e:
            raise RuntimeError(f"Error preprocessing audio {audio_path}: {str(e)}")

    def predict_single(self, audio_path):
        """Predict on a single audio file"""
        # Preprocess
        input_tensor = self.preprocess_audio(audio_path)
        input_tensor = input_tensor.to(self.device)

        # Inference
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

        result = {
            'file_path': audio_path,
            'predicted_class': predicted_class,
            'predicted_label': self.config.class_names[predicted_class],
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        }


        return result

    def predict_batch(self, audio_paths):
        """Predict on multiple audio files"""
        results = []
        print(f"🔍 Processing {len(audio_paths)} audio files...")

        for audio_path in audio_paths:
            try:
                result = self.predict_single(audio_path)
                results.append(result)

                print(
                    f"✅ {Path(audio_path).name}: {result['predicted_label']} (confidence: {result['confidence']:.3f})")

            except Exception as e:
                error_result = {
                    'file_path': audio_path,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
                print(f"❌ Error processing {audio_path}: {str(e)}")

        return results

    def predict_directory(self, directory_path, audio_extensions=None):
        """Predict on all audio files in a directory"""
        if audio_extensions is None:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']

        directory_path = Path(directory_path)
        audio_files = []

        for ext in audio_extensions:
            audio_files.extend(directory_path.glob(f"*{ext}"))
            audio_files.extend(directory_path.glob(f"*{ext.upper()}"))

        audio_files = [str(f) for f in audio_files]

        if not audio_files:
            print(f"⚠️ No audio files found in {directory_path}")
            return []

        print(f"📁 Found {len(audio_files)} audio files in {directory_path}")

        return self.predict_batch(audio_files)



def save_results(results, output_path):
    """Save inference results to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Results saved to {output_path}")


def print_summary(results):
    """Print summary of inference results"""
    total_files = len(results)
    successful_predictions = len([r for r in results if 'predicted_class' in r])
    errors = total_files - successful_predictions

    print(f"\n📊 INFERENCE SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total files processed: {total_files}")
    print(f"Successful predictions: {successful_predictions}")
    print(f"Errors: {errors}")

    if successful_predictions > 0:
        # Count predictions by class
        class_counts = {}
        for result in results:
            if 'predicted_label' in result:
                label = result['predicted_label']
                class_counts[label] = class_counts.get(label, 0) + 1

        print(f"\nPrediction distribution:")
        for label, count in class_counts.items():
            percentage = (count / successful_predictions) * 100
            print(f"  {label}: {count} files ({percentage:.1f}%)")


# ==============================================================================
# MAIN FUNCTION
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Audio Classification Inference")
    parser.add_argument('--input', '-i', required=True,
                        help='Input audio file or directory path')
    parser.add_argument('--output', '-o',
                        help='Output JSON file path (optional)')
    parser.add_argument('--model', '-m',
                        help='Model checkpoint path (optional)')
    parser.add_argument('--batch', action='store_true',
                        help='Process directory in batch mode')

    args = parser.parse_args()

    # Update model path if provided
    if args.model:
        InferenceConfig.model_path = args.model

    try:
        # Initialize inference
        inference = AudioInference()

        input_path = Path(args.input)

        # Determine processing mode
        if input_path.is_file():
            # Single file
            print(f"🎵 Processing single file: {input_path}")
            results = [inference.predict_single(str(input_path))]

        elif input_path.is_dir():
            # Directory
            print(f"📁 Processing directory: {input_path}")
            results = inference.predict_directory(str(input_path))

        else:
            raise FileNotFoundError(f"Input path not found: {input_path}")

        # Print results
        print(f"\n🎉 Inference completed!")
        for result in results:
            if 'predicted_class' in result:
                print(f"📄 {Path(result['file_path']).name}")
                print(f"   Prediction: {result['predicted_label']}")
                print(f"   Confidence: {result['confidence']:.3f}")
                if 'probabilities' in result:
                    print(f"   Probabilities: {result['probabilities']}")
                print()

        # Print summary
        print_summary(results)

        # Save results if output path specified
        if args.output:
            save_results(results, args.output)

    except KeyboardInterrupt:
        print("\n⚠️ Inference interrupted by user")
    except Exception as e:
        print(f"❌ Inference failed with error: {str(e)}")
        raise



if __name__ == "__main__":
    main()