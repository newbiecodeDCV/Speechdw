from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel
import uvicorn
import logging
from model import AudioClassifier


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Response models
class PredictionResponse(BaseModel):
    file_name: str
    predicted_class: int
    predicted_label: str
    confidence: float
    timestamp: str
    success: bool
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    total_files: int
    successful_predictions: int
    errors: int
    summary: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    timestamp: str


# Configuration class
class InferenceConfig:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_classes = 2
    class_names = ['F', 'M']

    # Audio processing
    sample_rate = 16000
    n_fft = 1024
    hop_length = 256
    n_mels = 128

    # Model path
    save_dir = 'checkpoints'
    model_path = os.path.join(save_dir, 'best_model.pth')



class AudioInferenceAPI:
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
        self.model = None
        self.model_loaded = False
        self.load_model()

    def load_model(self):
        try:
            if not os.path.exists(self.config.model_path):
                logger.warning(f"Model checkpoint not found at {self.config.model_path}")
                self.model = AudioClassifier(num_classes=self.config.num_classes).to(self.device)
                self.model.eval()
                self.model_loaded = True
                logger.info("Using dummy model for demonstration")
                return

            self.model = AudioClassifier(num_classes=self.config.num_classes).to(self.device)
            checkpoint = torch.load(self.config.model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.model_loaded = True
            logger.info(f"Model loaded successfully from {self.config.model_path}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model_loaded = False
            raise

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
            raise RuntimeError(f"Error preprocessing audio: {str(e)}")

    def predict_single(self, audio_path: str, file_name: str = None):
        """Predict on a single audio file"""
        if not self.model_loaded:
            raise HTTPException(status_code=500, detail="Model not loaded")

        try:
            # Preprocess
            input_tensor = self.preprocess_audio(audio_path)
            input_tensor = input_tensor.to(self.device)

            # Inference
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()

            result = PredictionResponse(
                file_name=file_name or Path(audio_path).name,
                predicted_class=predicted_class,
                predicted_label=self.config.class_names[predicted_class],
                confidence=confidence,
                timestamp=datetime.now().isoformat(),
                success=True
            )

            return result

        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
            return PredictionResponse(
                file_name=file_name or Path(audio_path).name,
                predicted_class=-1,
                predicted_label="Error",
                confidence=0.0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error=str(e)
            )


# Initialize the inference engine
inference_engine = AudioInferenceAPI()

# Create FastAPI app
app = FastAPI(
    title="Audio Classification API",
    description="API for gender classification from audio files",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Audio Classification API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Check API health",
            "/predict": "Predict single audio file",
            "/predict/batch": "Predict multiple audio files",
            "/docs": "API documentation"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if inference_engine.model_loaded else "unhealthy",
        model_loaded=inference_engine.model_loaded,
        device=str(inference_engine.device),
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_audio(file: UploadFile = File(...)):
    """
    Predict gender from a single audio file

    - **file**: Audio file (wav, mp3, flac, m4a, ogg)
    """
    # Check file extension
    allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']
    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {allowed_extensions}"
        )

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        try:
            # Copy uploaded file to temp file
            shutil.copyfileobj(file.file, temp_file)
            temp_file.flush()

            # Predict
            result = inference_engine.predict_single(temp_file.name, file.filename)

            return result

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_audio_batch(files: List[UploadFile] = File(...)):
    """
    Predict gender from multiple audio files

    - **files**: List of audio files (wav, mp3, flac, m4a, ogg)
    """
    if len(files) > 20:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 20 files allowed per batch")

    results = []
    allowed_extensions = ['.wav', '.mp3', '.flac', '.m4a', '.ogg']

    for file in files:
        file_extension = Path(file.filename).suffix.lower()

        if file_extension not in allowed_extensions:
            # Add error result for unsupported format
            error_result = PredictionResponse(
                file_name=file.filename,
                predicted_class=-1,
                predicted_label="Error",
                confidence=0.0,
                timestamp=datetime.now().isoformat(),
                success=False,
                error=f"Unsupported file format: {file_extension}"
            )
            results.append(error_result)
            continue

        # Process file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            try:
                shutil.copyfileobj(file.file, temp_file)
                temp_file.flush()

                result = inference_engine.predict_single(temp_file.name, file.filename)
                results.append(result)

            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                error_result = PredictionResponse(
                    file_name=file.filename,
                    predicted_class=-1,
                    predicted_label="Error",
                    confidence=0.0,
                    timestamp=datetime.now().isoformat(),
                    success=False,
                    error=str(e)
                )
                results.append(error_result)

            finally:
                try:
                    os.unlink(temp_file.name)
                except:
                    pass

    # Generate summary
    successful_predictions = len([r for r in results if r.success])
    errors = len(results) - successful_predictions

    # Count predictions by class
    class_counts = {}
    for result in results:
        if result.success:
            label = result.predicted_label
            class_counts[label] = class_counts.get(label, 0) + 1

    summary = {
        "class_distribution": class_counts,
        "success_rate": successful_predictions / len(results) if results else 0
    }

    return BatchPredictionResponse(
        results=results,
        total_files=len(results),
        successful_predictions=successful_predictions,
        errors=errors,
        summary=summary
    )


@app.get("/info")
async def get_model_info():
    """Get model and configuration information"""
    return {
        "model_info": {
            "num_classes": inference_engine.config.num_classes,
            "class_names": inference_engine.config.class_names,
            "model_loaded": inference_engine.model_loaded,
            "device": str(inference_engine.device)
        },
        "audio_config": {
            "sample_rate": inference_engine.config.sample_rate,
            "n_fft": inference_engine.config.n_fft,
            "hop_length": inference_engine.config.hop_length,
            "n_mels": inference_engine.config.n_mels
        },
        "supported_formats": [".wav", ".mp3", ".flac", ".m4a", ".ogg"]
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",  # Replace "main" with the actual filename
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
