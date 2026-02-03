"""
Pre-trained Deepfake Audio Detection Model
Uses MelodyMachine/Deepfake-audio-detection-V2 from Hugging Face
This model is specifically trained to detect AI-generated voices
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass 
class DeepfakeDetectionResult:
    """Result from deepfake detection model"""
    classification: str  # AI_GENERATED or HUMAN
    confidence: float    # 0.0 to 1.0
    explanation: str
    raw_scores: dict


class DeepfakeDetector:
    """
    Pre-trained Deepfake Audio Detection using Hugging Face model
    
    Model: MelodyMachine/Deepfake-audio-detection-V2
    - Fine-tuned on deepfake audio datasets
    - Wav2Vec2-based architecture
    - Binary classification: fake vs real
    """
    
    MODEL_NAME = "MelodyMachine/Deepfake-audio-detection-V2"
    TARGET_SAMPLE_RATE = 16000
    
    def __init__(self, use_gpu: bool = None):
        """Initialize the deepfake detector"""
        self.use_gpu = use_gpu if use_gpu is not None else settings.use_gpu
        self.device = torch.device(
            "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        )
        
        self.feature_extractor = None
        self.model = None
        self._loaded = False
        
        logger.info(f"DeepfakeDetector initialized (device: {self.device})")
    
    def load_model(self) -> bool:
        """
        Load the pre-trained deepfake detection model
        Downloads from Hugging Face on first run (~360MB)
        """
        if self._loaded:
            return True
        
        try:
            logger.info(f"Loading deepfake detection model: {self.MODEL_NAME}")
            
            # Load feature extractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.MODEL_NAME,
                cache_dir=settings.model_cache_dir
            )
            
            # Load model
            self.model = AutoModelForAudioClassification.from_pretrained(
                self.MODEL_NAME,
                cache_dir=settings.model_cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            
            self._loaded = True
            logger.info("Deepfake detection model loaded successfully")
            logger.info(f"Model labels: {self.model.config.id2label}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load deepfake model: {str(e)}")
            return False
    
    def preprocess_audio(
        self, 
        waveform: np.ndarray, 
        sample_rate: int
    ) -> Optional[torch.Tensor]:
        """
        Preprocess audio for the model
        
        Args:
            waveform: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Preprocessed input tensor
        """
        try:
            # Convert to torch tensor
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform).float()
            
            # Resample if necessary
            if sample_rate != self.TARGET_SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(
                    sample_rate, 
                    self.TARGET_SAMPLE_RATE
                )
                waveform = resampler(waveform)
            
            # Ensure 1D
            if waveform.dim() > 1:
                waveform = waveform.squeeze()
            
            # Truncate or pad to max length (30 seconds)
            max_length = self.TARGET_SAMPLE_RATE * 30
            if len(waveform) > max_length:
                waveform = waveform[:max_length]
            
            return waveform.numpy()
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {str(e)}")
            return None
    
    def detect(
        self, 
        waveform: np.ndarray, 
        sample_rate: int,
        language: str
    ) -> DeepfakeDetectionResult:
        """
        Run deepfake detection on audio
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            language: Language of the audio (unused but kept for API consistency)
            
        Returns:
            Detection result with classification and confidence
        """
        logger.info(f"Running deepfake detection for language: {language}")
        
        # Load model if not loaded
        if not self._loaded:
            if not self.load_model():
                return DeepfakeDetectionResult(
                    classification="HUMAN",
                    confidence=0.5,
                    explanation="Model not available, using fallback",
                    raw_scores={}
                )
        
        # Preprocess audio
        processed_audio = self.preprocess_audio(waveform, sample_rate)
        
        if processed_audio is None:
            return DeepfakeDetectionResult(
                classification="HUMAN",
                confidence=0.5,
                explanation="Audio preprocessing failed",
                raw_scores={}
            )
        
        try:
            # Extract features
            inputs = self.feature_extractor(
                processed_audio,
                sampling_rate=self.TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            input_values = inputs.input_values.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_values)
                logits = outputs.logits
                
                # Apply softmax to get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Get prediction
                predicted_class = torch.argmax(probs, dim=-1).item()
                
            # Get class probabilities
            probs_np = probs.cpu().numpy()[0]
            
            # Map model labels to our format
            # The model typically has labels like: {0: 'fake', 1: 'real'} or similar
            id2label = self.model.config.id2label
            
            raw_scores = {
                id2label.get(i, f"class_{i}"): float(probs_np[i])
                for i in range(len(probs_np))
            }
            
            logger.info(f"Raw model scores: {raw_scores}")
            
            # Determine classification based on model output
            # Check which label is "fake" or "deepfake" or "spoof"
            fake_keywords = ['fake', 'spoof', 'synthetic', 'deepfake', 'ai', 'generated']
            real_keywords = ['real', 'genuine', 'human', 'bonafide', 'authentic']
            
            fake_prob = 0.0
            real_prob = 0.0
            
            for label, prob in raw_scores.items():
                label_lower = label.lower()
                if any(kw in label_lower for kw in fake_keywords):
                    fake_prob = max(fake_prob, prob)
                elif any(kw in label_lower for kw in real_keywords):
                    real_prob = max(real_prob, prob)
            
            # If no clear mapping found, use class indices
            if fake_prob == 0.0 and real_prob == 0.0:
                # Assume class 0 is fake, class 1 is real (common pattern)
                fake_prob = probs_np[0] if len(probs_np) > 0 else 0.5
                real_prob = probs_np[1] if len(probs_np) > 1 else 0.5
            
            # Final classification
            if fake_prob > real_prob:
                classification = "AI_GENERATED"
                confidence = fake_prob
                explanation = f"Deepfake model detected synthetic voice patterns (confidence: {fake_prob:.1%})"
            else:
                classification = "HUMAN"
                confidence = real_prob
                explanation = f"Deepfake model detected authentic human voice (confidence: {real_prob:.1%})"
            
            logger.info(
                f"Deepfake detection: {classification} "
                f"(confidence: {confidence:.2f})"
            )
            
            return DeepfakeDetectionResult(
                classification=classification,
                confidence=round(confidence, 2),
                explanation=explanation,
                raw_scores=raw_scores
            )
            
        except Exception as e:
            logger.error(f"Deepfake detection failed: {str(e)}", exc_info=True)
            return DeepfakeDetectionResult(
                classification="HUMAN",
                confidence=0.5,
                explanation=f"Detection error: {str(e)}",
                raw_scores={}
            )


# Singleton instance
deepfake_detector = DeepfakeDetector()
