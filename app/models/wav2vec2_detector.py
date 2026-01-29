"""
Wav2Vec2-based Voice Detector
Uses Facebook's Wav2Vec2 model for deep feature extraction
Combined with custom classifier for AI voice detection
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class Wav2Vec2DetectionResult:
    """Result from Wav2Vec2-based detection"""
    classification: str
    confidence: float
    explanation: str
    model_scores: dict = field(default_factory=dict)


class Wav2Vec2Classifier(nn.Module):
    """
    Custom classifier head for AI voice detection
    Takes Wav2Vec2 features and outputs binary classification
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super().__init__()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, 2)  # 2 classes: AI, Human
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization"""
        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            features: [batch, seq_len, hidden_dim] from Wav2Vec2
        Returns:
            logits: [batch, 2] for AI/Human classification
        """
        # Global average pooling over sequence
        pooled = features.mean(dim=1)  # [batch, hidden_dim]
        return self.classifier(pooled)


class Wav2Vec2VoiceDetector:
    """
    Pre-trained Wav2Vec2 model for voice feature extraction
    Combined with heuristic classifier for AI detection
    """
    
    MODEL_NAME = "facebook/wav2vec2-base"
    TARGET_SAMPLE_RATE = 16000
    
    def __init__(self, use_gpu: bool = None):
        """
        Initialize the Wav2Vec2 detector
        
        Args:
            use_gpu: Whether to use GPU if available
        """
        self.use_gpu = use_gpu if use_gpu is not None else settings.use_gpu
        self.device = torch.device(
            "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        )
        
        self.processor = None
        self.model = None
        self.classifier = None
        self._loaded = False
        
        logger.info(f"Wav2Vec2VoiceDetector initialized (device: {self.device})")
    
    def load_model(self) -> bool:
        """
        Load the Wav2Vec2 model from Hugging Face
        Downloads on first run, then cached locally
        
        Returns:
            True if loaded successfully
        """
        if self._loaded:
            return True
        
        try:
            logger.info(f"Loading Wav2Vec2 model: {self.MODEL_NAME}")
            
            # Load processor (handles audio preprocessing)
            self.processor = Wav2Vec2Processor.from_pretrained(
                self.MODEL_NAME,
                cache_dir=settings.model_cache_dir
            )
            
            # Load pre-trained model
            self.model = Wav2Vec2Model.from_pretrained(
                self.MODEL_NAME,
                cache_dir=settings.model_cache_dir
            )
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Initialize classifier head
            self.classifier = Wav2Vec2Classifier(
                input_dim=self.model.config.hidden_size
            )
            self.classifier.to(self.device)
            self.classifier.eval()
            
            self._loaded = True
            logger.info("Wav2Vec2 model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load Wav2Vec2 model: {str(e)}")
            return False
    
    def extract_features(
        self, 
        waveform: np.ndarray, 
        sample_rate: int
    ) -> Optional[torch.Tensor]:
        """
        Extract deep features using Wav2Vec2
        
        Args:
            waveform: Audio waveform as numpy array
            sample_rate: Sample rate of the audio
            
        Returns:
            Feature tensor from Wav2Vec2 or None if failed
        """
        if not self._loaded:
            if not self.load_model():
                return None
        
        try:
            # Resample if necessary
            if sample_rate != self.TARGET_SAMPLE_RATE:
                waveform_tensor = torch.from_numpy(waveform).float()
                resampler = torchaudio.transforms.Resample(
                    sample_rate, 
                    self.TARGET_SAMPLE_RATE
                )
                waveform = resampler(waveform_tensor).numpy()
            
            # Process audio through Wav2Vec2 processor
            inputs = self.processor(
                waveform,
                sampling_rate=self.TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            input_values = inputs.input_values.to(self.device)
            
            # Extract features
            with torch.no_grad():
                outputs = self.model(input_values)
                # Use last hidden state
                features = outputs.last_hidden_state
            
            return features
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return None
    
    def analyze_features(
        self, 
        features: torch.Tensor
    ) -> Tuple[float, dict]:
        """
        Analyze Wav2Vec2 features for AI voice characteristics
        
        Uses statistical analysis of embeddings:
        - AI voices often have more uniform feature distributions
        - Human voices show more variability and dynamics
        
        Args:
            features: Wav2Vec2 output features [1, seq_len, hidden_dim]
            
        Returns:
            Tuple of (ai_probability, analysis_dict)
        """
        features_np = features.cpu().numpy().squeeze(0)  # [seq_len, hidden_dim]
        
        # Feature statistics
        feature_mean = np.mean(features_np, axis=0)
        feature_std = np.std(features_np, axis=0)
        
        # Temporal dynamics - how much features change over time
        temporal_diff = np.diff(features_np, axis=0)
        temporal_variance = np.var(temporal_diff)
        
        # Feature distribution analysis
        # AI voices tend to have more clustered/consistent features
        mean_feature_std = np.mean(feature_std)
        feature_range = np.max(features_np) - np.min(features_np)
        
        # Embedding space analysis
        # Calculate cosine similarity between consecutive frames
        norms = np.linalg.norm(features_np, axis=1, keepdims=True)
        normalized = features_np / (norms + 1e-8)
        
        if len(normalized) > 1:
            similarities = []
            for i in range(len(normalized) - 1):
                sim = np.dot(normalized[i], normalized[i + 1])
                similarities.append(sim)
            mean_similarity = np.mean(similarities)
            similarity_std = np.std(similarities)
        else:
            mean_similarity = 1.0
            similarity_std = 0.0
        
        # AI detection heuristics based on feature analysis
        scores = {
            "temporal_variance": temporal_variance,
            "mean_feature_std": mean_feature_std,
            "feature_range": feature_range,
            "mean_similarity": mean_similarity,
            "similarity_std": similarity_std
        }
        
        # Scoring logic
        # Low temporal variance suggests AI (too consistent)
        temporal_score = 0.7 if temporal_variance < 0.1 else 0.3
        
        # High similarity between frames suggests AI (repetitive patterns)
        similarity_score = 0.7 if mean_similarity > 0.95 else 0.4
        
        # Low similarity variation suggests AI (uniform transitions)
        variation_score = 0.65 if similarity_std < 0.02 else 0.35
        
        # Weighted combination
        ai_probability = (
            temporal_score * 0.35 +
            similarity_score * 0.35 +
            variation_score * 0.30
        )
        
        return ai_probability, scores
    
    def detect(
        self, 
        waveform: np.ndarray, 
        sample_rate: int,
        language: str
    ) -> Wav2Vec2DetectionResult:
        """
        Run full detection pipeline
        
        Args:
            waveform: Audio waveform
            sample_rate: Sample rate
            language: Language of the audio
            
        Returns:
            Detection result with classification and confidence
        """
        logger.info(f"Running Wav2Vec2 detection for language: {language}")
        
        # Extract features
        features = self.extract_features(waveform, sample_rate)
        
        if features is None:
            logger.warning("Feature extraction failed, using fallback")
            return Wav2Vec2DetectionResult(
                classification="HUMAN",
                confidence=0.5,
                explanation="Unable to extract deep features, using default classification",
                model_scores={}
            )
        
        # Analyze features
        ai_probability, scores = self.analyze_features(features)
        
        # Generate explanation
        explanations = []
        
        if scores["temporal_variance"] < 0.1:
            explanations.append("Low temporal variation in speech embeddings")
        else:
            explanations.append("Natural temporal dynamics detected")
        
        if scores["mean_similarity"] > 0.95:
            explanations.append("High frame-to-frame consistency")
        
        if scores["similarity_std"] < 0.02:
            explanations.append("Uniform embedding transitions")
        
        # Final classification
        if ai_probability > 0.5:
            classification = "AI_GENERATED"
            confidence = min(ai_probability + 0.1, 0.99)
        else:
            classification = "HUMAN"
            confidence = min((1 - ai_probability) + 0.1, 0.99)
        
        explanation = "; ".join(explanations[:2])
        
        logger.info(
            f"Wav2Vec2 detection: {classification} "
            f"(confidence: {confidence:.2f})"
        )
        
        return Wav2Vec2DetectionResult(
            classification=classification,
            confidence=round(confidence, 2),
            explanation=explanation,
            model_scores=scores
        )


# Singleton instance
wav2vec2_detector = Wav2Vec2VoiceDetector()
