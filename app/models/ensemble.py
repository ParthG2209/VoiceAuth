"""
Ensemble Voice Detector
Combines multiple detection methods for improved accuracy:
1. Feature-based analysis (MFCC, pitch, spectral)
2. Wav2Vec2 deep learning features
"""

import logging
from typing import Optional
from dataclasses import dataclass

from app.models.detector import voice_detector, DetectionResult
from app.models.wav2vec2_detector import wav2vec2_detector, Wav2Vec2DetectionResult
from app.utils.audio_processor import AudioFeatures

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Combined result from all detection methods"""
    classification: str
    confidence: float
    explanation: str
    
    # Individual model results
    feature_result: Optional[DetectionResult] = None
    wav2vec2_result: Optional[Wav2Vec2DetectionResult] = None


class EnsembleDetector:
    """
    Ensemble detector combining multiple methods
    
    Weights can be tuned based on validation performance
    """
    
    # Weights for ensemble (should sum to 1.0)
    FEATURE_WEIGHT = 0.4
    WAV2VEC2_WEIGHT = 0.6
    
    def __init__(self):
        """Initialize ensemble detector"""
        self.feature_detector = voice_detector
        self.wav2vec2_detector = wav2vec2_detector
        
        logger.info(
            f"EnsembleDetector initialized "
            f"(feature: {self.FEATURE_WEIGHT}, wav2vec2: {self.WAV2VEC2_WEIGHT})"
        )
    
    def detect(
        self, 
        features: AudioFeatures,
        language: str,
        use_wav2vec2: bool = True
    ) -> EnsembleResult:
        """
        Run ensemble detection
        
        Args:
            features: Extracted audio features
            language: Language of the audio
            use_wav2vec2: Whether to use Wav2Vec2 (may be slow on first run)
            
        Returns:
            EnsembleResult with combined classification
        """
        logger.info(f"Running ensemble detection for language: {language}")
        
        # Run feature-based detection
        feature_result = self.feature_detector.detect(features, language)
        
        # Initialize for potential wav2vec2 result
        wav2vec2_result = None
        
        if use_wav2vec2:
            try:
                # Run Wav2Vec2 detection
                wav2vec2_result = self.wav2vec2_detector.detect(
                    features.waveform,
                    features.sample_rate,
                    language
                )
            except Exception as e:
                logger.warning(f"Wav2Vec2 detection failed: {str(e)}")
        
        # Combine results
        if wav2vec2_result:
            # Convert classifications to scores (AI=1, Human=0)
            feature_ai_score = 1.0 if feature_result.classification == "AI_GENERATED" else 0.0
            wav2vec2_ai_score = 1.0 if wav2vec2_result.classification == "AI_GENERATED" else 0.0
            
            # Weight by confidence as well
            feature_weighted = (
                feature_ai_score * feature_result.confidence * self.FEATURE_WEIGHT
            )
            wav2vec2_weighted = (
                wav2vec2_ai_score * wav2vec2_result.confidence * self.WAV2VEC2_WEIGHT
            )
            
            # Normalize by total confidence weight
            total_weight = (
                feature_result.confidence * self.FEATURE_WEIGHT +
                wav2vec2_result.confidence * self.WAV2VEC2_WEIGHT
            )
            
            combined_ai_score = (feature_weighted + wav2vec2_weighted) / total_weight
            
            # Determine final classification
            if combined_ai_score > 0.5:
                classification = "AI_GENERATED"
                confidence = combined_ai_score
            else:
                classification = "HUMAN"
                confidence = 1 - combined_ai_score
            
            # Combine explanations
            explanations = []
            
            # Add the most confident explanation first
            if feature_result.confidence > wav2vec2_result.confidence:
                explanations.append(feature_result.explanation)
                if wav2vec2_result.explanation:
                    explanations.append(wav2vec2_result.explanation.split(";")[0])
            else:
                explanations.append(wav2vec2_result.explanation)
                if feature_result.explanation:
                    explanations.append(feature_result.explanation.split(";")[0])
            
            explanation = "; ".join(explanations)
            
        else:
            # Fallback to feature-based only
            classification = feature_result.classification
            confidence = feature_result.confidence
            explanation = feature_result.explanation
        
        logger.info(
            f"Ensemble result: {classification} (confidence: {confidence:.2f})"
        )
        
        return EnsembleResult(
            classification=classification,
            confidence=round(confidence, 2),
            explanation=explanation,
            feature_result=feature_result,
            wav2vec2_result=wav2vec2_result
        )


# Singleton instance
ensemble_detector = EnsembleDetector()
