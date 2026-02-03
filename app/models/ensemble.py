"""
Ensemble Voice Detector - V2
Combines multiple detection methods for maximum accuracy:
1. Feature-based analysis (MFCC, pitch, spectral)
2. Wav2Vec2 deep learning features
3. Pre-trained Deepfake Detection model (PRIMARY)
"""

import logging
from typing import Optional
from dataclasses import dataclass

from app.models.detector import voice_detector, DetectionResult
from app.models.deepfake_detector import deepfake_detector, DeepfakeDetectionResult
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
    deepfake_result: Optional[DeepfakeDetectionResult] = None


class EnsembleDetector:
    """
    Ensemble detector combining multiple methods
    
    Priority:
    1. Pre-trained Deepfake Model (most reliable when available)
    2. Feature-based analysis (fallback)
    
    The deepfake model is specifically trained on real/fake audio
    and should be much more accurate than general features.
    """
    
    # When deepfake model is available, use it primarily
    DEEPFAKE_WEIGHT = 0.75
    FEATURE_WEIGHT = 0.25
    
    def __init__(self):
        """Initialize ensemble detector"""
        self.feature_detector = voice_detector
        self.deepfake_detector = deepfake_detector
        
        logger.info(
            f"EnsembleDetector V2 initialized "
            f"(deepfake: {self.DEEPFAKE_WEIGHT}, feature: {self.FEATURE_WEIGHT})"
        )
    
    def detect(
        self, 
        features: AudioFeatures,
        language: str,
        use_wav2vec2: bool = True  # Now uses deepfake model instead
    ) -> EnsembleResult:
        """
        Run ensemble detection
        
        Args:
            features: Extracted audio features
            language: Language of the audio
            use_wav2vec2: If True, use pre-trained deepfake model (recommended)
            
        Returns:
            EnsembleResult with combined classification
        """
        logger.info(f"Running ensemble V2 detection for language: {language}")
        
        # Run feature-based detection
        feature_result = self.feature_detector.detect(features, language)
        
        # Initialize deepfake result
        deepfake_result = None
        
        if use_wav2vec2:
            try:
                # Run pre-trained deepfake detection (primary model)
                deepfake_result = self.deepfake_detector.detect(
                    features.waveform,
                    features.sample_rate,
                    language
                )
                logger.info(
                    f"Deepfake model result: {deepfake_result.classification} "
                    f"({deepfake_result.confidence:.2f})"
                )
            except Exception as e:
                logger.warning(f"Deepfake detection failed: {str(e)}")
        
        # Combine results
        if deepfake_result and deepfake_result.confidence > 0.5:
            # Deepfake model is available and confident
            
            # Convert classifications to AI scores
            feature_ai_score = 1.0 if feature_result.classification == "AI_GENERATED" else 0.0
            deepfake_ai_score = 1.0 if deepfake_result.classification == "AI_GENERATED" else 0.0
            
            # Weighted combination - prioritize deepfake model
            combined_ai_score = (
                deepfake_ai_score * deepfake_result.confidence * self.DEEPFAKE_WEIGHT +
                feature_ai_score * feature_result.confidence * self.FEATURE_WEIGHT
            )
            
            # Normalize
            total_weight = (
                deepfake_result.confidence * self.DEEPFAKE_WEIGHT +
                feature_result.confidence * self.FEATURE_WEIGHT
            )
            
            if total_weight > 0:
                combined_ai_score = combined_ai_score / total_weight
            
            # Final classification
            if combined_ai_score > 0.5:
                classification = "AI_GENERATED"
                confidence = combined_ai_score
            else:
                classification = "HUMAN"
                confidence = 1 - combined_ai_score
            
            # Use deepfake model's explanation primarily
            explanation = deepfake_result.explanation
            if feature_result.classification == deepfake_result.classification:
                explanation += f"; {feature_result.explanation}"
            else:
                explanation += f" (feature analysis disagrees: {feature_result.explanation})"
            
        else:
            # Fallback to feature-based only
            classification = feature_result.classification
            confidence = feature_result.confidence
            explanation = feature_result.explanation
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        logger.info(
            f"Ensemble V2 result: {classification} (confidence: {confidence:.2f})"
        )
        
        return EnsembleResult(
            classification=classification,
            confidence=round(confidence, 2),
            explanation=explanation,
            feature_result=feature_result,
            deepfake_result=deepfake_result
        )


# Singleton instance
ensemble_detector = EnsembleDetector()
