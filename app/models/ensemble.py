"""
Ensemble Voice Detector - V3
Uses custom-trained AI/Human voice classifier as the primary detection method
"""

import logging
from typing import Optional
from dataclasses import dataclass

from app.models.custom_classifier import custom_classifier, ClassificationResult
from app.utils.audio_processor import AudioFeatures

logger = logging.getLogger(__name__)


@dataclass
class EnsembleResult:
    """Result from voice detection"""
    classification: str
    confidence: float
    explanation: str
    
    # Detailed result for debugging
    classifier_result: Optional[ClassificationResult] = None


class EnsembleDetector:
    """
    Voice detector using custom-trained classifier
    
    Uses a custom Keras model specifically trained on AI vs Human voice samples.
    """
    
    def __init__(self):
        """Initialize ensemble detector"""
        self.classifier = custom_classifier
        logger.info("EnsembleDetector V3 initialized with custom classifier")
    
    def detect(
        self, 
        features: AudioFeatures,
        language: str,
        use_wav2vec2: bool = True  # Kept for API compatibility
    ) -> EnsembleResult:
        """
        Run voice detection
        
        Args:
            features: Extracted audio features
            language: Language of the audio
            use_wav2vec2: Ignored (kept for API compatibility)
            
        Returns:
            EnsembleResult with classification
        """
        logger.info(f"Running voice detection for language: {language}")
        
        # Run custom classifier
        try:
            classifier_result = self.classifier.detect(
                features.waveform,
                features.sample_rate,
                language
            )
            
            logger.info(
                f"Custom classifier result: {classifier_result.classification} "
                f"(confidence: {classifier_result.confidence:.2f})"
            )
            
            classification = classifier_result.classification
            confidence = classifier_result.confidence
            explanation = classifier_result.explanation
            
        except Exception as e:
            logger.error(f"Custom classifier failed: {str(e)}", exc_info=True)
            classifier_result = None
            classification = "HUMAN"
            confidence = 0.5
            explanation = f"Classification error: {str(e)}"
        
        # Ensure confidence is in valid range
        confidence = max(0.0, min(1.0, confidence))
        
        logger.info(
            f"Detection result: {classification} (confidence: {confidence:.2f})"
        )
        
        return EnsembleResult(
            classification=classification,
            confidence=round(confidence, 2),
            explanation=explanation,
            classifier_result=classifier_result
        )


# Singleton instance
ensemble_detector = EnsembleDetector()
