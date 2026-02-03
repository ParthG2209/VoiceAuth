"""
AI Voice Detection Model - Feature Analysis Module
Provides supplementary feature-based analysis for voice detection
"""

import logging
from typing import Tuple
from dataclasses import dataclass
import numpy as np
from scipy import stats

from app.utils.audio_processor import AudioFeatures

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from feature-based voice detection analysis"""
    classification: str  # AI_GENERATED or HUMAN
    confidence: float    # 0.0 to 1.0
    explanation: str     # Human-readable reason
    feature_scores: dict  # Detailed scores for debugging


class VoiceDetector:
    """
    Feature-based voice analysis module
    
    Provides supplementary analysis based on audio features.
    The primary detection is now handled by the custom classifier.
    """
    
    def __init__(self):
        """Initialize detector"""
        self.model_loaded = True
        logger.info("VoiceDetector initialized for supplementary feature analysis")
    
    def analyze_features(self, features: AudioFeatures) -> dict:
        """
        Analyze audio features and return analysis scores
        
        Args:
            features: Extracted audio features
            
        Returns:
            Dictionary of feature analysis scores
        """
        scores = {}
        
        # Pitch analysis
        pitch_std = features.pitch_std
        if pitch_std < 20.0:
            scores["pitch"] = {"score": 0.9, "note": "Very smooth pitch (possible AI)"}
        elif pitch_std < 45.0:
            scores["pitch"] = {"score": 0.7, "note": "Consistent pitch patterns"}
        else:
            scores["pitch"] = {"score": 0.3, "note": "Natural pitch variation"}
        
        # Energy analysis
        energy_var = features.energy_variance
        if energy_var < 0.0001:
            scores["energy"] = {"score": 0.85, "note": "Flat energy envelope"}
        elif energy_var < 0.001:
            scores["energy"] = {"score": 0.65, "note": "Smooth energy dynamics"}
        else:
            scores["energy"] = {"score": 0.25, "note": "Natural energy dynamics"}
        
        # Spectral analysis
        centroid = features.spectral_centroid
        if len(centroid) > 0:
            centroid_cv = np.std(centroid) / (np.mean(centroid) + 1e-6)
            if centroid_cv < 0.15:
                scores["spectral"] = {"score": 0.8, "note": "Uniform spectral characteristics"}
            else:
                scores["spectral"] = {"score": 0.3, "note": "Natural spectral variation"}
        
        return scores
    
    def detect(self, features: AudioFeatures, language: str) -> DetectionResult:
        """
        Run feature-based detection (supplementary to main classifier)
        
        Args:
            features: Extracted audio features
            language: Language of the audio
            
        Returns:
            DetectionResult with feature-based analysis
        """
        logger.info(f"Running feature analysis for language: {language}")
        
        analysis = self.analyze_features(features)
        
        # Calculate overall score
        feature_scores = {name: data["score"] for name, data in analysis.items()}
        avg_score = np.mean(list(feature_scores.values())) if feature_scores else 0.5
        
        # Classification based on average score
        if avg_score > 0.5:
            classification = "AI_GENERATED"
            confidence = avg_score
        else:
            classification = "HUMAN"
            confidence = 1.0 - avg_score
        
        # Generate explanation
        explanations = [data["note"] for data in analysis.values()]
        explanation = "; ".join(explanations[:2]) if explanations else "Feature analysis complete"
        
        return DetectionResult(
            classification=classification,
            confidence=round(confidence, 2),
            explanation=explanation,
            feature_scores=feature_scores
        )


# Singleton detector instance
voice_detector = VoiceDetector()
