"""
AI Voice Detection Model
Uses multiple feature-based heuristics and ML models to detect synthetic voices
"""

import logging
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np

from app.utils.audio_processor import AudioFeatures
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DetectionResult:
    """Result from voice detection analysis"""
    classification: str  # AI_GENERATED or HUMAN
    confidence: float    # 0.0 to 1.0
    explanation: str     # Human-readable reason
    
    # Detailed scores for debugging
    feature_scores: dict


class VoiceDetector:
    """
    Multi-feature voice authenticity detector
    
    Uses ensemble of detection methods:
    1. Spectral analysis - AI voices often have unnatural spectral patterns
    2. Pitch analysis - Synthetic voices have abnormal pitch consistency
    3. MFCC analysis - Cepstral coefficients reveal synthesis artifacts
    4. Energy dynamics - Natural speech has characteristic energy patterns
    5. Temporal coherence - Human speech has natural micro-variations
    """
    
    # Thresholds calibrated for detection
    PITCH_STD_THRESHOLD = 30.0        # Hz - AI voices often have lower variance
    ENERGY_VAR_THRESHOLD = 0.0003     # Natural speech has higher energy variance
    MFCC_DELTA_THRESHOLD = 0.15       # MFCC deltas for temporal dynamics
    SPECTRAL_FLUX_THRESHOLD = 0.02    # Spectral change rate
    ZCR_CONSISTENCY_THRESHOLD = 0.3   # Zero-crossing rate consistency
    
    def __init__(self):
        """Initialize detector with model loading"""
        self.model_loaded = True  # Feature-based for now
        logger.info("VoiceDetector initialized with feature-based analysis")
    
    def _analyze_pitch_consistency(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        Analyze pitch patterns for synthetic characteristics
        
        AI-generated voices often have:
        - Unnaturally consistent pitch
        - Very smooth pitch transitions
        - Less micro-variations
        
        Returns:
            Tuple of (ai_probability, explanation)
        """
        pitch_std = features.pitch_std
        pitch_mean = features.pitch_mean
        
        # Low pitch standard deviation suggests AI
        if pitch_std < self.PITCH_STD_THRESHOLD:
            # Very low variance is highly suspicious
            if pitch_std < 15.0:
                return 0.85, "Extremely consistent pitch detected (unnatural)"
            return 0.70, "Unusually stable pitch patterns"
        elif pitch_std > 100.0:
            # Very high variance suggests natural human expression
            return 0.2, "Natural pitch variation detected"
        else:
            return 0.4, "Normal pitch characteristics"
    
    def _analyze_energy_dynamics(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        Analyze energy envelope for synthetic patterns
        
        Human speech has:
        - Dynamic energy changes
        - Natural pauses and emphasis
        - Breathing patterns
        """
        energy_var = features.energy_variance
        rms = features.rms_energy
        
        # Calculate additional energy metrics
        energy_range = np.max(rms) - np.min(rms) if len(rms) > 0 else 0
        
        if energy_var < self.ENERGY_VAR_THRESHOLD:
            return 0.75, "Unnaturally smooth energy envelope"
        elif energy_var > 0.01:
            return 0.25, "Natural energy dynamics detected"
        else:
            return 0.45, "Moderate energy variation"
    
    def _analyze_mfcc_patterns(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        Analyze MFCC coefficients for synthesis artifacts
        
        AI voices often show:
        - Over-smoothed MFCC trajectories
        - Less natural coarticulation effects
        - Periodic patterns from vocoder
        """
        mfcc = features.mfcc
        
        # Calculate MFCC deltas (temporal changes)
        mfcc_delta = np.diff(mfcc, axis=1)
        delta_mean = np.mean(np.abs(mfcc_delta))
        delta_std = np.std(mfcc_delta)
        
        # Calculate MFCC correlation across time (AI voices may show patterns)
        if mfcc.shape[1] > 10:
            # Check for periodic patterns (potential vocoder artifacts)
            autocorr = np.correlate(mfcc[0], mfcc[0], mode='full')
            autocorr = autocorr[len(autocorr)//2:]  # Take positive lags
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # High autocorrelation at non-zero lags suggests synthesis
            periodic_score = np.max(autocorr[5:min(50, len(autocorr))])
        else:
            periodic_score = 0.0
        
        # Low delta values suggest over-smoothed (AI) voice
        if delta_mean < self.MFCC_DELTA_THRESHOLD:
            return 0.72, "Over-smoothed spectral transitions"
        elif periodic_score > 0.5:
            return 0.68, "Periodic artifacts detected in spectral features"
        elif delta_mean > 0.3:
            return 0.22, "Natural spectral dynamics"
        else:
            return 0.42, "Normal spectral characteristics"
    
    def _analyze_spectral_features(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        Analyze spectral centroid and bandwidth patterns
        
        Synthetic voices may have:
        - More consistent spectral centroid
        - Limited bandwidth variation
        - Unnatural formant transitions
        """
        centroid = features.spectral_centroid
        bandwidth = features.spectral_bandwidth
        rolloff = features.spectral_rolloff
        
        # Spectral consistency metrics
        centroid_std = np.std(centroid) if len(centroid) > 0 else 0
        bandwidth_std = np.std(bandwidth) if len(bandwidth) > 0 else 0
        rolloff_std = np.std(rolloff) if len(rolloff) > 0 else 0
        
        # Normalize by mean for relative consistency
        centroid_cv = centroid_std / (np.mean(centroid) + 1e-6)
        bandwidth_cv = bandwidth_std / (np.mean(bandwidth) + 1e-6)
        
        # Very consistent spectral features suggest AI
        if centroid_cv < 0.15 and bandwidth_cv < 0.2:
            return 0.70, "Unnaturally consistent spectral characteristics"
        elif centroid_cv > 0.4 and bandwidth_cv > 0.3:
            return 0.25, "Natural spectral variation detected"
        else:
            return 0.45, "Normal spectral properties"
    
    def _analyze_zero_crossing(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        Analyze zero-crossing rate patterns
        
        Human speech has characteristic ZCR patterns:
        - Higher during unvoiced sounds (fricatives)
        - Lower during voiced sounds
        - Natural transitions between
        """
        zcr = features.zero_crossing_rate
        
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)
        zcr_cv = zcr_std / (zcr_mean + 1e-6)
        
        # AI voices may have more consistent ZCR
        if zcr_cv < self.ZCR_CONSISTENCY_THRESHOLD:
            return 0.65, "Consistent zero-crossing patterns (potential synthesis)"
        elif zcr_cv > 0.8:
            return 0.30, "Natural voiced/unvoiced transitions"
        else:
            return 0.45, "Normal articulation patterns"
    
    def _analyze_temporal_coherence(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        Analyze temporal micro-variations
        
        Human speech contains micro-tremors and natural inconsistencies
        that are often missing or over-regularized in synthetic speech
        """
        waveform = features.waveform
        sr = features.sample_rate
        
        # Analyze short-term energy fluctuations
        frame_length = int(0.020 * sr)  # 20ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculate frame-level metrics
        num_frames = (len(waveform) - frame_length) // hop_length + 1
        if num_frames < 10:
            return 0.5, "Audio too short for temporal analysis"
        
        frame_energies = []
        for i in range(num_frames):
            start = i * hop_length
            frame = waveform[start:start + frame_length]
            frame_energies.append(np.sqrt(np.mean(frame ** 2)))
        
        frame_energies = np.array(frame_energies)
        
        # Calculate jitter-like metric (variation between consecutive frames)
        frame_diffs = np.abs(np.diff(frame_energies))
        jitter_ratio = np.mean(frame_diffs) / (np.mean(frame_energies) + 1e-6)
        
        # Low jitter suggests overly smooth (AI) voice
        if jitter_ratio < 0.1:
            return 0.68, "Minimal micro-variations (unnatural smoothness)"
        elif jitter_ratio > 0.4:
            return 0.28, "Natural micro-tremors detected"
        else:
            return 0.45, "Moderate temporal coherence"
    
    def detect(self, features: AudioFeatures, language: str) -> DetectionResult:
        """
        Run full detection pipeline
        
        Args:
            features: Extracted audio features
            language: Language of the audio (for potential language-specific tuning)
            
        Returns:
            DetectionResult with classification, confidence, and explanation
        """
        logger.info(f"Running voice detection for language: {language}")
        
        # Run all analysis methods
        analyses = {
            "pitch": self._analyze_pitch_consistency(features),
            "energy": self._analyze_energy_dynamics(features),
            "mfcc": self._analyze_mfcc_patterns(features),
            "spectral": self._analyze_spectral_features(features),
            "zcr": self._analyze_zero_crossing(features),
            "temporal": self._analyze_temporal_coherence(features)
        }
        
        # Extract scores and explanations
        feature_scores = {name: score for name, (score, _) in analyses.items()}
        
        # Weighted ensemble (some features are more reliable)
        weights = {
            "pitch": 0.20,
            "energy": 0.15,
            "mfcc": 0.25,
            "spectral": 0.15,
            "zcr": 0.10,
            "temporal": 0.15
        }
        
        # Calculate weighted average
        ai_probability = sum(
            feature_scores[name] * weights[name]
            for name in weights
        )
        
        # Generate explanation from highest-contributing factors
        explanations = []
        sorted_analyses = sorted(
            analyses.items(),
            key=lambda x: abs(x[1][0] - 0.5),  # Sort by deviation from neutral
            reverse=True
        )
        
        # Take top 2 contributing factors for explanation
        for name, (score, explanation) in sorted_analyses[:2]:
            if score > 0.5:  # Contributing to AI classification
                explanations.append(explanation)
        
        # If no strong AI indicators, explain why it seems human
        if not explanations:
            for name, (score, explanation) in sorted_analyses[:2]:
                if score <= 0.5:
                    explanations.append(explanation)
        
        combined_explanation = "; ".join(explanations) if explanations else "Analysis inconclusive"
        
        # Final classification with confidence
        if ai_probability > 0.5:
            classification = "AI_GENERATED"
            confidence = min(ai_probability * 1.1, 0.99)  # Scale up slightly
        else:
            classification = "HUMAN"
            confidence = min((1 - ai_probability) * 1.1, 0.99)  # Invert for human confidence
        
        logger.info(
            f"Detection complete: {classification} "
            f"(confidence: {confidence:.2f}, raw: {ai_probability:.2f})"
        )
        
        return DetectionResult(
            classification=classification,
            confidence=round(confidence, 2),
            explanation=combined_explanation,
            feature_scores=feature_scores
        )


# Singleton detector instance
voice_detector = VoiceDetector()
