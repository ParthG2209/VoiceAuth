"""
AI Voice Detection Model - Enhanced Version
Uses multiple feature-based heuristics calibrated for modern AI voice generators
"""

import logging
from typing import Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy import stats

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
    Enhanced Multi-feature voice authenticity detector
    
    Calibrated for modern AI voice generators (ElevenLabs, etc.)
    which produce very natural-sounding audio.
    
    Key indicators for AI voices:
    1. TOO perfect pitch consistency (humans have natural jitter)
    2. Unnatural formant transitions
    3. Missing breath sounds and micro-pauses
    4. Overly smooth energy envelope
    5. Lack of natural speech disfluencies
    """
    
    # MORE AGGRESSIVE thresholds - tuned for modern AI
    PITCH_STD_THRESHOLD = 45.0        # Raised - AI voices are smoother
    ENERGY_VAR_THRESHOLD = 0.001      # Raised - AI has consistent energy
    MFCC_DELTA_THRESHOLD = 0.20       # Raised threshold
    ZCR_CONSISTENCY_THRESHOLD = 0.35  # Raised threshold
    
    def __init__(self):
        """Initialize detector with model loading"""
        self.model_loaded = True
        logger.info("VoiceDetector initialized with ENHANCED feature-based analysis")
    
    def _analyze_pitch_consistency(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        Analyze pitch patterns for synthetic characteristics
        
        Modern AI voices have:
        - Very smooth pitch contours
        - Regular pitch patterns
        - Less natural vibrato
        """
        pitch_std = features.pitch_std
        pitch_mean = features.pitch_mean
        
        # Calculate coefficient of variation
        pitch_cv = pitch_std / (pitch_mean + 1e-6)
        
        # Modern AI voices are VERY smooth
        if pitch_std < 20.0:
            return 0.90, "Extremely smooth pitch (synthetic indicator)"
        elif pitch_std < self.PITCH_STD_THRESHOLD:
            return 0.75, "Unnaturally consistent pitch patterns"
        elif pitch_std < 60.0:
            return 0.55, "Slightly smooth pitch"
        elif pitch_std > 120.0:
            return 0.15, "Natural pitch variation with expression"
        else:
            return 0.35, "Normal pitch characteristics"
    
    def _analyze_energy_dynamics(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        Analyze energy envelope for synthetic patterns
        
        Human speech has:
        - Dynamic energy changes
        - Breath sounds (low energy spikes)
        - Natural pauses
        - Emphasis variations
        """
        energy_var = features.energy_variance
        rms = features.rms_energy
        
        if len(rms) < 10:
            return 0.5, "Insufficient energy data"
        
        # Calculate additional metrics
        energy_range = np.max(rms) - np.min(rms)
        energy_mean = np.mean(rms)
        energy_cv = np.std(rms) / (energy_mean + 1e-6)
        
        # Check for natural pauses (near-zero energy regions)
        silence_threshold = energy_mean * 0.1
        has_pauses = np.any(rms < silence_threshold)
        
        # Check for sudden energy changes (breath, emphasis)
        energy_diff = np.abs(np.diff(rms))
        sudden_changes = np.sum(energy_diff > energy_mean * 0.5)
        
        if energy_var < 0.0001:
            return 0.88, "Extremely flat energy envelope (synthetic)"
        elif energy_var < self.ENERGY_VAR_THRESHOLD:
            return 0.72, "Unnaturally smooth energy dynamics"
        elif not has_pauses and energy_cv < 0.4:
            return 0.65, "Missing natural pauses and breath sounds"
        elif energy_var > 0.01 and sudden_changes > 5:
            return 0.20, "Natural energy dynamics with emphasis"
        else:
            return 0.45, "Moderate energy variation"
    
    def _analyze_mfcc_patterns(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        Analyze MFCC coefficients for synthesis artifacts
        
        AI voices show:
        - Over-smoothed MFCC trajectories
        - Repetitive patterns
        - Lack of coarticulation complexity
        """
        mfcc = features.mfcc
        
        if mfcc.shape[1] < 10:
            return 0.5, "Insufficient MFCC data"
        
        # Calculate MFCC deltas (temporal changes)
        mfcc_delta = np.diff(mfcc, axis=1)
        delta_mean = np.mean(np.abs(mfcc_delta))
        delta_std = np.std(mfcc_delta)
        
        # Calculate second-order deltas
        mfcc_delta2 = np.diff(mfcc_delta, axis=1)
        delta2_mean = np.mean(np.abs(mfcc_delta2))
        
        # Check for autocorrelation (periodic patterns)
        autocorr_scores = []
        for i in range(min(5, mfcc.shape[0])):
            autocorr = np.correlate(mfcc[i], mfcc[i], mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if len(autocorr) > 20:
                autocorr = autocorr / (autocorr[0] + 1e-6)
                peak = np.max(autocorr[10:min(100, len(autocorr))])
                autocorr_scores.append(peak)
        
        periodic_score = np.mean(autocorr_scores) if autocorr_scores else 0
        
        # Scoring
        if delta_mean < 0.10:
            return 0.85, "Extremely smooth MFCC transitions (synthetic)"
        elif delta_mean < self.MFCC_DELTA_THRESHOLD:
            return 0.70, "Over-smoothed spectral dynamics"
        elif periodic_score > 0.6:
            return 0.75, "Periodic vocoder artifacts detected"
        elif delta_mean > 0.35 and delta2_mean > 0.15:
            return 0.18, "Natural spectral complexity"
        else:
            return 0.42, "Normal spectral characteristics"
    
    def _analyze_spectral_features(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        Analyze spectral centroid and bandwidth patterns
        
        AI voices often have:
        - Overly consistent formants
        - Limited spectral variation
        - Clean high frequencies (no noise)
        """
        centroid = features.spectral_centroid
        bandwidth = features.spectral_bandwidth
        rolloff = features.spectral_rolloff
        
        if len(centroid) < 10:
            return 0.5, "Insufficient spectral data"
        
        # Spectral consistency metrics
        centroid_cv = np.std(centroid) / (np.mean(centroid) + 1e-6)
        bandwidth_cv = np.std(bandwidth) / (np.mean(bandwidth) + 1e-6)
        rolloff_cv = np.std(rolloff) / (np.mean(rolloff) + 1e-6)
        
        # Check for spectral flatness (synthetic voices can be TOO clean)
        spectral_consistency = (centroid_cv + bandwidth_cv + rolloff_cv) / 3
        
        if spectral_consistency < 0.12:
            return 0.85, "Extremely uniform spectral characteristics (synthetic)"
        elif spectral_consistency < 0.20:
            return 0.70, "Unnaturally consistent formants"
        elif spectral_consistency > 0.45:
            return 0.20, "Natural spectral variation with noise"
        else:
            return 0.45, "Normal spectral properties"
    
    def _analyze_zero_crossing(self, features: AudioFeatures) -> Tuple[float, str]:
        """Analyze zero-crossing rate patterns"""
        zcr = features.zero_crossing_rate
        
        if len(zcr) < 10:
            return 0.5, "Insufficient ZCR data"
        
        zcr_cv = np.std(zcr) / (np.mean(zcr) + 1e-6)
        
        # Check for unnatural ZCR distribution
        zcr_skew = abs(stats.skew(zcr))
        
        if zcr_cv < 0.25:
            return 0.72, "Unnaturally consistent articulation patterns"
        elif zcr_cv < self.ZCR_CONSISTENCY_THRESHOLD:
            return 0.60, "Slightly smooth articulation"
        elif zcr_cv > 0.8 and zcr_skew > 0.5:
            return 0.22, "Natural voiced/unvoiced transitions"
        else:
            return 0.42, "Normal articulation patterns"
    
    def _analyze_temporal_coherence(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        Analyze temporal micro-variations (jitter/shimmer-like)
        
        Human speech contains:
        - Micro-tremors
        - Slight pitch instabilities
        - Natural amplitude variations
        """
        waveform = features.waveform
        sr = features.sample_rate
        
        frame_length = int(0.020 * sr)
        hop_length = int(0.010 * sr)
        
        num_frames = (len(waveform) - frame_length) // hop_length + 1
        if num_frames < 20:
            return 0.5, "Audio too short for temporal analysis"
        
        frame_energies = []
        for i in range(num_frames):
            start = i * hop_length
            frame = waveform[start:start + frame_length]
            frame_energies.append(np.sqrt(np.mean(frame ** 2)))
        
        frame_energies = np.array(frame_energies)
        
        # Jitter-like metric
        frame_diffs = np.abs(np.diff(frame_energies))
        jitter_ratio = np.mean(frame_diffs) / (np.mean(frame_energies) + 1e-6)
        
        # Shimmer-like metric (amplitude variation)
        shimmer = np.std(frame_energies) / (np.mean(frame_energies) + 1e-6)
        
        # Check for micro-variations
        micro_var = np.std(frame_diffs)
        
        if jitter_ratio < 0.08 and shimmer < 0.3:
            return 0.82, "Missing natural micro-tremors (synthetic)"
        elif jitter_ratio < 0.12:
            return 0.68, "Unnaturally smooth temporal envelope"
        elif jitter_ratio > 0.35 and shimmer > 0.5:
            return 0.18, "Natural micro-variations present"
        else:
            return 0.42, "Moderate temporal coherence"
    
    def _analyze_breath_and_pauses(self, features: AudioFeatures) -> Tuple[float, str]:
        """
        NEW: Analyze for breath sounds and natural pauses
        
        Human speech has:
        - Audible breaths between phrases
        - Variable pause lengths
        - Natural rhythm patterns
        """
        rms = features.rms_energy
        
        if len(rms) < 20:
            return 0.5, "Insufficient data for breath analysis"
        
        mean_energy = np.mean(rms)
        
        # Find potential pause regions
        pause_threshold = mean_energy * 0.15
        pause_mask = rms < pause_threshold
        
        # Count distinct pause regions
        pause_changes = np.diff(pause_mask.astype(int))
        num_pauses = np.sum(pause_changes == 1)
        
        # Calculate pause ratio
        pause_ratio = np.sum(pause_mask) / len(rms)
        
        # Analyze pause distribution
        if num_pauses == 0 and pause_ratio < 0.05:
            return 0.78, "No natural pauses or breaths detected"
        elif num_pauses < 2 and pause_ratio < 0.10:
            return 0.62, "Limited natural pausing"
        elif num_pauses > 3 and 0.1 < pause_ratio < 0.3:
            return 0.22, "Natural breath patterns detected"
        else:
            return 0.45, "Moderate pause patterns"
    
    def detect(self, features: AudioFeatures, language: str) -> DetectionResult:
        """
        Run full detection pipeline with enhanced analysis
        """
        logger.info(f"Running ENHANCED voice detection for language: {language}")
        
        # Run all analysis methods (including new breath analysis)
        analyses = {
            "pitch": self._analyze_pitch_consistency(features),
            "energy": self._analyze_energy_dynamics(features),
            "mfcc": self._analyze_mfcc_patterns(features),
            "spectral": self._analyze_spectral_features(features),
            "zcr": self._analyze_zero_crossing(features),
            "temporal": self._analyze_temporal_coherence(features),
            "breath": self._analyze_breath_and_pauses(features)
        }
        
        feature_scores = {name: score for name, (score, _) in analyses.items()}
        
        # Updated weights - emphasize features that catch AI
        weights = {
            "pitch": 0.15,
            "energy": 0.15,
            "mfcc": 0.20,
            "spectral": 0.15,
            "zcr": 0.08,
            "temporal": 0.15,
            "breath": 0.12
        }
        
        # Calculate weighted average
        ai_probability = sum(
            feature_scores[name] * weights[name]
            for name in weights
        )
        
        # BIAS TOWARD AI DETECTION for ambiguous cases
        # Modern AI is very good, so we're more suspicious
        if ai_probability > 0.40:
            ai_probability = min(ai_probability * 1.15, 0.95)
        
        # Log all scores for debugging
        logger.info(f"Feature scores: {feature_scores}")
        logger.info(f"Raw AI probability: {ai_probability:.3f}")
        
        # Generate explanation
        explanations = []
        sorted_analyses = sorted(
            analyses.items(),
            key=lambda x: x[1][0],  # Sort by score (highest = most AI-like)
            reverse=True
        )
        
        for name, (score, explanation) in sorted_analyses[:3]:
            if score > 0.55:
                explanations.append(explanation)
        
        if not explanations:
            for name, (score, explanation) in sorted_analyses[-2:]:
                explanations.append(explanation)
        
        combined_explanation = "; ".join(explanations[:2]) if explanations else "Analysis inconclusive"
        
        # Final classification - LOWER threshold to catch more AI
        if ai_probability > 0.45:  # Changed from 0.5
            classification = "AI_GENERATED"
            confidence = min(ai_probability * 1.1, 0.99)
        else:
            classification = "HUMAN"
            confidence = min((1 - ai_probability) * 1.1, 0.99)
        
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
