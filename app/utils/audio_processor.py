"""
Audio Processing Utilities
Handles Base64 decoding, format validation, and feature extraction
"""

import base64
import io
import tempfile
import os
from typing import Tuple, Optional
from dataclasses import dataclass

import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment

from app.config import settings


@dataclass
class AudioFeatures:
    """Extracted audio features for ML analysis"""
    
    # Basic properties
    sample_rate: int
    duration: float
    num_channels: int
    
    # Spectral features
    mfcc: np.ndarray                    # Mel-frequency cepstral coefficients
    spectral_centroid: np.ndarray       # Center of mass of spectrum
    spectral_rolloff: np.ndarray        # Frequency below which 85% of energy is contained
    spectral_bandwidth: np.ndarray      # Width of the spectral band
    zero_crossing_rate: np.ndarray      # Rate of sign changes
    
    # Pitch and rhythm
    pitch_mean: float
    pitch_std: float
    tempo: float
    
    # Energy features
    rms_energy: np.ndarray
    energy_variance: float
    
    # Raw waveform for model input
    waveform: np.ndarray


class AudioProcessor:
    """Process and extract features from audio data"""
    
    def __init__(self, target_sr: int = 16000):
        """
        Initialize audio processor
        
        Args:
            target_sr: Target sample rate for processing (16kHz default for speech)
        """
        self.target_sr = target_sr
        self.max_duration = settings.max_audio_duration_seconds
        self.max_size_bytes = settings.max_audio_size_mb * 1024 * 1024
    
    def decode_base64(self, audio_base64: str) -> bytes:
        """
        Decode Base64 audio string to bytes
        
        Args:
            audio_base64: Base64-encoded audio string
            
        Returns:
            Raw audio bytes
            
        Raises:
            ValueError: If decoding fails
        """
        try:
            # Handle data URL format
            if audio_base64.startswith("data:"):
                audio_base64 = audio_base64.split(",", 1)[-1]
            
            audio_bytes = base64.b64decode(audio_base64)
            
            # Check size limit
            if len(audio_bytes) > self.max_size_bytes:
                raise ValueError(
                    f"Audio file too large. Maximum size: {settings.max_audio_size_mb}MB"
                )
            
            return audio_bytes
        
        except base64.binascii.Error as e:
            raise ValueError(f"Invalid Base64 encoding: {str(e)}")
    
    def load_audio(self, audio_bytes: bytes) -> Tuple[np.ndarray, int]:
        """
        Load audio from bytes into numpy array
        
        Args:
            audio_bytes: Raw audio bytes (MP3 format expected)
            
        Returns:
            Tuple of (waveform array, sample rate)
        """
        try:
            # Create temp file for pydub to read
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tmp_file.write(audio_bytes)
                tmp_path = tmp_file.name
            
            try:
                # Load with pydub (handles MP3)
                audio_segment = AudioSegment.from_mp3(tmp_path)
                
                # Convert to mono if stereo
                if audio_segment.channels > 1:
                    audio_segment = audio_segment.set_channels(1)
                
                # Export to WAV for librosa
                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                
                # Load with librosa for analysis
                waveform, sr = librosa.load(wav_buffer, sr=self.target_sr, mono=True)
                
                # Check duration
                duration = len(waveform) / sr
                if duration > self.max_duration:
                    raise ValueError(
                        f"Audio too long ({duration:.1f}s). "
                        f"Maximum duration: {self.max_duration}s"
                    )
                
                return waveform, sr
            
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
        
        except Exception as e:
            if "decoding" in str(e).lower() or "format" in str(e).lower():
                raise ValueError(f"Invalid MP3 audio format: {str(e)}")
            raise ValueError(f"Error loading audio: {str(e)}")
    
    def extract_features(self, waveform: np.ndarray, sr: int) -> AudioFeatures:
        """
        Extract comprehensive audio features for AI detection
        
        Args:
            waveform: Audio waveform array
            sr: Sample rate
            
        Returns:
            AudioFeatures dataclass with all extracted features
        """
        # Duration and basic info
        duration = len(waveform) / sr
        
        # MFCC (Mel-frequency cepstral coefficients) - key for voice analysis
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=waveform, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(waveform)[0]
        
        # Pitch analysis
        pitches, magnitudes = librosa.piptrack(y=waveform, sr=sr)
        pitch_values = pitches[magnitudes > np.median(magnitudes)]
        pitch_values = pitch_values[pitch_values > 0]  # Remove zeros
        
        pitch_mean = float(np.mean(pitch_values)) if len(pitch_values) > 0 else 0.0
        pitch_std = float(np.std(pitch_values)) if len(pitch_values) > 0 else 0.0
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=waveform, sr=sr)
        tempo = float(tempo) if isinstance(tempo, (int, float, np.number)) else float(tempo[0]) if len(tempo) > 0 else 0.0
        
        # Energy features
        rms_energy = librosa.feature.rms(y=waveform)[0]
        energy_variance = float(np.var(rms_energy))
        
        return AudioFeatures(
            sample_rate=sr,
            duration=duration,
            num_channels=1,
            mfcc=mfcc,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            spectral_bandwidth=spectral_bandwidth,
            zero_crossing_rate=zero_crossing_rate,
            pitch_mean=pitch_mean,
            pitch_std=pitch_std,
            tempo=tempo,
            rms_energy=rms_energy,
            energy_variance=energy_variance,
            waveform=waveform
        )
    
    def process_base64_audio(self, audio_base64: str) -> AudioFeatures:
        """
        Complete pipeline: Base64 → Audio Features
        
        Args:
            audio_base64: Base64-encoded MP3 audio
            
        Returns:
            Extracted audio features
        """
        audio_bytes = self.decode_base64(audio_base64)
        waveform, sr = self.load_audio(audio_bytes)
        features = self.extract_features(waveform, sr)
        return features


# Singleton processor instance
audio_processor = AudioProcessor()
