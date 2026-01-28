"""Audio preprocessing utilities."""

import librosa
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from src.config import config


class AudioPreprocessor:
    """Preprocess audio files for embedding generation."""
    
    def __init__(
        self,
        sample_rate: int = None,
        n_mels: int = None,
        n_fft: int = None,
        hop_length: int = None
    ):
        """
        Initialize AudioPreprocessor.
        
        Args:
            sample_rate: Target sample rate
            n_mels: Number of mel bands
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate or config.SAMPLE_RATE
        self.n_mels = n_mels or config.N_MELS
        self.n_fft = n_fft or config.N_FFT
        self.hop_length = hop_length or config.HOP_LENGTH
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range.
        
        Args:
            audio: Audio array
            
        Returns:
            Normalized audio
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            return audio / max_val
        return audio
    
    def trim_silence(
        self, 
        audio: np.ndarray,
        top_db: int = 30
    ) -> np.ndarray:
        """
        Trim silence from audio.
        
        Args:
            audio: Audio array
            top_db: Threshold in dB below reference
            
        Returns:
            Trimmed audio
        """
        audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
        return audio_trimmed
    
    def pad_or_truncate(
        self,
        audio: np.ndarray,
        target_length: int
    ) -> np.ndarray:
        """
        Pad or truncate audio to target length.
        
        Args:
            audio: Audio array
            target_length: Target length in samples
            
        Returns:
            Padded or truncated audio
        """
        if len(audio) > target_length:
            return audio[:target_length]
        elif len(audio) < target_length:
            padding = target_length - len(audio)
            return np.pad(audio, (0, padding), mode='constant')
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract mel spectrogram from audio.
        
        Args:
            audio: Audio array
            
        Returns:
            Mel spectrogram
        """
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def extract_mfcc(
        self,
        audio: np.ndarray,
        n_mfcc: int = 13
    ) -> np.ndarray:
        """
        Extract MFCC features from audio.
        
        Args:
            audio: Audio array
            n_mfcc: Number of MFCC coefficients
            
        Returns:
            MFCC features
        """
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        return mfcc
    
    def preprocess(
        self,
        audio: np.ndarray,
        normalize: bool = True,
        trim: bool = True,
        target_duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply full preprocessing pipeline.
        
        Args:
            audio: Audio array
            normalize: Whether to normalize
            trim: Whether to trim silence
            target_duration: Target duration in seconds
            
        Returns:
            Preprocessed audio
        """
        if trim:
            audio = self.trim_silence(audio)
        
        if target_duration:
            target_length = int(target_duration * self.sample_rate)
            audio = self.pad_or_truncate(audio, target_length)
        
        if normalize:
            audio = self.normalize_audio(audio)
        
        return audio


if __name__ == "__main__":
    # Example usage
    preprocessor = AudioPreprocessor()
    
    # Create dummy audio
    dummy_audio = np.random.randn(44100 * 5)  # 5 seconds
    
    # Preprocess
    processed = preprocessor.preprocess(
        dummy_audio,
        normalize=True,
        trim=True,
        target_duration=10.0
    )
    
    print(f"Original length: {len(dummy_audio)} samples")
    print(f"Processed length: {len(processed)} samples")
    
    # Extract features
    mel_spec = preprocessor.extract_mel_spectrogram(processed)
    print(f"Mel spectrogram shape: {mel_spec.shape}")
