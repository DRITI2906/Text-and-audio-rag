"""Audio file loading utilities."""

import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import soundfile as sf

from src.config import config


class AudioLoader:
    """Load and manage audio files."""
    
    def __init__(self, sample_rate: int = None):
        """
        Initialize AudioLoader.
        
        Args:
            sample_rate: Target sample rate for loading audio
        """
        self.sample_rate = sample_rate or config.SAMPLE_RATE
    
    def load_audio(
        self, 
        file_path: Path, 
        duration: Optional[float] = None,
        offset: float = 0.0
    ) -> Tuple[np.ndarray, int]:
        """
        Load an audio file.
        
        Args:
            file_path: Path to audio file
            duration: Duration to load (None for full file)
            offset: Start time offset
            
        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio, sr = librosa.load(
            file_path,
            sr=self.sample_rate,
            mono=True,
            duration=duration,
            offset=offset
        )
        return audio, sr
    
    def load_category_samples(
        self, 
        category: str,
        max_samples: Optional[int] = None
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Load all audio samples from a category.
        
        Args:
            category: Category name (drums or keys)
            max_samples: Maximum number of samples to load
            
        Returns:
            List of (audio_array, filename) tuples
        """
        category_path = config.get_category_path(category)
        audio_files = sorted(category_path.glob("*.wav"))
        
        if max_samples:
            audio_files = audio_files[:max_samples]
        
        samples = []
        for audio_file in audio_files:
            try:
                audio, _ = self.load_audio(audio_file)
                samples.append((audio, audio_file.name))
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
        
        return samples
    
    def load_all_samples(self) -> Dict[str, List[Tuple[np.ndarray, str]]]:
        """
        Load all samples from all categories.
        
        Returns:
            Dictionary mapping category to list of (audio, filename) tuples
        """
        all_samples = {}
        for category in config.CATEGORIES:
            all_samples[category] = self.load_category_samples(
                category, 
                max_samples=config.SAMPLES_PER_CATEGORY
            )
        return all_samples
    
    def get_audio_info(self, file_path: Path) -> Dict:
        """
        Get information about an audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary with audio information
        """
        info = sf.info(file_path)
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
        }
    
    def load_metadata(self) -> pd.DataFrame:
        """
        Load metadata from CSV file.
        
        Returns:
            DataFrame with sample metadata
        """
        csv_path = config.get_samples_csv_path()
        if csv_path.exists():
            return pd.read_csv(csv_path)
        else:
            print(f"Metadata file not found: {csv_path}")
            return pd.DataFrame()


if __name__ == "__main__":
    # Example usage
    loader = AudioLoader()
    
    # Load all samples
    samples = loader.load_all_samples()
    for category, audio_list in samples.items():
        print(f"{category}: {len(audio_list)} samples loaded")
