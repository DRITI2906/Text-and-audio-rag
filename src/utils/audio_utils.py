"""Audio utility functions."""

import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class AudioUtils:
    """Utility functions for audio processing and analysis."""
    
    @staticmethod
    def plot_waveform(audio: np.ndarray, sr: int, title: str = "Waveform", save_path: Path = None):
        """Plot audio waveform."""
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio, sr=sr)
        plt.title(title)
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    @staticmethod
    def plot_spectrogram(audio: np.ndarray, sr: int, title: str = "Spectrogram", save_path: Path = None):
        """Plot mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        plt.figure(figsize=(12, 6))
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    
    @staticmethod
    def get_audio_duration(file_path: Path) -> float:
        """Get duration of audio file in seconds."""
        return librosa.get_duration(path=file_path)
    
    @staticmethod
    def extract_tempo(audio: np.ndarray, sr: int) -> float:
        """Extract tempo (BPM) from audio."""
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
        return float(tempo)


if __name__ == "__main__":
    print("Audio utilities module")
