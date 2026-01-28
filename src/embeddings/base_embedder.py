"""Base embedder interface."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Union, List
from pathlib import Path


class BaseEmbedder(ABC):
    """Abstract base class for all embedders."""
    
    @abstractmethod
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text.
        
        Args:
            texts: Single text or list of texts
            
        Returns:
            Numpy array of embeddings
        """
        pass
    
    @abstractmethod
    def embed_audio(self, audio_paths: Union[str, Path, List[Union[str, Path]]]) -> np.ndarray:
        """
        Generate embeddings for audio files.
        
        Args:
            audio_paths: Single path or list of paths
            
        Returns:
            Numpy array of embeddings
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        pass
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: Path):
        """Save embeddings to file."""
        np.save(output_path, embeddings)
    
    def load_embeddings(self, input_path: Path) -> np.ndarray:
        """Load embeddings from file."""
        return np.load(input_path)
