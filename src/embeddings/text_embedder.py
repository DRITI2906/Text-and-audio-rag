"""Text encoder for generating text embeddings (used with audio-only approach)."""

import torch
import numpy as np
from typing import Union, List
from loguru import logger
from sentence_transformers import SentenceTransformer

from config.settings import settings


class TextEncoder:
    """Text encoder using sentence transformers for text embedding generation."""
    
    def __init__(
        self, 
        model_name: str = None,
        device: str = None
    ):
        """
        Initialize text encoder.
        
        Args:
            model_name: Sentence transformer model name
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.model_name = model_name or settings.TEXT_MODEL_NAME
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing text encoder: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(self.model_name, device=self.device)
        
        logger.info("Text encoder loaded successfully")
    
    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """
        Encode texts into embeddings.
        
        Args:
            texts: Single text string or list of text strings
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
            show_progress_bar=False
        )
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        return self.model.get_sentence_embedding_dimension()


if __name__ == "__main__":
    # Example usage
    encoder = TextEncoder()
    
    texts = ["drum samples", "piano keys in C major", "upbeat electronic loops"]
    embeddings = encoder.encode(texts)
    
    print(f"Text embeddings shape: {embeddings.shape}")
    print(f"Embedding dimension: {encoder.get_embedding_dim()}")
