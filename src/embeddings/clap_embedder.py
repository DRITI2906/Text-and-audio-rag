"""CLAP-based embedding model for text and audio."""

import torch
import numpy as np
from pathlib import Path
from typing import Union, List
from loguru import logger

try:
    import laion_clap
except ImportError:
    logger.warning("laion_clap not installed. Install with: pip install laion-clap")

from config.settings import settings


class CLAPEmbedder:
    """CLAP model for generating text and audio embeddings in a shared space."""
    
    def __init__(
        self, 
        model_name: str = None,
        device: str = None,
        enable_fusion: bool = True
    ):
        """
        Initialize CLAP embedder.
        
        Args:
            model_name: CLAP model name (default from config)
            device: Device to run model on ('cuda' or 'cpu')
            enable_fusion: Whether to enable fusion in CLAP model
        """
        self.model_name = model_name or settings.CLAP_MODEL_NAME
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.enable_fusion = enable_fusion
        
        logger.info(f"Initializing CLAP model: {self.model_name}")
        logger.info(f"Using device: {self.device}")
        
        # Load CLAP model
        self.model = laion_clap.CLAP_Module(enable_fusion=enable_fusion)
        self.model.load_ckpt(model_name=self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info("CLAP model loaded successfully")
    
    def embed_text(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for text queries.
        
        Args:
            texts: Single text string or list of text strings
            
        Returns:
            Numpy array of embeddings (shape: [num_texts, embedding_dim])
        """
        if isinstance(texts, str):
            texts = [texts]
        
        with torch.no_grad():
            text_embeddings = self.model.get_text_embedding(texts, use_tensor=False)
        
        return text_embeddings
    
    def embed_audio(self, audio_paths: Union[str, Path, List[Union[str, Path]]]) -> np.ndarray:
        """
        Generate embeddings for audio files.
        
        Args:
            audio_paths: Single audio path or list of audio paths
            
        Returns:
            Numpy array of embeddings (shape: [num_audios, embedding_dim])
        """
        if isinstance(audio_paths, (str, Path)):
            audio_paths = [str(audio_paths)]
        else:
            audio_paths = [str(p) for p in audio_paths]
        
        with torch.no_grad():
            audio_embeddings = self.model.get_audio_embedding_from_filelist(
                x=audio_paths, 
                use_tensor=False
            )
        
        return audio_embeddings
    
    def compute_similarity(
        self, 
        text_embedding: np.ndarray, 
        audio_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarity between text and audio embeddings.
        
        Args:
            text_embedding: Text embedding (shape: [1, embedding_dim])
            audio_embeddings: Audio embeddings (shape: [num_audios, embedding_dim])
            
        Returns:
            Similarity scores (shape: [num_audios])
        """
        # Normalize embeddings
        text_norm = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)
        audio_norm = audio_embeddings / np.linalg.norm(audio_embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarity
        similarity = np.dot(audio_norm, text_norm.T).squeeze()
        
        return similarity
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        return 512  # CLAP typically uses 512-dim embeddings
    
    def save_embeddings(self, embeddings: np.ndarray, output_path: Path):
        """Save embeddings to file."""
        np.save(output_path, embeddings)
        logger.info(f"Saved embeddings to {output_path}")
    
    def load_embeddings(self, input_path: Path) -> np.ndarray:
        """Load embeddings from file."""
        embeddings = np.load(input_path)
        logger.info(f"Loaded embeddings from {input_path}")
        return embeddings


if __name__ == "__main__":
    # Example usage
    embedder = CLAPEmbedder()
    
    # Test text embedding
    text_queries = ["drum samples", "piano keys in C major"]
    text_emb = embedder.embed_text(text_queries)
    print(f"Text embeddings shape: {text_emb.shape}")
    
    # Test audio embedding (requires audio files)
    # audio_paths = ["path/to/audio1.wav", "path/to/audio2.wav"]
    # audio_emb = embedder.embed_audio(audio_paths)
    # print(f"Audio embeddings shape: {audio_emb.shape}")
