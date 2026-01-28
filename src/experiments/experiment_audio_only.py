"""Experiment with audio-only models."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.embeddings.audio_only_embedder import AudioOnlyEmbedder
from src.embeddings.text_embedder import TextEmbedder
from src.config import config
from loguru import logger


def run_audio_only_experiment(model_type: str = "panns"):
    """
    Run experiment with audio-only model.
    
    Args:
        model_type: Type of audio model (panns, vggish, openl3, wav2vec2)
    """
    logger.info("="*60)
    logger.info(f"Audio-Only Experiment - {model_type.upper()}")
    logger.info("="*60)
    
    # Initialize audio embedder
    logger.info(f"Initializing {model_type} audio embedder...")
    audio_embedder = AudioOnlyEmbedder(model_type=model_type)
    
    # Initialize text embedder for alignment
    logger.info("Initializing text embedder...")
    text_embedder = TextEmbedder()
    
    logger.info(f"\nAudio embedding dimension: {audio_embedder.get_embedding_dim()}")
    logger.info(f"Text embedding dimension: {text_embedder.get_embedding_dim()}")
    
    # Note: Full implementation would require:
    # 1. Cross-modal alignment (e.g., projection layers)
    # 2. Training/fine-tuning on paired text-audio data
    # 3. Building separate pipeline with alignment
    
    logger.warning("\nNote: Audio-only approach requires text-audio alignment.")
    logger.warning("This is a placeholder - full implementation needs:")
    logger.warning("  1. Projection layers to align text and audio embeddings")
    logger.warning("  2. Training on paired text-audio data")
    logger.warning("  3. Custom retrieval pipeline with alignment")
    
    logger.info("\nAudio-only experiment setup complete!")


if __name__ == "__main__":
    # Run with PANNs
    run_audio_only_experiment("panns")
