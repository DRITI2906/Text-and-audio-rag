"""Script to download pretrained models."""

import os
from pathlib import Path
from loguru import logger

from config.settings import settings


def download_clap_model():
    """Download CLAP model."""
    logger.info("Downloading CLAP model...")
    
    try:
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=True)
        model.load_ckpt(model_name=settings.CLAP_MODEL_NAME)
        logger.info("CLAP model downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download CLAP model: {e}")


def download_text_model():
    """Download sentence transformer model."""
    logger.info("Downloading text encoder model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.TEXT_MODEL_NAME)
        logger.info("Text encoder model downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download text model: {e}")


def download_audio_models():
    """Download audio-only models."""
    logger.info("Downloading audio models...")
    
    # PANNs
    try:
        from panns_inference import AudioTagging
        model = AudioTagging(checkpoint_path=None)
        logger.info("PANNs model downloaded successfully")
    except Exception as e:
        logger.warning(f"Failed to download PANNs model: {e}")
    
    # Wav2Vec2
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        logger.info("Wav2Vec2 model downloaded successfully")
    except Exception as e:
        logger.warning(f"Failed to download Wav2Vec2 model: {e}")


def main():
    """Download all pretrained models."""
    logger.info("Starting model downloads...")
    
    # Create models directory
    settings.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Download models
    download_clap_model()
    download_text_model()
    download_audio_models()
    
    logger.info("All models downloaded!")


if __name__ == "__main__":
    main()
