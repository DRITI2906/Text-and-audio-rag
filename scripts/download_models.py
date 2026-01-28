"""Download pretrained models."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import config
from loguru import logger


def download_clap_model():
    """Download CLAP model."""
    logger.info("Downloading CLAP model...")
    
    try:
        import laion_clap
        model = laion_clap.CLAP_Module(enable_fusion=True)
        model.load_ckpt(model_name=config.CLAP_MODEL_NAME)
        logger.info("✓ CLAP model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download CLAP model: {e}")
        return False


def download_text_model():
    """Download sentence transformer model."""
    logger.info("Downloading text encoder model...")
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(config.TEXT_MODEL_NAME)
        logger.info("✓ Text encoder model downloaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to download text model: {e}")
        return False


def download_audio_models():
    """Download audio-only models."""
    logger.info("Downloading audio models...")
    success_count = 0
    
    # PANNs
    try:
        from panns_inference import AudioTagging
        model = AudioTagging(checkpoint_path=None)
        logger.info("✓ PANNs model downloaded successfully")
        success_count += 1
    except Exception as e:
        logger.warning(f"✗ Failed to download PANNs model: {e}")
    
    # Wav2Vec2
    try:
        from transformers import Wav2Vec2Model, Wav2Vec2Processor
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        logger.info("✓ Wav2Vec2 model downloaded successfully")
        success_count += 1
    except Exception as e:
        logger.warning(f"✗ Failed to download Wav2Vec2 model: {e}")
    
    return success_count > 0


def main():
    """Download all pretrained models."""
    logger.info("="*60)
    logger.info("Model Download Script")
    logger.info("="*60)
    
    # Create models directory
    config.MODELS_PATH.mkdir(parents=True, exist_ok=True)
    
    # Download models
    results = {
        "CLAP": download_clap_model(),
        "Text Encoder": download_text_model(),
        "Audio Models": download_audio_models()
    }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("Download Summary")
    logger.info("="*60)
    for model_name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        logger.info(f"{model_name}: {status}")
    
    if all(results.values()):
        logger.info("\n✓ All models downloaded successfully!")
    else:
        logger.warning("\n⚠ Some models failed to download")


if __name__ == "__main__":
    main()
