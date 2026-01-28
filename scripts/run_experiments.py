"""Script to run experiments comparing different models."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.rag_pipeline import RAGPipeline
from src.embeddings.clap_embedder import CLAPEmbedder
from src.embeddings.audio_only_embedder import AudioOnlyEmbedder
from src.embeddings.text_embedder import TextEmbedder
from src.evaluation.evaluator import Evaluator
from src.config import config
from loguru import logger


def main():
    """Run experiments comparing CLAP and audio-only approaches."""
    logger.info("Starting model comparison experiments...")
    
    # Initialize evaluator
    evaluator = Evaluator()
    
    # Dictionary to store pipelines
    pipelines = {}
    
    # 1. CLAP-based approach
    logger.info("Initializing CLAP model...")
    try:
        clap_embedder = CLAPEmbedder()
        clap_pipeline = RAGPipeline(clap_embedder)
        
        # Load or build index
        if config.FAISS_INDEX_PATH.exists():
            clap_pipeline.load_index()
        else:
            logger.info("Building CLAP index...")
            clap_pipeline.build_index()
            clap_pipeline.save_index()
        
        pipelines["CLAP"] = clap_pipeline
        logger.info("CLAP model ready")
    except Exception as e:
        logger.error(f"Failed to initialize CLAP model: {e}")
    
    # 2. Audio-only approach (PANNs)
    logger.info("Initializing Audio-Only (PANNs) model...")
    try:
        audio_embedder = AudioOnlyEmbedder(model_type="panns")
        # Note: For audio-only, we need text-audio alignment
        # This is a simplified version - full implementation would need alignment
        logger.warning("Audio-only approach requires text-audio alignment (not fully implemented)")
        # pipelines["Audio-Only-PANNs"] = audio_pipeline
    except Exception as e:
        logger.error(f"Failed to initialize Audio-Only model: {e}")
    
    # Run comparison
    if pipelines:
        logger.info(f"Comparing {len(pipelines)} models...")
        results = evaluator.compare_models(pipelines, save_results=True)
        
        logger.info("Experiments complete!")
        logger.info(f"Results saved to {config.RETRIEVAL_RESULTS_DIR}")
        logger.info(f"Confusion matrices saved to {config.CONFUSION_MATRICES_DIR}")
    else:
        logger.error("No models available for comparison")


if __name__ == "__main__":
    main()
