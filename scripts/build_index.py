"""Script to build the vector index from audio samples."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.rag_pipeline import RAGPipeline
from src.embeddings.clap_embedder import CLAPEmbedder
from src.config import config
from loguru import logger


def main():
    """Build the index from audio samples."""
    logger.info("Starting index building process...")
    
    # Initialize embedder
    logger.info("Initializing CLAP embedder...")
    embedder = CLAPEmbedder()
    
    # Initialize pipeline
    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline(embedder)
    
    # Build index
    logger.info("Building index from audio samples...")
    try:
        pipeline.build_index()
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        return
    
    # Save index
    logger.info("Saving index...")
    pipeline.save_index()
    
    # Print statistics
    stats = pipeline.get_stats()
    logger.info("Index building complete!")
    logger.info(f"Total samples indexed: {stats['total_samples']}")
    logger.info(f"Categories: {stats['categories']}")
    for cat, count in stats['category_counts'].items():
        logger.info(f"  - {cat}: {count} samples")


if __name__ == "__main__":
    main()
