"""Experiment with CLAP model."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval.rag_pipeline import RAGPipeline
from src.embeddings.clap_embedder import CLAPEmbedder
from src.evaluation.evaluator import Evaluator
from src.config import config
from loguru import logger


def run_clap_experiment():
    """Run experiment with CLAP model."""
    logger.info("="*60)
    logger.info("CLAP Model Experiment")
    logger.info("="*60)
    
    # Initialize CLAP embedder
    logger.info("Initializing CLAP embedder...")
    embedder = CLAPEmbedder()
    
    # Initialize pipeline
    logger.info("Initializing RAG pipeline...")
    pipeline = RAGPipeline(embedder)
    
    # Build or load index
    if config.FAISS_INDEX_PATH.exists():
        logger.info("Loading existing index...")
        pipeline.load_index()
    else:
        logger.info("Building new index...")
        pipeline.build_index()
        pipeline.save_index()
    
    # Get statistics
    stats = pipeline.get_stats()
    logger.info(f"Index statistics:")
    logger.info(f"  Total samples: {stats['total_samples']}")
    logger.info(f"  Categories: {stats['categories']}")
    for cat, count in stats['category_counts'].items():
        logger.info(f"    - {cat}: {count} samples")
    
    # Evaluate
    logger.info("\nEvaluating CLAP model...")
    evaluator = Evaluator()
    results = evaluator.evaluate_model(pipeline)
    
    # Print results
    logger.info("\nResults:")
    metrics = results["metrics"]
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  MAP: {metrics['map']:.4f}")
    logger.info(f"  Precision@5: {metrics.get('precision@5', 0):.4f}")
    logger.info(f"  NDCG@5: {metrics.get('ndcg@5', 0):.4f}")
    
    # Save results
    results_path = config.RETRIEVAL_RESULTS_DIR / "clap_results.json"
    import json
    with open(results_path, 'w') as f:
        json.dump({
            "model": "CLAP",
            "metrics": metrics,
            "per_class_metrics": results["per_class_metrics"]
        }, f, indent=2)
    
    logger.info(f"\nResults saved to {results_path}")
    logger.info("CLAP experiment complete!")


if __name__ == "__main__":
    run_clap_experiment()
