"""Script to evaluate models on test queries."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval.rag_pipeline import RAGPipeline
from src.embeddings.clap_embedder import CLAPEmbedder
from src.evaluation.evaluator import Evaluator
from src.data.text_generator import TextGenerator
from src.config import config
from loguru import logger


def main():
    """Evaluate model performance."""
    logger.info("Starting model evaluation...")
    
    # Initialize embedder and pipeline
    logger.info("Initializing CLAP model...")
    embedder = CLAPEmbedder()
    pipeline = RAGPipeline(embedder)
    
    # Load index
    if not config.FAISS_INDEX_PATH.exists():
        logger.error("Index not found. Please build the index first using build_index.py")
        return
    
    pipeline.load_index()
    
    # Generate test queries
    logger.info("Generating test queries...")
    text_gen = TextGenerator()
    test_queries = text_gen.generate_queries(num_queries=50)
    
    # Evaluate
    logger.info("Running evaluation...")
    evaluator = Evaluator()
    results = evaluator.evaluate_model(pipeline, test_queries)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS")
    logger.info("="*60)
    
    metrics = results["metrics"]
    logger.info(f"\nOverall Metrics:")
    logger.info(f"  Accuracy:      {metrics['accuracy']:.4f}")
    logger.info(f"  MAP:           {metrics['map']:.4f}")
    logger.info(f"  Precision@1:   {metrics.get('precision@1', 0):.4f}")
    logger.info(f"  Precision@5:   {metrics.get('precision@5', 0):.4f}")
    logger.info(f"  Precision@10:  {metrics.get('precision@10', 0):.4f}")
    logger.info(f"  NDCG@5:        {metrics.get('ndcg@5', 0):.4f}")
    logger.info(f"  NDCG@10:       {metrics.get('ndcg@10', 0):.4f}")
    
    logger.info(f"\nPer-Class Metrics:")
    for category, cat_metrics in results["per_class_metrics"].items():
        logger.info(f"  {category.capitalize()}:")
        logger.info(f"    Precision: {cat_metrics['precision']:.4f}")
        logger.info(f"    Recall:    {cat_metrics['recall']:.4f}")
        logger.info(f"    F1:        {cat_metrics['f1']:.4f}")
    
    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
