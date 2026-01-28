"""Baseline experiments for comparison."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data.text_generator import TextGenerator
from src.config import config
from loguru import logger
import random


def run_random_baseline():
    """Run random baseline experiment."""
    logger.info("="*60)
    logger.info("Random Baseline Experiment")
    logger.info("="*60)
    
    # Generate test queries
    text_gen = TextGenerator()
    test_queries = text_gen.generate_queries(num_queries=50)
    
    # Random predictions
    predictions = []
    expected = []
    
    for query in test_queries:
        predictions.append(random.choice(config.CATEGORIES))
        expected.append(query["expected_category"])
    
    # Calculate accuracy
    correct = sum(1 for p, e in zip(predictions, expected) if p == e)
    accuracy = correct / len(expected)
    
    logger.info(f"\nRandom Baseline Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Expected: ~{1/len(config.CATEGORIES):.4f} (random chance)")
    
    return accuracy


def run_keyword_baseline():
    """Run simple keyword matching baseline."""
    logger.info("="*60)
    logger.info("Keyword Matching Baseline")
    logger.info("="*60)
    
    # Generate test queries
    text_gen = TextGenerator()
    test_queries = text_gen.generate_queries(num_queries=50)
    
    # Keyword matching
    predictions = []
    expected = []
    
    for query in test_queries:
        query_text = query["query"].lower()
        
        # Simple keyword matching
        if any(kw in query_text for kw in ["drum", "percussion", "beat", "kick", "snare"]):
            pred = "drums"
        elif any(kw in query_text for kw in ["key", "piano", "keyboard", "synth"]):
            pred = "keys"
        else:
            pred = random.choice(config.CATEGORIES)
        
        predictions.append(pred)
        expected.append(query["expected_category"])
    
    # Calculate accuracy
    correct = sum(1 for p, e in zip(predictions, expected) if p == e)
    accuracy = correct / len(expected)
    
    logger.info(f"\nKeyword Baseline Results:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    
    return accuracy


if __name__ == "__main__":
    # Run baselines
    random_acc = run_random_baseline()
    keyword_acc = run_keyword_baseline()
    
    logger.info("\n" + "="*60)
    logger.info("Baseline Comparison")
    logger.info("="*60)
    logger.info(f"Random Baseline:  {random_acc:.4f}")
    logger.info(f"Keyword Baseline: {keyword_acc:.4f}")
