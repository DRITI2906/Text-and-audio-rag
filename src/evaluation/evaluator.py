"""Main evaluator for comparing models."""

from typing import List, Dict
from pathlib import Path
import json

from src.retrieval.rag_pipeline import RAGPipeline
from src.data.text_generator import TextGenerator
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.confusion_matrix import ConfusionMatrixGenerator
from src.config import config


class Evaluator:
    """Evaluate and compare different embedding models."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.text_generator = TextGenerator()
        self.metrics_calculator = EvaluationMetrics()
        self.cm_generator = ConfusionMatrixGenerator()
    
    def evaluate_model(
        self,
        pipeline: RAGPipeline,
        test_queries: List[Dict] = None,
        k_values: List[int] = [1, 5, 10]
    ) -> Dict:
        """
        Evaluate a single model.
        
        Args:
            pipeline: RAG pipeline to evaluate
            test_queries: List of test queries with expected categories
            k_values: List of K values for metrics
            
        Returns:
            Dictionary of evaluation results
        """
        # Generate test queries if not provided
        if test_queries is None:
            test_queries = self.text_generator.generate_queries(num_queries=20)
        
        # Run queries
        all_results = []
        expected_categories = []
        predictions = []
        
        for test_query in test_queries:
            query = test_query["query"]
            expected_cat = test_query["expected_category"]
            
            results = pipeline.query(query, k=max(k_values))
            all_results.append(results)
            expected_categories.append(expected_cat)
            
            # Get top-1 prediction
            if results:
                predictions.append(results[0].get("category", "unknown"))
            else:
                predictions.append("unknown")
        
        # Calculate metrics
        metrics = self.metrics_calculator.calculate_all_metrics(
            all_results,
            expected_categories,
            k_values=k_values
        )
        
        # Generate confusion matrix
        cm = self.cm_generator.generate_matrix(predictions, expected_categories)
        per_class_metrics = self.cm_generator.calculate_per_class_metrics(cm)
        
        return {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "per_class_metrics": per_class_metrics,
            "predictions": predictions,
            "expected_categories": expected_categories
        }
    
    def compare_models(
        self,
        pipelines: Dict[str, RAGPipeline],
        test_queries: List[Dict] = None,
        save_results: bool = True
    ) -> Dict[str, Dict]:
        """
        Compare multiple models.
        
        Args:
            pipelines: Dictionary mapping model name to pipeline
            test_queries: List of test queries
            save_results: Whether to save results to disk
            
        Returns:
            Dictionary mapping model name to evaluation results
        """
        # Generate test queries if not provided
        if test_queries is None:
            test_queries = self.text_generator.generate_queries(num_queries=50)
        
        results = {}
        model_predictions = {}
        expected_categories = [q["expected_category"] for q in test_queries]
        
        # Evaluate each model
        for model_name, pipeline in pipelines.items():
            print(f"\nEvaluating {model_name}...")
            model_results = self.evaluate_model(pipeline, test_queries)
            results[model_name] = model_results
            model_predictions[model_name] = model_results["predictions"]
        
        # Generate comparison confusion matrices
        self.cm_generator.compare_models(
            model_predictions,
            expected_categories
        )
        
        # Save results if requested
        if save_results:
            self.save_results(results)
        
        # Print summary
        self.print_comparison_summary(results)
        
        return results
    
    def save_results(self, results: Dict[str, Dict]):
        """Save evaluation results to JSON."""
        results_dir = config.RETRIEVAL_RESULTS_DIR
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for model_name, model_results in results.items():
            serializable_results[model_name] = {
                "metrics": model_results["metrics"],
                "per_class_metrics": model_results["per_class_metrics"],
                "confusion_matrix": model_results["confusion_matrix"]
            }
        
        save_path = results_dir / "evaluation_results.json"
        with open(save_path, "w") as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nResults saved to {save_path}")
    
    def print_comparison_summary(self, results: Dict[str, Dict]):
        """Print comparison summary."""
        print("\n" + "="*60)
        print("MODEL COMPARISON SUMMARY")
        print("="*60)
        
        for model_name, model_results in results.items():
            metrics = model_results["metrics"]
            print(f"\n{model_name}:")
            print(f"  Accuracy:      {metrics['accuracy']:.4f}")
            print(f"  MAP:           {metrics['map']:.4f}")
            print(f"  Precision@1:   {metrics.get('precision@1', 0):.4f}")
            print(f"  Precision@5:   {metrics.get('precision@5', 0):.4f}")
            print(f"  NDCG@5:        {metrics.get('ndcg@5', 0):.4f}")


if __name__ == "__main__":
    print("Evaluator - use with initialized RAG pipelines")
