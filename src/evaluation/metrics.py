"""Evaluation metrics for retrieval systems."""

import numpy as np
from typing import List, Dict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class EvaluationMetrics:
    """Calculate evaluation metrics for retrieval results."""
    
    @staticmethod
    def precision_at_k(results: List[Dict], k: int, expected_category: str) -> float:
        """
        Calculate Precision@K.
        
        Args:
            results: List of retrieval results
            k: Number of top results to consider
            expected_category: Expected category
            
        Returns:
            Precision@K score
        """
        if not results or k == 0:
            return 0.0
        
        top_k = results[:k]
        relevant = sum(1 for r in top_k if r.get("category") == expected_category)
        return relevant / k
    
    @staticmethod
    def recall_at_k(
        results: List[Dict],
        k: int,
        expected_category: str,
        total_relevant: int
    ) -> float:
        """
        Calculate Recall@K.
        
        Args:
            results: List of retrieval results
            k: Number of top results to consider
            expected_category: Expected category
            total_relevant: Total number of relevant items in dataset
            
        Returns:
            Recall@K score
        """
        if not results or k == 0 or total_relevant == 0:
            return 0.0
        
        top_k = results[:k]
        relevant = sum(1 for r in top_k if r.get("category") == expected_category)
        return relevant / total_relevant
    
    @staticmethod
    def mean_average_precision(
        all_results: List[List[Dict]],
        expected_categories: List[str]
    ) -> float:
        """
        Calculate Mean Average Precision (MAP).
        
        Args:
            all_results: List of result lists for each query
            expected_categories: List of expected categories for each query
            
        Returns:
            MAP score
        """
        if not all_results:
            return 0.0
        
        average_precisions = []
        
        for results, expected_cat in zip(all_results, expected_categories):
            precisions = []
            relevant_count = 0
            
            for i, result in enumerate(results, 1):
                if result.get("category") == expected_cat:
                    relevant_count += 1
                    precision = relevant_count / i
                    precisions.append(precision)
            
            if precisions:
                average_precisions.append(np.mean(precisions))
            else:
                average_precisions.append(0.0)
        
        return np.mean(average_precisions)
    
    @staticmethod
    def ndcg_at_k(results: List[Dict], k: int, expected_category: str) -> float:
        """
        Calculate Normalized Discounted Cumulative Gain (NDCG@K).
        
        Args:
            results: List of retrieval results
            k: Number of top results to consider
            expected_category: Expected category
            
        Returns:
            NDCG@K score
        """
        if not results or k == 0:
            return 0.0
        
        top_k = results[:k]
        
        # Calculate DCG
        dcg = 0.0
        for i, result in enumerate(top_k, 1):
            relevance = 1.0 if result.get("category") == expected_category else 0.0
            dcg += relevance / np.log2(i + 1)
        
        # Calculate IDCG (ideal DCG)
        ideal_relevances = [1.0] * min(k, len([r for r in results if r.get("category") == expected_category]))
        idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_relevances))
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    @staticmethod
    def category_accuracy(results: List[Dict], expected_categories: List[str]) -> float:
        """
        Calculate category prediction accuracy (top-1).
        
        Args:
            results: List of top results for each query
            expected_categories: List of expected categories
            
        Returns:
            Accuracy score
        """
        if not results or not expected_categories:
            return 0.0
        
        correct = sum(
            1 for result, expected in zip(results, expected_categories)
            if result.get("category") == expected
        )
        
        return correct / len(expected_categories)
    
    @staticmethod
    def calculate_all_metrics(
        all_results: List[List[Dict]],
        expected_categories: List[str],
        k_values: List[int] = [1, 5, 10]
    ) -> Dict:
        """
        Calculate all metrics for a set of queries.
        
        Args:
            all_results: List of result lists for each query
            expected_categories: List of expected categories
            k_values: List of K values to evaluate
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate metrics for each K
        for k in k_values:
            precisions = []
            ndcgs = []
            
            for results, expected_cat in zip(all_results, expected_categories):
                p_at_k = EvaluationMetrics.precision_at_k(results, k, expected_cat)
                ndcg = EvaluationMetrics.ndcg_at_k(results, k, expected_cat)
                
                precisions.append(p_at_k)
                ndcgs.append(ndcg)
            
            metrics[f"precision@{k}"] = np.mean(precisions)
            metrics[f"ndcg@{k}"] = np.mean(ndcgs)
        
        # Calculate MAP
        metrics["map"] = EvaluationMetrics.mean_average_precision(
            all_results, expected_categories
        )
        
        # Calculate top-1 accuracy
        top_1_results = [results[0] if results else {} for results in all_results]
        metrics["accuracy"] = EvaluationMetrics.category_accuracy(
            top_1_results, expected_categories
        )
        
        return metrics


if __name__ == "__main__":
    # Example usage
    results = [
        [{"category": "drums", "score": 0.9}, {"category": "keys", "score": 0.7}],
        [{"category": "keys", "score": 0.95}, {"category": "keys", "score": 0.8}],
    ]
    expected = ["drums", "keys"]
    
    metrics = EvaluationMetrics.calculate_all_metrics(results, expected, k_values=[1, 2])
    print("Metrics:", metrics)
