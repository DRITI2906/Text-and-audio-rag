"""Confusion matrix generation and visualization."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Dict
from pathlib import Path

from src.config import config


class ConfusionMatrixGenerator:
    """Generate and visualize confusion matrices for model comparison."""
    
    def __init__(self, categories: List[str] = None):
        """
        Initialize confusion matrix generator.
        
        Args:
            categories: List of category names (uses config if None)
        """
        self.categories = categories or config.CATEGORIES
    
    def generate_matrix(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> np.ndarray:
        """
        Generate confusion matrix.
        
        Args:
            predictions: List of predicted categories
            ground_truth: List of true categories
            
        Returns:
            Confusion matrix as numpy array
        """
        return confusion_matrix(
            ground_truth,
            predictions,
            labels=self.categories
        )
    
    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        title: str = "Confusion Matrix",
        save_path: Path = None,
        normalize: bool = False
    ):
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            title: Plot title
            save_path: Path to save figure (optional)
            normalize: Whether to normalize the matrix
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
        else:
            fmt = 'd'
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap='Blues',
            xticklabels=self.categories,
            yticklabels=self.categories,
            cbar_kws={'label': 'Count' if not normalize else 'Proportion'}
        )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Category', fontsize=12)
        plt.xlabel('Predicted Category', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved confusion matrix to {save_path}")
        
        plt.close()
    
    def compare_models(
        self,
        model_results: Dict[str, List[str]],
        ground_truth: List[str],
        save_dir: Path = None
    ):
        """
        Compare multiple models using confusion matrices.
        
        Args:
            model_results: Dictionary mapping model name to predictions
            ground_truth: List of true categories
            save_dir: Directory to save plots (uses config if None)
        """
        save_dir = save_dir or config.CONFUSION_MATRICES_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for model_name, predictions in model_results.items():
            # Generate confusion matrix
            cm = self.generate_matrix(predictions, ground_truth)
            
            # Plot and save
            save_path = save_dir / f"{model_name}_confusion_matrix.png"
            self.plot_confusion_matrix(
                cm,
                title=f"Confusion Matrix - {model_name}",
                save_path=save_path,
                normalize=False
            )
            
            # Also save normalized version
            save_path_norm = save_dir / f"{model_name}_confusion_matrix_normalized.png"
            self.plot_confusion_matrix(
                cm,
                title=f"Normalized Confusion Matrix - {model_name}",
                save_path=save_path_norm,
                normalize=True
            )
    
    def calculate_per_class_metrics(
        self,
        cm: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate per-class precision, recall, and F1 from confusion matrix.
        
        Args:
            cm: Confusion matrix
            
        Returns:
            Dictionary mapping category to metrics
        """
        metrics = {}
        
        for i, category in enumerate(self.categories):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[category] = {
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        
        return metrics


if __name__ == "__main__":
    # Example usage
    generator = ConfusionMatrixGenerator(categories=["drums", "keys"])
    
    # Dummy data
    predictions = ["drums", "drums", "keys", "keys", "drums"]
    ground_truth = ["drums", "keys", "keys", "keys", "drums"]
    
    # Generate and plot
    cm = generator.generate_matrix(predictions, ground_truth)
    print("Confusion Matrix:\n", cm)
    
    # Calculate metrics
    metrics = generator.calculate_per_class_metrics(cm)
    print("\nPer-class metrics:", metrics)
