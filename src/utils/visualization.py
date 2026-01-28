"""Visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import List, Dict
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from src.config import config


class Visualizer:
    """Visualization utilities for embeddings and results."""
    
    def __init__(self):
        """Initialize visualizer."""
        sns.set_style("whitegrid")
        self.colors = sns.color_palette("husl", len(config.CATEGORIES))
        self.category_colors = {cat: self.colors[i] for i, cat in enumerate(config.CATEGORIES)}
    
    def plot_embedding_space(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        method: str = "tsne",
        title: str = "Embedding Space",
        save_path: Path = None
    ):
        """
        Visualize embedding space in 2D.
        
        Args:
            embeddings: Array of embeddings
            labels: List of category labels
            method: Dimensionality reduction method ('tsne' or 'pca')
            title: Plot title
            save_path: Path to save figure
        """
        # Reduce to 2D
        if method == "tsne":
            reducer = TSNE(n_components=2, random_state=42)
        elif method == "pca":
            reducer = PCA(n_components=2)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        for category in config.CATEGORIES:
            mask = np.array(labels) == category
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[self.category_colors[category]],
                label=category.capitalize(),
                alpha=0.7,
                s=100,
                edgecolors='black',
                linewidth=0.5
            )
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel(f"{method.upper()} Component 1", fontsize=12)
        plt.ylabel(f"{method.upper()} Component 2", fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_retrieval_scores(
        self,
        results: List[Dict],
        title: str = "Retrieval Scores",
        save_path: Path = None
    ):
        """
        Plot retrieval scores.
        
        Args:
            results: List of retrieval results
            title: Plot title
            save_path: Path to save figure
        """
        if not results:
            return
        
        categories = [r.get("category", "unknown") for r in results]
        scores = [r.get("score", 0) for r in results]
        colors = [self.category_colors.get(cat, "gray") for cat in categories]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(range(len(results)), scores, color=colors, alpha=0.7, edgecolor='black')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Rank", fontsize=12)
        plt.ylabel("Similarity Score", fontsize=12)
        plt.xticks(range(len(results)), [f"{i+1}" for i in range(len(results))])
        
        # Add legend
        handles = [plt.Rectangle((0,0),1,1, color=self.category_colors[cat]) for cat in config.CATEGORIES]
        plt.legend(handles, [cat.capitalize() for cat in config.CATEGORIES])
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_metrics_comparison(
        self,
        model_metrics: Dict[str, Dict],
        save_path: Path = None
    ):
        """
        Plot metrics comparison across models.
        
        Args:
            model_metrics: Dictionary mapping model name to metrics
            save_path: Path to save figure
        """
        metrics_to_plot = ["accuracy", "map", "precision@1", "precision@5", "ndcg@5"]
        
        models = list(model_metrics.keys())
        x = np.arange(len(metrics_to_plot))
        width = 0.8 / len(models)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i, model in enumerate(models):
            metrics = model_metrics[model]
            values = [metrics.get(m, 0) for m in metrics_to_plot]
            offset = width * i - (width * len(models) / 2)
            ax.bar(x + offset, values, width, label=model, alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_to_plot, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved comparison to {save_path}")
        else:
            plt.show()
        
        plt.close()


if __name__ == "__main__":
    print("Visualization utilities module")
