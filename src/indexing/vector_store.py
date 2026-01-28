"""Abstract vector store interface."""

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Dict, Optional


class VectorStore(ABC):
    """Abstract base class for vector stores."""
    
    @abstractmethod
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict]] = None
    ):
        """
        Add vectors to the store.
        
        Args:
            vectors: Array of vectors to add
            ids: List of unique IDs for each vector
            metadata: Optional metadata for each vector
        """
        pass
    
    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> Tuple[List[str], List[float]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            filter_dict: Optional metadata filters
            
        Returns:
            Tuple of (ids, distances)
        """
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save the index to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load the index from disk."""
        pass
    
    @abstractmethod
    def get_vector_count(self) -> int:
        """Get the number of vectors in the store."""
        pass
