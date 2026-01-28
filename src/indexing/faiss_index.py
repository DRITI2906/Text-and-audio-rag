"""FAISS-based vector index implementation."""

import faiss
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import pickle

from src.indexing.vector_store import VectorStore
from src.config import config


class FAISSIndex(VectorStore):
    """FAISS-based vector index for similarity search."""
    
    def __init__(self, dimension: int, metric: str = "cosine"):
        """
        Initialize FAISS index.
        
        Args:
            dimension: Dimension of vectors
            metric: Distance metric ('cosine', 'euclidean', 'dot_product')
        """
        self.dimension = dimension
        self.metric = metric
        self.id_map = {}  # Map from internal index to external ID
        self.reverse_id_map = {}  # Map from external ID to internal index
        self.metadata_map = {}  # Map from ID to metadata
        self.current_index = 0
        
        # Create appropriate FAISS index based on metric
        if metric == "cosine":
            # For cosine similarity, normalize vectors and use inner product
            self.index = faiss.IndexFlatIP(dimension)
        elif metric == "euclidean":
            self.index = faiss.IndexFlatL2(dimension)
        elif metric == "dot_product":
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        ids: List[str],
        metadata: Optional[List[Dict]] = None
    ):
        """Add vectors to the index."""
        if len(vectors) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")
        
        # Normalize vectors for cosine similarity
        if self.metric == "cosine":
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Convert to float32 for FAISS
        vectors = vectors.astype(np.float32)
        
        # Add to index
        self.index.add(vectors)
        
        # Update ID mappings
        for i, external_id in enumerate(ids):
            internal_idx = self.current_index + i
            self.id_map[internal_idx] = external_id
            self.reverse_id_map[external_id] = internal_idx
            
            if metadata and i < len(metadata):
                self.metadata_map[external_id] = metadata[i]
        
        self.current_index += len(vectors)
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> Tuple[List[str], List[float]]:
        """Search for similar vectors."""
        # Normalize query for cosine similarity
        if self.metric == "cosine":
            query_vector = query_vector / np.linalg.norm(query_vector)
        
        # Ensure correct shape and type
        if len(query_vector.shape) == 1:
            query_vector = query_vector.reshape(1, -1)
        query_vector = query_vector.astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Convert to external IDs
        result_ids = []
        result_distances = []
        
        for idx, dist in zip(indices[0], distances[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            
            external_id = self.id_map.get(idx)
            if external_id is None:
                continue
            
            # Apply metadata filters if provided
            if filter_dict:
                metadata = self.metadata_map.get(external_id, {})
                if not self._matches_filter(metadata, filter_dict):
                    continue
            
            result_ids.append(external_id)
            result_distances.append(float(dist))
        
        return result_ids, result_distances
    
    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def save(self, path: str):
        """Save index to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save mappings
        with open(path / "mappings.pkl", "wb") as f:
            pickle.dump({
                "id_map": self.id_map,
                "reverse_id_map": self.reverse_id_map,
                "metadata_map": self.metadata_map,
                "current_index": self.current_index,
                "dimension": self.dimension,
                "metric": self.metric,
            }, f)
    
    def load(self, path: str):
        """Load index from disk."""
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load mappings
        with open(path / "mappings.pkl", "rb") as f:
            data = pickle.load(f)
            self.id_map = data["id_map"]
            self.reverse_id_map = data["reverse_id_map"]
            self.metadata_map = data["metadata_map"]
            self.current_index = data["current_index"]
            self.dimension = data["dimension"]
            self.metric = data["metric"]
    
    def get_vector_count(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal


if __name__ == "__main__":
    # Example usage
    index = FAISSIndex(dimension=512, metric="cosine")
    
    # Add some dummy vectors
    vectors = np.random.randn(10, 512).astype(np.float32)
    ids = [f"sample_{i}" for i in range(10)]
    metadata = [{"category": "drums" if i < 5 else "keys"} for i in range(10)]
    
    index.add_vectors(vectors, ids, metadata)
    
    # Search
    query = np.random.randn(512).astype(np.float32)
    result_ids, distances = index.search(query, k=5)
    
    print(f"Found {len(result_ids)} results")
    for rid, dist in zip(result_ids, distances):
        print(f"  {rid}: {dist:.4f}")
