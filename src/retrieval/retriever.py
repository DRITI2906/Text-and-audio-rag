"""Retriever for similarity search."""

import numpy as np
from typing import List, Dict, Optional, Tuple

from src.indexing.vector_store import VectorStore
from src.indexing.metadata_store import MetadataStore
from src.embeddings.base_embedder import BaseEmbedder
from src.retrieval.query_parser import QueryParser
from src.config import config


class Retriever:
    """Retrieve similar audio samples based on text queries."""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
        metadata_store: MetadataStore,
        query_parser: Optional[QueryParser] = None
    ):
        """
        Initialize retriever.
        
        Args:
            embedder: Embedding model
            vector_store: Vector index
            metadata_store: Metadata storage
            query_parser: Optional query parser
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.metadata_store = metadata_store
        self.query_parser = query_parser or QueryParser()
    
    def retrieve(
        self,
        query: str,
        k: int = None,
        use_filters: bool = True
    ) -> List[Dict]:
        """
        Retrieve similar samples for a query.
        
        Args:
            query: Text query
            k: Number of results (default from config)
            use_filters: Whether to use metadata filters from query
            
        Returns:
            List of result dictionaries with metadata and scores
        """
        k = k or config.TOP_K_RESULTS
        
        # Parse query
        parsed_query = self.query_parser.parse(query)
        
        # Generate query embedding
        query_embedding = self.embedder.embed_text(parsed_query["cleaned_query"])
        
        # Prepare filters
        filters = parsed_query["filters"] if use_filters else None
        
        # Search vector store
        result_ids, distances = self.vector_store.search(
            query_embedding,
            k=k,
            filter_dict=filters
        )
        
        # Get metadata for results
        results = []
        for sample_id, distance in zip(result_ids, distances):
            metadata = self.metadata_store.get(sample_id)
            if metadata:
                result = metadata.copy()
                result["score"] = float(distance)
                result["query"] = query
                results.append(result)
        
        return results
    
    def retrieve_batch(
        self,
        queries: List[str],
        k: int = None
    ) -> List[List[Dict]]:
        """
        Retrieve results for multiple queries.
        
        Args:
            queries: List of text queries
            k: Number of results per query
            
        Returns:
            List of result lists
        """
        return [self.retrieve(query, k=k) for query in queries]
    
    def get_similar_samples(
        self,
        sample_id: str,
        k: int = 5
    ) -> List[Dict]:
        """
        Find samples similar to a given sample.
        
        Args:
            sample_id: ID of the reference sample
            k: Number of similar samples to return
            
        Returns:
            List of similar samples with metadata
        """
        # Get metadata for reference sample
        metadata = self.metadata_store.get(sample_id)
        if not metadata:
            return []
        
        # Create query from metadata
        query_parts = []
        if "category" in metadata:
            query_parts.append(metadata["category"])
        if "tags" in metadata:
            tags = metadata["tags"]
            if isinstance(tags, list):
                query_parts.extend(tags[:3])  # Use first 3 tags
            elif isinstance(tags, str):
                query_parts.append(tags)
        
        query = " ".join(query_parts)
        
        # Retrieve similar samples
        results = self.retrieve(query, k=k+1, use_filters=False)
        
        # Filter out the reference sample itself
        results = [r for r in results if r.get("id") != sample_id]
        
        return results[:k]


if __name__ == "__main__":
    print("Retriever module - use with initialized embedder and vector store")
