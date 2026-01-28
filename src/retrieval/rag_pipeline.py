"""RAG pipeline for end-to-end retrieval."""

from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

from src.embeddings.base_embedder import BaseEmbedder
from src.indexing.faiss_index import FAISSIndex
from src.indexing.metadata_store import MetadataStore
from src.retrieval.retriever import Retriever
from src.retrieval.query_parser import QueryParser
from src.data.audio_loader import AudioLoader
from src.config import config


class RAGPipeline:
    """End-to-end RAG pipeline for audio-text retrieval."""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        index_path: Optional[Path] = None
    ):
        """
        Initialize RAG pipeline.
        
        Args:
            embedder: Embedding model
            index_path: Path to load existing index (optional)
        """
        self.embedder = embedder
        self.embedding_dim = embedder.get_embedding_dim()
        
        # Initialize components
        self.vector_store = FAISSIndex(
            dimension=self.embedding_dim,
            metric=config.DISTANCE_METRIC
        )
        self.metadata_store = MetadataStore()
        self.query_parser = QueryParser()
        self.retriever = Retriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            metadata_store=self.metadata_store,
            query_parser=self.query_parser
        )
        
        # Load existing index if provided
        if index_path and index_path.exists():
            self.load_index(index_path)
    
    def build_index(
        self,
        audio_paths: Optional[List[Path]] = None,
        metadata_list: Optional[List[Dict]] = None
    ):
        """
        Build the index from audio files.
        
        Args:
            audio_paths: List of audio file paths (loads from config if None)
            metadata_list: List of metadata dictionaries
        """
        # Load audio files if not provided
        if audio_paths is None:
            loader = AudioLoader()
            all_samples = loader.load_all_samples()
            audio_paths = []
            metadata_list = []
            
            for category, samples in all_samples.items():
                for audio, filename in samples:
                    file_path = config.get_category_path(category) / filename
                    audio_paths.append(file_path)
                    metadata_list.append({
                        "id": filename.replace(".wav", ""),
                        "filename": filename,
                        "category": category
                    })
        
        if not audio_paths:
            raise ValueError("No audio files to index")
        
        # Generate embeddings
        print(f"Generating embeddings for {len(audio_paths)} samples...")
        embeddings = self.embedder.embed_audio(audio_paths)
        
        # Extract IDs from metadata
        ids = [m["id"] for m in metadata_list]
        
        # Add to vector store
        print("Adding to vector store...")
        self.vector_store.add_vectors(embeddings, ids, metadata_list)
        
        # Add to metadata store
        print("Adding to metadata store...")
        self.metadata_store.add_batch(metadata_list)
        
        print(f"Index built with {len(ids)} samples")
    
    def query(
        self,
        query: str,
        k: int = None,
        return_paths: bool = True
    ) -> List[Dict]:
        """
        Query the index.
        
        Args:
            query: Text query
            k: Number of results
            return_paths: Whether to include file paths in results
            
        Returns:
            List of results with metadata
        """
        results = self.retriever.retrieve(query, k=k)
        
        # Add file paths if requested
        if return_paths:
            for result in results:
                category = result.get("category")
                filename = result.get("filename")
                if category and filename:
                    result["file_path"] = str(
                        config.get_category_path(category) / filename
                    )
        
        return results
    
    def save_index(self, path: Optional[Path] = None):
        """
        Save the index to disk.
        
        Args:
            path: Save path (uses config default if None)
        """
        save_path = path or config.FAISS_INDEX_PATH
        
        print(f"Saving index to {save_path}...")
        self.vector_store.save(str(save_path))
        self.metadata_store.save()
        print("Index saved successfully")
    
    def load_index(self, path: Optional[Path] = None):
        """
        Load the index from disk.
        
        Args:
            path: Load path (uses config default if None)
        """
        load_path = path or config.FAISS_INDEX_PATH
        
        print(f"Loading index from {load_path}...")
        self.vector_store.load(str(load_path))
        self.metadata_store.load()
        print(f"Index loaded with {self.vector_store.get_vector_count()} samples")
    
    def get_stats(self) -> Dict:
        """Get statistics about the index."""
        categories = self.metadata_store.get_categories()
        category_counts = {}
        
        for category in categories:
            results = self.metadata_store.filter({"category": category})
            category_counts[category] = len(results)
        
        return {
            "total_samples": self.vector_store.get_vector_count(),
            "categories": categories,
            "category_counts": category_counts,
            "embedding_dim": self.embedding_dim
        }


if __name__ == "__main__":
    print("RAG Pipeline - use with initialized embedder")
