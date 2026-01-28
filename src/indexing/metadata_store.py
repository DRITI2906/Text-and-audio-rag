"""Metadata storage and management."""

import json
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from src.config import config


class MetadataStore:
    """Store and manage metadata for audio samples."""
    
    def __init__(self, storage_path: Optional[Path] = None):
        """
        Initialize metadata store.
        
        Args:
            storage_path: Path to store metadata (default from config)
        """
        self.storage_path = storage_path or config.METADATA_STORE_PATH
        self.metadata = {}
        
        # Load existing metadata if available
        if self.storage_path.exists():
            self.load()
    
    def add(self, sample_id: str, metadata: Dict):
        """
        Add metadata for a sample.
        
        Args:
            sample_id: Unique sample identifier
            metadata: Metadata dictionary
        """
        self.metadata[sample_id] = metadata
    
    def add_batch(self, metadata_list: List[Dict]):
        """
        Add multiple metadata entries.
        
        Args:
            metadata_list: List of metadata dictionaries (must contain 'id' field)
        """
        for metadata in metadata_list:
            if "id" not in metadata:
                raise ValueError("Metadata must contain 'id' field")
            self.add(metadata["id"], metadata)
    
    def get(self, sample_id: str) -> Optional[Dict]:
        """
        Get metadata for a sample.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            Metadata dictionary or None if not found
        """
        return self.metadata.get(sample_id)
    
    def get_batch(self, sample_ids: List[str]) -> List[Dict]:
        """
        Get metadata for multiple samples.
        
        Args:
            sample_ids: List of sample identifiers
            
        Returns:
            List of metadata dictionaries
        """
        return [self.get(sid) for sid in sample_ids if self.get(sid) is not None]
    
    def filter(self, filter_dict: Dict) -> List[Dict]:
        """
        Filter metadata by criteria.
        
        Args:
            filter_dict: Dictionary of filter criteria
            
        Returns:
            List of matching metadata dictionaries
        """
        results = []
        for sample_id, metadata in self.metadata.items():
            if self._matches_filter(metadata, filter_dict):
                results.append(metadata)
        return results
    
    def _matches_filter(self, metadata: Dict, filter_dict: Dict) -> bool:
        """Check if metadata matches filter criteria."""
        for key, value in filter_dict.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def get_all(self) -> Dict[str, Dict]:
        """Get all metadata."""
        return self.metadata.copy()
    
    def get_categories(self) -> List[str]:
        """Get list of unique categories."""
        categories = set()
        for metadata in self.metadata.values():
            if "category" in metadata:
                categories.add(metadata["category"])
        return sorted(list(categories))
    
    def save(self, path: Optional[Path] = None):
        """
        Save metadata to file.
        
        Args:
            path: Optional custom path (uses default if not provided)
        """
        save_path = path or self.storage_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
    
    def load(self, path: Optional[Path] = None):
        """
        Load metadata from file.
        
        Args:
            path: Optional custom path (uses default if not provided)
        """
        load_path = path or self.storage_path
        
        if load_path.exists():
            with open(load_path, "r") as f:
                self.metadata = json.load(f)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metadata to pandas DataFrame."""
        if not self.metadata:
            return pd.DataFrame()
        return pd.DataFrame.from_dict(self.metadata, orient="index")
    
    def from_csv(self, csv_path: Path):
        """
        Load metadata from CSV file.
        
        Args:
            csv_path: Path to CSV file
        """
        df = pd.read_csv(csv_path)
        
        for _, row in df.iterrows():
            sample_id = row.get("id") or row.get("filename")
            if sample_id:
                self.add(sample_id, row.to_dict())


if __name__ == "__main__":
    # Example usage
    store = MetadataStore()
    
    # Add metadata
    store.add("drum_01", {
        "id": "drum_01",
        "filename": "drum_01.wav",
        "category": "drums",
        "bpm": 120,
        "tags": ["acoustic", "punchy"]
    })
    
    # Get metadata
    metadata = store.get("drum_01")
    print(metadata)
    
    # Filter
    drums = store.filter({"category": "drums"})
    print(f"Found {len(drums)} drum samples")
