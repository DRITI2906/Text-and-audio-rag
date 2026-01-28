"""Generate text descriptions and queries for audio samples."""

import pandas as pd
from typing import List, Dict
import random

from src.config import config


class TextGenerator:
    """Generate text descriptions and queries for audio samples."""
    
    def __init__(self):
        """Initialize TextGenerator."""
        self.category_templates = {
            "drums": [
                "drum samples",
                "drum loops",
                "percussion sounds",
                "drum hits",
                "rhythmic drum patterns",
                "acoustic drums",
                "electronic drum sounds",
                "drum kit samples",
            ],
            "keys": [
                "piano samples",
                "keyboard sounds",
                "key samples",
                "piano keys",
                "melodic keys",
                "synthesizer keys",
                "acoustic piano",
                "electric piano sounds",
            ]
        }
    
    def generate_query_for_category(self, category: str) -> str:
        """
        Generate a random query for a category.
        
        Args:
            category: Category name (drums or keys)
            
        Returns:
            Query string
        """
        templates = self.category_templates.get(category, [category])
        return random.choice(templates)
    
    def generate_queries(self, num_queries: int = 10) -> List[Dict[str, str]]:
        """
        Generate multiple queries for testing.
        
        Args:
            num_queries: Number of queries to generate
            
        Returns:
            List of dictionaries with query and expected category
        """
        queries = []
        for _ in range(num_queries):
            category = random.choice(config.CATEGORIES)
            query = self.generate_query_for_category(category)
            queries.append({
                "query": query,
                "expected_category": category
            })
        return queries
    
    def generate_description_from_metadata(self, metadata: Dict) -> str:
        """
        Generate a text description from sample metadata.
        
        Args:
            metadata: Dictionary with sample metadata
            
        Returns:
            Text description
        """
        parts = []
        
        if metadata.get("title"):
            parts.append(metadata["title"])
        
        if metadata.get("category"):
            parts.append(f"{metadata['category']} sample")
        
        if metadata.get("bpm"):
            parts.append(f"{metadata['bpm']} BPM")
        
        if metadata.get("key"):
            parts.append(f"in {metadata['key']}")
        
        if metadata.get("tags"):
            if isinstance(metadata["tags"], list):
                parts.extend(metadata["tags"])
            elif isinstance(metadata["tags"], str):
                parts.extend(metadata["tags"].split(","))
        
        return " ".join(parts)
    
    def create_test_queries(self) -> List[str]:
        """
        Create a standard set of test queries.
        
        Returns:
            List of test queries
        """
        return [
            # Drums queries
            "give me drum samples",
            "find drum loops",
            "show me percussion sounds",
            "rhythmic drum patterns",
            "acoustic drum hits",
            
            # Keys queries
            "piano samples",
            "keyboard sounds",
            "melodic piano keys",
            "synthesizer sounds",
            "electric piano",
            
            # Mixed/specific queries
            "upbeat drum samples",
            "mellow piano keys",
            "fast percussion",
            "soft keyboard sounds",
        ]


if __name__ == "__main__":
    # Example usage
    generator = TextGenerator()
    
    # Generate test queries
    queries = generator.generate_queries(num_queries=5)
    for q in queries:
        print(f"Query: '{q['query']}' -> Expected: {q['expected_category']}")
    
    # Generate description
    metadata = {
        "title": "Acoustic Kick",
        "category": "drums",
        "bpm": 120,
        "tags": ["acoustic", "punchy", "deep"]
    }
    description = generator.generate_description_from_metadata(metadata)
    print(f"\nDescription: {description}")
