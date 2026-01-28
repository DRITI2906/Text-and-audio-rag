"""Query parsing and preprocessing."""

import re
from typing import Dict, List, Optional


class QueryParser:
    """Parse and preprocess user queries."""
    
    def __init__(self):
        """Initialize query parser."""
        self.category_keywords = {
            "drums": ["drum", "drums", "percussion", "beat", "rhythm", "kick", "snare", "hi-hat", "cymbal"],
            "keys": ["key", "keys", "piano", "keyboard", "synth", "synthesizer", "melodic", "chord"]
        }
        
        self.bpm_pattern = re.compile(r'(\d{2,3})\s*bpm', re.IGNORECASE)
        self.key_pattern = re.compile(r'\b([A-G][#b]?)\s*(major|minor|maj|min|m)?\b', re.IGNORECASE)
    
    def parse(self, query: str) -> Dict:
        """
        Parse a query into structured components.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with parsed components
        """
        query_lower = query.lower()
        
        result = {
            "original_query": query,
            "cleaned_query": self.clean_query(query),
            "category": self.extract_category(query_lower),
            "bpm": self.extract_bpm(query),
            "key": self.extract_key(query),
            "filters": {}
        }
        
        # Add filters
        if result["category"]:
            result["filters"]["category"] = result["category"]
        if result["bpm"]:
            result["filters"]["bpm"] = result["bpm"]
        if result["key"]:
            result["filters"]["key"] = result["key"]
        
        return result
    
    def clean_query(self, query: str) -> str:
        """
        Clean and normalize query text.
        
        Args:
            query: Raw query string
            
        Returns:
            Cleaned query string
        """
        # Remove extra whitespace
        query = " ".join(query.split())
        
        # Convert to lowercase for processing
        query = query.lower()
        
        # Remove special characters but keep spaces
        query = re.sub(r'[^\w\s#-]', '', query)
        
        return query.strip()
    
    def extract_category(self, query: str) -> Optional[str]:
        """
        Extract category from query.
        
        Args:
            query: Query string (lowercase)
            
        Returns:
            Category name or None
        """
        for category, keywords in self.category_keywords.items():
            for keyword in keywords:
                if keyword in query:
                    return category
        return None
    
    def extract_bpm(self, query: str) -> Optional[int]:
        """
        Extract BPM from query.
        
        Args:
            query: Query string
            
        Returns:
            BPM value or None
        """
        match = self.bpm_pattern.search(query)
        if match:
            return int(match.group(1))
        return None
    
    def extract_key(self, query: str) -> Optional[str]:
        """
        Extract musical key from query.
        
        Args:
            query: Query string
            
        Returns:
            Key string or None
        """
        match = self.key_pattern.search(query)
        if match:
            key = match.group(1)
            mode = match.group(2)
            if mode:
                return f"{key} {mode.capitalize()}"
            return key
        return None
    
    def expand_query(self, query: str) -> List[str]:
        """
        Expand query with synonyms and variations.
        
        Args:
            query: Original query
            
        Returns:
            List of query variations
        """
        variations = [query]
        
        # Add category-specific expansions
        parsed = self.parse(query)
        category = parsed["category"]
        
        if category == "drums":
            variations.extend([
                query.replace("drum", "percussion"),
                query.replace("drum", "beat"),
            ])
        elif category == "keys":
            variations.extend([
                query.replace("key", "piano"),
                query.replace("key", "keyboard"),
            ])
        
        return list(set(variations))  # Remove duplicates


if __name__ == "__main__":
    # Example usage
    parser = QueryParser()
    
    queries = [
        "give me drum samples",
        "piano keys in C major",
        "120 bpm drum loops",
        "mellow synthesizer sounds"
    ]
    
    for query in queries:
        parsed = parser.parse(query)
        print(f"\nQuery: {query}")
        print(f"Category: {parsed['category']}")
        print(f"BPM: {parsed['bpm']}")
        print(f"Key: {parsed['key']}")
        print(f"Filters: {parsed['filters']}")
