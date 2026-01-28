"""Simple chatbot interface for querying audio samples."""

from typing import Optional
from pathlib import Path

from src.retrieval.rag_pipeline import RAGPipeline
from src.embeddings.clap_embedder import CLAPEmbedder
from src.config import config


class AudioChatbot:
    """Interactive chatbot for audio sample retrieval."""
    
    def __init__(self, pipeline: Optional[RAGPipeline] = None):
        """
        Initialize chatbot.
        
        Args:
            pipeline: RAG pipeline (creates new one if None)
        """
        if pipeline is None:
            embedder = CLAPEmbedder()
            self.pipeline = RAGPipeline(embedder)
            
            # Load index if available
            if config.FAISS_INDEX_PATH.exists():
                self.pipeline.load_index()
            else:
                print("Warning: No index found. Please build the index first.")
        else:
            self.pipeline = pipeline
    
    def query(self, text: str, k: int = 5) -> None:
        """
        Query for audio samples and display results.
        
        Args:
            text: Query text
            k: Number of results to return
        """
        print(f"\n🔍 Searching for: '{text}'")
        print("=" * 60)
        
        results = self.pipeline.query(text, k=k, return_paths=True)
        
        if not results:
            print("No results found.")
            return
        
        print(f"\nFound {len(results)} results:\n")
        
        for i, result in enumerate(results, 1):
            print(f"{i}. {result.get('filename', 'Unknown')}")
            print(f"   Category: {result.get('category', 'Unknown')}")
            print(f"   Score: {result.get('score', 0):.4f}")
            
            if result.get('tags'):
                tags = result['tags']
                if isinstance(tags, list):
                    print(f"   Tags: {', '.join(tags)}")
                else:
                    print(f"   Tags: {tags}")
            
            if result.get('bpm'):
                print(f"   BPM: {result['bpm']}")
            
            if result.get('key'):
                print(f"   Key: {result['key']}")
            
            print()
    
    def interactive_mode(self):
        """Run interactive chatbot mode."""
        print("\n" + "="*60)
        print("🎵 Audio Sample Chatbot")
        print("="*60)
        print("\nType your queries to search for audio samples.")
        print("Commands:")
        print("  - Type 'quit' or 'exit' to quit")
        print("  - Type 'stats' to see index statistics")
        print("="*60 + "\n")
        
        while True:
            try:
                query = input("You: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("\nGoodbye! 👋")
                    break
                
                if query.lower() == 'stats':
                    stats = self.pipeline.get_stats()
                    print("\n📊 Index Statistics:")
                    print(f"   Total samples: {stats['total_samples']}")
                    print(f"   Categories: {', '.join(stats['categories'])}")
                    for cat, count in stats['category_counts'].items():
                        print(f"   - {cat}: {count} samples")
                    print()
                    continue
                
                self.query(query)
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    """Main function to run the chatbot."""
    chatbot = AudioChatbot()
    chatbot.interactive_mode()


if __name__ == "__main__":
    main()
