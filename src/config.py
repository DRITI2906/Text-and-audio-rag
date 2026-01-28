"""Central configuration for the multimodal audio-text retrieval system."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Configuration class for the project."""
    
    # Project paths
    BASE_DIR = Path(__file__).parent.parent
    DATA_DIR = BASE_DIR / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    METADATA_DIR = DATA_DIR / "metadata"
    
    # Results paths
    RESULTS_DIR = BASE_DIR / "results"
    EMBEDDINGS_RESULTS_DIR = RESULTS_DIR / "embeddings"
    CONFUSION_MATRICES_DIR = RESULTS_DIR / "confusion_matrices"
    RETRIEVAL_RESULTS_DIR = RESULTS_DIR / "retrieval_results"
    PLOTS_DIR = RESULTS_DIR / "plots"
    
    # Audio categories
    CATEGORIES = ["drums", "keys"]
    SAMPLES_PER_CATEGORY = 20
    
    # Audio processing parameters
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "44100"))
    AUDIO_DURATION = int(os.getenv("AUDIO_DURATION", "10"))
    N_MELS = int(os.getenv("N_MELS", "128"))
    N_FFT = int(os.getenv("N_FFT", "2048"))
    HOP_LENGTH = int(os.getenv("HOP_LENGTH", "512"))
    
    # Model configuration
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "clap")  # clap, audio_only
    CLAP_MODEL_NAME = os.getenv("CLAP_MODEL_NAME", "laion/larger_clap_music")
    TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
    AUDIO_MODEL_NAME = os.getenv("AUDIO_MODEL_NAME", "panns")  # panns, vggish, openl3, wav2vec2
    
    # Vector store configuration
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")  # faiss, chromadb, pinecone
    FAISS_INDEX_PATH = PROCESSED_DATA_DIR / "embeddings" / "faiss_index"
    METADATA_STORE_PATH = PROCESSED_DATA_DIR / "embeddings" / "metadata.json"
    
    # Retrieval configuration
    TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "10"))
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    DISTANCE_METRIC = os.getenv("DISTANCE_METRIC", "cosine")  # cosine, euclidean, dot_product
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    
    # Device configuration
    DEVICE = os.getenv("DEVICE", "cuda")  # cuda, cpu
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = BASE_DIR / "logs" / "app.log"
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories."""
        directories = [
            cls.RAW_DATA_DIR / "drums",
            cls.RAW_DATA_DIR / "keys",
            cls.PROCESSED_DATA_DIR / "audio_features",
            cls.PROCESSED_DATA_DIR / "embeddings",
            cls.METADATA_DIR,
            cls.EMBEDDINGS_RESULTS_DIR,
            cls.CONFUSION_MATRICES_DIR,
            cls.RETRIEVAL_RESULTS_DIR,
            cls.PLOTS_DIR,
            cls.LOG_FILE.parent,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_category_path(cls, category: str) -> Path:
        """Get the path for a specific category."""
        return cls.RAW_DATA_DIR / category
    
    @classmethod
    def get_samples_csv_path(cls) -> Path:
        """Get the path to the samples CSV file."""
        return cls.METADATA_DIR / "samples.csv"


# Create singleton instance
config = Config()

# Create directories on import
config.create_directories()
