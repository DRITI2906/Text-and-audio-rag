"""Audio-only embedding model (bonus approach)."""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Union, List, Optional
from loguru import logger

from src.config import config



class AudioOnlyEmbedder:
    """Audio-only embedding model using PANNs, VGGish, or other audio encoders."""
    
    def __init__(
        self, 
        model_type: str = "panns",
        device: str = None
    ):
        """
        Initialize audio-only embedder.
        
        Args:
            model_type: Type of audio model ('panns', 'vggish', 'openl3', 'wav2vec2')
            device: Device to run model on ('cuda' or 'cpu')
        """
        self.model_type = model_type.lower()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing {self.model_type} audio embedder")
        logger.info(f"Using device: {self.device}")
        
        self.model = self._load_model()
        self.sample_rate = self._get_sample_rate()
        
        logger.info(f"{self.model_type} model loaded successfully")
    
    def _load_model(self):
        """Load the appropriate audio model based on model_type."""
        if self.model_type == "panns":
            return self._load_panns()
        elif self.model_type == "vggish":
            return self._load_vggish()
        elif self.model_type == "openl3":
            return self._load_openl3()
        elif self.model_type == "wav2vec2":
            return self._load_wav2vec2()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_panns(self):
        """Load PANNs model."""
        try:
            from panns_inference import AudioTagging
            model = AudioTagging(checkpoint_path=None, device=self.device)
            return model
        except ImportError:
            logger.error("panns_inference not installed. Install with: pip install panns-inference")
            raise
    
    def _load_vggish(self):
        """Load VGGish model."""
        # Placeholder - implement VGGish loading
        logger.warning("VGGish not yet implemented")
        return None
    
    def _load_openl3(self):
        """Load OpenL3 model."""
        try:
            import openl3
            return openl3
        except ImportError:
            logger.error("openl3 not installed. Install with: pip install openl3")
            raise
    
    def _load_wav2vec2(self):
        """Load Wav2Vec2 model."""
        try:
            from transformers import Wav2Vec2Model, Wav2Vec2Processor
            processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
            model.to(self.device)
            model.eval()
            return {"model": model, "processor": processor}
        except ImportError:
            logger.error("transformers not installed. Install with: pip install transformers")
            raise
    
    def _get_sample_rate(self) -> int:
        """Get the required sample rate for the model."""
        sample_rates = {
            "panns": 32000,
            "vggish": 16000,
            "openl3": 48000,
            "wav2vec2": 16000
        }
        return sample_rates.get(self.model_type, 44100)
    
    def load_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Preprocessed audio array
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
        return audio
    
    def embed_audio(self, audio_paths: Union[str, Path, List[Union[str, Path]]]) -> np.ndarray:
        """
        Generate embeddings for audio files.
        
        Args:
            audio_paths: Single audio path or list of audio paths
            
        Returns:
            Numpy array of embeddings
        """
        if isinstance(audio_paths, (str, Path)):
            audio_paths = [audio_paths]
        
        embeddings = []
        
        for audio_path in audio_paths:
            audio = self.load_audio(audio_path)
            
            if self.model_type == "panns":
                embedding = self._embed_panns(audio)
            elif self.model_type == "openl3":
                embedding = self._embed_openl3(audio)
            elif self.model_type == "wav2vec2":
                embedding = self._embed_wav2vec2(audio)
            else:
                raise NotImplementedError(f"Embedding not implemented for {self.model_type}")
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def _embed_panns(self, audio: np.ndarray) -> np.ndarray:
        """Generate embedding using PANNs."""
        with torch.no_grad():
            # PANNs expects audio in specific format
            audio_tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
            embedding = self.model.inference(audio_tensor)
            return embedding['embedding'].cpu().numpy().squeeze()
    
    def _embed_openl3(self, audio: np.ndarray) -> np.ndarray:
        """Generate embedding using OpenL3."""
        import openl3
        emb, ts = openl3.get_audio_embedding(
            audio, 
            self.sample_rate,
            content_type="music",
            embedding_size=512
        )
        # Average over time dimension
        return np.mean(emb, axis=0)
    
    def _embed_wav2vec2(self, audio: np.ndarray) -> np.ndarray:
        """Generate embedding using Wav2Vec2."""
        processor = self.model["processor"]
        model = self.model["model"]
        
        inputs = processor(audio, sampling_rate=self.sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Use mean pooling over time dimension
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
        
        return embedding
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings."""
        dims = {
            "panns": 2048,
            "vggish": 128,
            "openl3": 512,
            "wav2vec2": 768
        }
        return dims.get(self.model_type, 512)


if __name__ == "__main__":
    # Example usage
    embedder = AudioOnlyEmbedder(model_type="panns")
    
    # Test audio embedding (requires audio files)
    # audio_paths = ["path/to/audio1.wav", "path/to/audio2.wav"]
    # audio_emb = embedder.embed_audio(audio_paths)
    # print(f"Audio embeddings shape: {audio_emb.shape}")
