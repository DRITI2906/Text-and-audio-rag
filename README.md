# Multimodal Audio-Text Retrieval System

A production-ready system for indexing and querying audio samples using text queries through multimodal embeddings.

## 🎯 Overview

This system creates a common embedding space for text and audio, enabling natural language queries to retrieve relevant audio samples. It supports two approaches:
1. **CLAP**: Unified text-audio embeddings
2. **Audio-Only**: Separate audio embeddings with text alignment

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/DRITI2906/Text-and-audio-rag.git
cd Text-and-audio-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your settings
```

### 2. Collect Data

Place 40 audio samples (20 drums + 20 keys) in:
- `data/raw/drums/` - drum_01.wav through drum_20.wav
- `data/raw/keys/` - key_01.wav through key_20.wav

Create `data/metadata/samples.csv` with sample metadata.

### 3. Build Index

```bash
python scripts/build_index.py
```

### 4. Query Samples

```bash
# Interactive chatbot
python app/chatbot.py

# Or start API server
uvicorn app.api:app --reload
# Then visit http://localhost:8000/docs
```

### 5. Run Experiments

```bash
# Evaluate single model
python scripts/evaluate_models.py

# Compare multiple models
python scripts/run_experiments.py
```

## 📁 Project Structure

```
├── src/                    # Source code
│   ├── config.py          # Configuration
│   ├── data/              # Data loading & preprocessing
│   ├── embeddings/        # CLAP, audio-only, text embedders
│   ├── indexing/          # FAISS vector store
│   ├── retrieval/         # Query parsing & RAG pipeline
│   ├── evaluation/        # Metrics & confusion matrices
│   ├── experiments/       # Experiment scripts
│   └── utils/             # Utilities
├── app/                   # Applications
│   ├── api.py            # FastAPI server
│   ├── chatbot.py        # Interactive chatbot
│   └── ui.py             # UI wrapper
├── scripts/              # Utility scripts
├── notebooks/            # Jupyter notebooks
├── data/                 # Data directory
└── results/              # Evaluation results
```

## 🔧 Configuration

Edit `.env` file:

```bash
# Model Selection
EMBEDDING_MODEL=clap

# Vector Store
VECTOR_STORE_TYPE=faiss
DISTANCE_METRIC=cosine

# Audio Processing
SAMPLE_RATE=44100
AUDIO_DURATION=10

# Retrieval
TOP_K_RESULTS=10
SIMILARITY_THRESHOLD=0.7
```

## 📊 Features

- ✅ Multimodal embeddings (text + audio)
- ✅ FAISS-based similarity search
- ✅ Natural language queries
- ✅ Metadata filtering
- ✅ Confusion matrix evaluation
- ✅ REST API
- ✅ Interactive chatbot

## 🧪 Evaluation

The system generates:
- Confusion matrices comparing models
- Precision@K, Recall@K, MAP, NDCG metrics
- Per-class performance analysis

Results saved to `results/` directory.

## 📝 Example Queries

```python
"give me drum samples"
"piano keys in C major"
"upbeat drum loops around 120 BPM"
"mellow synthesizer sounds"
```

## 🤝 Contributing

Contributions welcome! Please submit pull requests.

## 📄 License

MIT License
