%SAME%

## 🎯 Project Overview

This system enables seamless retrieval and matching between text queries and audio samples by creating embeddings that represent both modalities in a shared space. The project explores multiple embedding approaches and evaluates their performance on audio samples scraped from Splice.com.

## ✨ Features

- **Multimodal Embeddings**: Text and audio represented in common embedding space
- **Dual Approach**: CLAP-based and audio-only embedding models
- **Natural Language Queries**: Chatbot interface for intuitive sample retrieval
- **Vector Database**: Efficient similarity search and indexing
- **Model Comparison**: Confusion matrices and performance metrics
- **RESTful API**: Easy integration and deployment

## 📊 Dataset

- **Source**: Splice.com
- **Classes**: Keys, Drums
- **Samples**: 20 from each class (40 total)
- **Collection Method**: Browser extension

## 🏗️ Architecture

### Components

1. **Scraper**: Browser extension for Splice.com data collection
2. **Embedding Models**: CLAP and audio-only approaches
3. **Vector Database**: ChromaDB/Pinecone/FAISS for indexing
4. **Retrieval System**: Query processing and similarity search
5. **Chatbot**: Natural language interface
6. **Evaluation**: Model comparison and metrics

### Embedding Approaches

#### Approach 1: CLAP-Based
- Uses CLAP (Contrastive Language-Audio Pretraining)
- Direct text-audio alignment in shared space

#### Approach 2: Audio-Only (Bonus)
- Audio embeddings: PANNs, VGGish, or Wav2Vec2
- Text alignment using sentence transformers

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip or conda
- Chrome/Firefox browser (for scraping)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd Text-and-audio-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python scripts/download_models.py

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Usage

#### 1. Collect Data
```bash
# Install browser extension from scraper/browser_extension/
# Use extension to scrape samples from Splice.com
```

#### 2. Generate Embeddings
```bash
python scripts/process_data.py
python scripts/generate_embeddings.py --model clap
python scripts/generate_embeddings.py --model audio_only  # Bonus approach
```

#### 3. Index Database
```bash
python scripts/index_database.py
```

#### 4. Run Chatbot
```bash
# Start API server
uvicorn api.app:app --reload

# Or use the chatbot directly
python chatbot/bot.py
```

#### 5. Evaluate Models
```bash
python scripts/run_evaluation.py
```

## 📝 Example Queries

```python
# Text queries
"Give me drum samples"
"Find keys in C major"
"Show me upbeat drum loops around 120 BPM"
"Mellow piano samples"
```

## 🧪 Evaluation

The system generates confusion matrices comparing:
- CLAP vs. Audio-only approaches
- Different audio encoders (PANNs, VGGish, etc.)
- Text-to-audio retrieval accuracy

Results are saved in `outputs/results/`

## 📁 Project Structure

```
Text-and-audio-rag/
├── config/              # Configuration files
├── data/                # Raw and processed data
├── scraper/             # Splice.com scraper
├── models/              # Embedding models
├── embeddings/          # Embedding generation
├── database/            # Vector database
├── retrieval/           # Query and retrieval
├── chatbot/             # Chatbot interface
├── evaluation/          # Metrics and evaluation
├── api/                 # REST API
├── notebooks/           # Jupyter notebooks
├── scripts/             # Utility scripts
└── outputs/             # Results and logs
```

## 🛠️ Technology Stack

- **Audio Processing**: librosa, soundfile, pydub
- **ML Models**: transformers, CLAP, sentence-transformers
- **Vector DB**: ChromaDB, Pinecone, FAISS
- **API**: FastAPI, uvicorn
- **Evaluation**: scikit-learn, matplotlib, seaborn

## 📊 Results

Evaluation results and confusion matrices will be available in `outputs/results/` after running experiments.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

## 🙏 Acknowledgments

- Splice.com for audio samples
- LAION for CLAP model
- Open-source audio embedding models
