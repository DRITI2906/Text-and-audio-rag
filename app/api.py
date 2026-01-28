"""FastAPI application for audio-text retrieval."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from pathlib import Path

from src.retrieval.rag_pipeline import RAGPipeline
from src.embeddings.clap_embedder import CLAPEmbedder
from src.config import config

# Initialize FastAPI app
app = FastAPI(
    title="Multimodal Audio-Text Retrieval API",
    description="API for querying audio samples using text",
    version="0.1.0"
)

# Global pipeline instance (will be initialized on startup)
pipeline: Optional[RAGPipeline] = None


class QueryRequest(BaseModel):
    """Request model for queries."""
    query: str
    k: int = 10
    use_filters: bool = True


class QueryResponse(BaseModel):
    """Response model for queries."""
    results: List[Dict]
    query: str
    num_results: int


@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup."""
    global pipeline
    
    # Initialize embedder
    embedder = CLAPEmbedder()
    
    # Initialize pipeline and load index
    pipeline = RAGPipeline(embedder)
    
    # Try to load existing index
    if config.FAISS_INDEX_PATH.exists():
        pipeline.load_index()
        print(f"Loaded index with {pipeline.vector_store.get_vector_count()} samples")
    else:
        print("No existing index found. Please build the index first.")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Multimodal Audio-Text Retrieval API",
        "version": "0.1.0",
        "endpoints": {
            "/query": "POST - Query for audio samples",
            "/stats": "GET - Get index statistics",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return {
        "status": "healthy",
        "index_loaded": pipeline.vector_store.get_vector_count() > 0
    }


@app.get("/stats")
async def get_stats():
    """Get index statistics."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    return pipeline.get_stats()


@app.post("/query", response_model=QueryResponse)
async def query_samples(request: QueryRequest):
    """
    Query for audio samples.
    
    Args:
        request: Query request with text query and parameters
        
    Returns:
        Query response with results
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        results = pipeline.query(
            request.query,
            k=request.k,
            return_paths=True
        )
        
        return QueryResponse(
            results=results,
            query=request.query,
            num_results=len(results)
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )
