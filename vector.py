# Vector Database Integration - Pinecone + Embeddings + Semantic Search
# requirements.txt additions: pinecone-client, sentence-transformers, chromadb, faiss-cpu

import asyncio
import uuid
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import json

import pinecone
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import faiss

@dataclass 
class DocumentChunk:
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None

@dataclass
class SearchResult:
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    highlights: List[str]

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        # Multiple embedding options for different use cases
        self.sentence_transformer = SentenceTransformer(model_name)
        self.openai_embeddings = None  # Initialize with API key later
        
        # Indonesian-specific model option
        try:
            self.indonesian_model = SentenceTransformer("indobenchmark/indobert-base-p1")
        except:
            print("Indonesian model not available, using multilingual model")
            self.indonesian_model = self.sentence_transformer
    
    def initialize_openai(self, api_key: str):
        """Initialize OpenAI embeddings"""
        self.openai_embeddings = OpenAIEmbeddings(
            openai_api_key=api_key,
            model="text-embedding-3-large"  # Latest high-quality model
        )
    
    async def generate_embeddings(self, texts: List[str], method: str = "sentence_transformer") -> List[List[float]]:
        """Generate embeddings using specified method"""
        
        if method == "openai" and self.openai_embe
