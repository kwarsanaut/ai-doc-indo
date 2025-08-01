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
        
        if method == "openai" and self.openai_embeddings:
            # Use OpenAI embeddings for best quality
            return await self.openai_embeddings.aembed_documents(texts)
        
        elif method == "indonesian":
            # Use Indonesian-specific model
            return self.indonesian_model.encode(texts, convert_to_tensor=False).tolist()
        
        else:
            # Default: sentence transformer
            return self.sentence_transformer.encode(texts, convert_to_tensor=False).tolist()
    
    def generate_embedding_sync(self, text: str, method: str = "sentence_transformer") -> List[float]:
        """Synchronous embedding generation"""
        if method == "indonesian":
            return self.indonesian_model.encode([text], convert_to_tensor=False)[0].tolist()
        else:
            return self.sentence_transformer.encode([text], convert_to_tensor=False)[0].tolist()

class PineconeVectorStore:
    def __init__(self, api_key: str, environment: str, index_name: str = "document-processor"):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.index = None
        self.embedding_manager = EmbeddingManager()
        
        # Initialize Pinecone
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create or connect to index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize Pinecone index"""
        dimension = 384  # Dimension for all-MiniLM-L6-v2
        
        # Create index if it doesn't exist
        existing_indexes = pinecone.list_indexes()
        if self.index_name not in existing_indexes:
            pinecone.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                metadata_config={
                    "indexed": ["document_type", "category", "language", "timestamp"]
                }
            )
        
        self.index = pinecone.Index(self.index_name)
    
    async def store_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Store document with chunking and embeddings"""
        
        # Chunk the document
        chunks = self._chunk_document(content, metadata)
        
        # Generate embeddings for all chunks
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = await self.embedding_manager.generate_embeddings(chunk_texts)
        
        # Prepare vectors for Pinecone
        vectors = []
        chunk_ids = []
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
            
            vector_data = {
                "id": chunk.chunk_id,
                "values": embedding,
                "metadata": {
                    "document_id": document_id,
                    "content": chunk.content[:1000],  # Truncate for metadata
                    "full_content": chunk.content,
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "document_type": metadata.get("doc_type", "unknown"),
                    "category": metadata.get("category", "general"),
                    "timestamp": datetime.now().isoformat(),
                    "language": metadata.get("language", "id"),
                    **chunk.metadata
                }
            }
            
            vectors.append(vector_data)
            chunk_ids.append(chunk.chunk_id)
        
        # Batch upsert to Pinecone
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            batch = vectors[i:i + batch_size]
            self.index.upsert(vectors=batch)
        
        return chunk_ids
    
    def _chunk_document(self, content: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Intelligent document chunking"""
        
        # Adaptive chunking based on document type
        doc_type = metadata.get("doc_type", "general")
        
        if doc_type == "pdf":
            chunk_size = 1000
            overlap = 200
        elif doc_type == "contract":
            chunk_size = 800
            overlap = 150
        else:
            chunk_size = 600
            overlap = 100
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        texts = text_splitter.split_text(content)
        chunks = []
        
        for i, text in enumerate(texts):
            chunk_id = str(uuid.uuid4())
            chunk_metadata = {
                "chunk_index": i,
                "total_chunks": len(texts),
                "chunk_size": len(text),
                "chunk_type": self._classify_chunk(text)
            }
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=metadata.get("document_id", "unknown"),
                content=text,
                metadata=chunk_metadata
            )
            chunks.append(chunk)
        
        return chunks
    
    def _classify_chunk(self, text: str) -> str:
        """Classify chunk content type"""
        text_lower = text.lower()
        
        if any(keyword in text_lower for keyword in ["tabel", "table", "|", "jumlah", "total"]):
            return "table"
        elif any(keyword in text_lower for keyword in ["kesimpulan", "summary", "ringkasan"]):
            return "summary"
        elif any(keyword in text_lower for keyword in ["tanggal", "date", "alamat", "address"]):
            return "metadata"
        elif len(text) > 500:
            return "main_content"
        else:
            return "fragment"
    
    async def semantic_search(self, query: str, top_k: int = 10, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Advanced semantic search with filtering"""
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embedding_sync(query)
        
        # Prepare filter
        pinecone_filter = {}
        if filters:
            for key, value in filters.items():
                if key in ["document_type", "category", "language"]:
                    pinecone_filter[key] = {"$eq": value}
                elif key == "date_range":
                    pinecone_filter["timestamp"] = {
                        "$gte": value.get("start"),
                        "$lte": value.get("end")
                    }
        
        # Search in Pinecone
        search_results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=pinecone_filter if pinecone_filter else None
        )
        
        # Process and rank results
        results = []
        for match in search_results["matches"]:
            result = SearchResult(
                document_id=match["metadata"]["document_id"],
                chunk_id=match["id"],
                content=match["metadata"]["full_content"],
                score=match["score"],
                metadata=match["metadata"],
                highlights=self._extract_highlights(query, match["metadata"]["full_content"])
            )
            results.append(result)
        
        # Re-rank results using advanced scoring
        return self._rerank_results(query, results)
    
    def _extract_highlights(self, query: str, content: str) -> List[str]:
        """Extract relevant highlights from content"""
        query_words = query.lower().split()
        content_lower = content.lower()
        
        highlights = []
        sentences = content.split('. ')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words):
                # Extract context around matching words
                highlight = sentence.strip()
                if len(highlight) > 150:
                    # Find the matching word and create context
                    for word in query_words:
                        if word in sentence_lower:
                            word_pos = sentence_lower.find(word)
                            start = max(0, word_pos - 50)
                            end = min(len(sentence), word_pos + 100)
                            highlight = "..." + sentence[start:end] + "..."
                            break
                
                highlights.append(highlight)
        
        return highlights[:3]  # Return top 3 highlights
    
    def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Advanced re-ranking using multiple signals"""
        
        query_words = set(query.lower().split())
        
        for result in results:
            content_words = set(result.content.lower().split())
            
            # Calculate additional scoring factors
            word_overlap = len(query_words.intersection(content_words)) / len(query_words)
            content_quality = min(1.0, len(result.content) / 500)  # Prefer substantial content
            chunk_type_bonus = 0.1 if result.metadata.get("chunk_type") == "main_content" else 0
            
            # Combine scores
            combined_score = (
                result.score * 0.7 +  # Semantic similarity
                word_overlap * 0.2 +  # Keyword overlap
                content_quality * 0.05 +  # Content quality
                chunk_type_bonus  # Chunk type preference
            )
            
            result.score = combined_score
        
        return sorted(results, key=lambda x: x.score, reverse=True)

class ChromaDBVectorStore:
    """Alternative vector store using ChromaDB for local development"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "Document processor vector store"}
        )
        self.embedding_manager = EmbeddingManager()
    
    async def store_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Store document in ChromaDB"""
        
        # Chunk document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )
        
        chunks = text_splitter.split_text(content)
        chunk_ids = [f"{document_id}_{i}" for i in range(len(chunks))]
        
        # Generate embeddings
        embeddings = await self.embedding_manager.generate_embeddings(chunks)
        
        # Prepare metadata for each chunk
        chunk_metadata = []
        for i, chunk in enumerate(chunks):
            chunk_meta = {
                "document_id": document_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": datetime.now().isoformat(),
                **metadata
            }
            chunk_metadata.append(chunk_meta)
        
        # Add to ChromaDB
        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=chunk_metadata,
            ids=chunk_ids
        )
        
        return chunk_ids
    
    async def semantic_search(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """Search using ChromaDB"""
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embedding_sync(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Convert to SearchResult objects
        search_results = []
        for i in range(len(results["documents"][0])):
            result = SearchResult(
                document_id=results["metadatas"][0][i]["document_id"],
                chunk_id=results["ids"][0][i],
                content=results["documents"][0][i],
                score=1 - results["distances"][0][i],  # Convert distance to similarity
                metadata=results["metadatas"][0][i],
                highlights=self._extract_highlights(query, results["documents"][0][i])
            )
            search_results.append(result)
        
        return search_results
    
    def _extract_highlights(self, query: str, content: str) -> List[str]:
        """Extract highlights (same logic as Pinecone version)"""
        query_words = query.lower().split()
        sentences = content.split('. ')
        
        highlights = []
        for sentence in sentences:
            if any(word in sentence.lower() for word in query_words):
                highlights.append(sentence.strip())
        
        return highlights[:3]

# FastAPI Integration
from fastapi import HTTPException
from pydantic import BaseModel

class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    filters: Dict[str, Any] = {}
    vector_store: str = "pinecone"  # or "chromadb"

class StoreDocumentRequest(BaseModel):
    document_id: str
    content: str
    metadata: Dict[str, Any]
    vector_store: str = "pinecone"

# Global vector stores
pinecone_store = None
chroma_store = None

def initialize_vector_stores(pinecone_api_key: str, pinecone_env: str):
    """Initialize vector stores"""
    global pinecone_store, chroma_store
    
    try:
        pinecone_store = PineconeVectorStore(pinecone_api_key, pinecone_env)
        print("Pinecone vector store initialized")
    except Exception as e:
        print(f"Pinecone initialization failed: {e}")
    
    try:
        chroma_store = ChromaDBVectorStore()
        print("ChromaDB vector store initialized")
    except Exception as e:
        print(f"ChromaDB initialization failed: {e}")

@app.post("/store-document-vector")
async def store_document_vector(request: StoreDocumentRequest):
    """Store document in vector database"""
    
    store = pinecone_store if request.vector_store == "pinecone" else chroma_store
    
    if not store:
        raise HTTPException(status_code=500, detail=f"{request.vector_store} store not available")
    
    try:
        chunk_ids = await store.store_document(
            request.document_id,
            request.content,
            request.metadata
        )
        
        return {
            "status": "success",
            "document_id": request.document_id,
            "chunks_stored": len(chunk_ids),
            "chunk_ids": chunk_ids,
            "vector_store": request.vector_store
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Storage failed: {str(e)}")

@app.post("/semantic-search")
async def semantic_search_endpoint(request: SearchRequest):
    """Semantic search across stored documents"""
    
    store = pinecone_store if request.vector_store == "pinecone" else chroma_store
    
    if not store:
        raise HTTPException(status_code=500, detail=f"{request.vector_store} store not available")
    
    try:
        results = await store.semantic_search(
            query=request.query,
            top_k=request.top_k,
            filters=request.filters if hasattr(store, 'semantic_search') and 'filters' in store.semantic_search.__code__.co_varnames else None
        )
        
        return {
            "status": "success",
            "query": request.query,
            "results": [
                {
                    "document_id": result.document_id,
                    "chunk_id": result.chunk_id,
                    "content": result.content[:500] + "..." if len(result.content) > 500 else result.content,
                    "score": result.score,
                    "highlights": result.highlights,
                    "metadata": result.metadata
                }
                for result in results
            ],
            "total_results": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

# Usage example
if __name__ == "__main__":
    import os
    
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
    
    if pinecone_key:
        initialize_vector_stores(pinecone_key, pinecone_env)
        print("Vector stores ready for semantic search!")
    else:
        print("Set PINECONE_API_KEY environment variable")
