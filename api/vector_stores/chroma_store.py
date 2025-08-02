# api/vector_stores/chroma_store.py
import asyncio
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import json

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from . import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

class ChromaDBVectorStore:
    """Local/development vector store using ChromaDB"""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "documents"):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize ChromaDB
        self._initialize_chromadb()
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create client with persistence
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={
                    "description": "AI Document Processor vector store",
                    "created_at": datetime.now().isoformat()
                }
            )
            
            logger.info(f"Initialized ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def store_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Store document chunks in ChromaDB"""
        try:
            # Create document chunks
            chunks = self._create_chunks(content, document_id, metadata)
            
            if not chunks:
                logger.warning(f"No chunks created for document {document_id}")
                return []
            
            # Prepare data for ChromaDB
            chunk_ids = []
            documents = []
            embeddings = []
            metadatas = []
            
            for chunk in chunks:
                chunk_ids.append(chunk.chunk_id)
                documents.append(chunk.content)
                
                # Generate embedding
                embedding = self.embedding_model.encode([chunk.content], convert_to_tensor=False)[0]
                embeddings.append(embedding.tolist())
                
                # Prepare metadata (ChromaDB requires JSON-serializable values)
                chunk_metadata = {
                    "document_id": document_id,
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "total_chunks": chunk.metadata.get("total_chunks", 1),
                    "chunk_size": len(chunk.content),
                    "chunk_type": chunk.metadata.get("chunk_type", "content"),
                    "document_type": metadata.get("doc_type", "unknown"),
                    "category": metadata.get("category", "general"),
                    "language": metadata.get("language", "id"),
                    "user_id": metadata.get("user_id", "anonymous"),
                    "timestamp": datetime.now().isoformat(),
                    "created_at": chunk.metadata.get("created_at", datetime.now().isoformat())
                }
                
                # Add additional metadata fields (flattened)
                for key, value in metadata.items():
                    if key not in chunk_metadata and isinstance(value, (str, int, float, bool)):
                        chunk_metadata[key] = value
                
                metadatas.append(chunk_metadata)
            
            # Add to ChromaDB collection
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=chunk_ids
            )
            
            logger.info(f"Stored {len(chunk_ids)} chunks for document {document_id}")
            return chunk_ids
            
        except Exception as e:
            logger.error(f"Failed to store document {document_id}: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 10, filters: Dict[str, Any] = None) -> List[SearchResult]:
        """Search documents using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_tensor=False)[0]
            
            # Prepare ChromaDB where clause
            where_clause = self._prepare_where_clause(filters)
            
            # Search in ChromaDB
            search_results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                include=["documents", "metadatas", "distances"],
                where=where_clause
            )
            
            # Convert to SearchResult objects
            results = []
            if search_results["documents"] and search_results["documents"][0]:
                for i in range(len(search_results["documents"][0])):
                    # Convert distance to similarity score (ChromaDB returns distances)
                    distance = search_results["distances"][0][i]
                    similarity_score = 1 / (1 + distance)  # Convert distance to similarity
                    
                    result = SearchResult(
                        document_id=search_results["metadatas"][0][i]["document_id"],
                        chunk_id=search_results["ids"][0][i] if "ids" in search_results else f"chunk_{i}",
                        content=search_results["documents"][0][i],
                        score=similarity_score,
                        metadata=search_results["metadatas"][0][i],
                        highlights=self._extract_highlights(query, search_results["documents"][0][i])
                    )
                    results.append(result)
            
            # Re-rank results
            results = self._rerank_results(query, results)
            
            logger.info(f"Found {len(results
