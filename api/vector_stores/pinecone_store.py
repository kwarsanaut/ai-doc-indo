# api/vector_stores/pinecone_store.py
import asyncio
import uuid
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import pinecone
from sentence_transformers import SentenceTransformer
import numpy as np

from . import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)

class PineconeVectorStore:
    """Production vector store using Pinecone"""
    
    def __init__(self, api_key: str, environment: str, index_name: str = "document-processor"):
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.index = None
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dimension = 384
        
        # Initialize Pinecone
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection and index"""
        try:
            # Initialize Pinecone
            pinecone.init(api_key=self.api_key, environment=self.environment)
            
            # Create index if it doesn't exist
            existing_indexes = pinecone.list_indexes()
            if self.index_name not in existing_indexes:
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dimension,
                    metric="cosine",
                    metadata_config={
                        "indexed": [
                            "document_type", 
                            "category", 
                            "language", 
                            "timestamp",
                            "user_id",
                            "chunk_type"
                        ]
                    }
                )
                logger.info(f"Created Pinecone index: {self.index_name}")
            
            # Connect to index
            self.index = pinecone.Index(self.index_name)
            logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    async def store_document(self, document_id: str, content: str, metadata: Dict[str, Any]) -> List[str]:
        """Store document chunks in Pinecone"""
        try:
            # Create document chunks
            chunks = self._create_chunks(content, document_id, metadata)
            
            if not chunks:
                logger.warning(f"No chunks created for document {document_id}")
                return []
            
            # Generate embeddings
            chunk_texts = [chunk.content for chunk in chunks]
            embeddings = self.embedding_model.encode(chunk_texts, convert_to_tensor=False)
            
            # Prepare vectors for Pinecone
            vectors = []
            chunk_ids = []
            
            for chunk, embedding in zip(chunks, embeddings):
                vector_data = {
                    "id": chunk.chunk_id,
                    "values": embedding.tolist(),
                    "metadata": {
                        "document_id": document_id,
                        "content": chunk.content[:1000],  # Pinecone metadata limit
                        "full_content": chunk.content,
                        "chunk_index": chunk.metadata.get("chunk_index", 0),
                        "document_type": metadata.get("doc_type", "unknown"),
                        "category": metadata.get("category", "general"),
                        "language": metadata.get("language", "id"),
                        "timestamp": datetime.now().isoformat(),
                        "user_id": metadata.get("user_id", "anonymous"),
                        "chunk_type": chunk.metadata.get("chunk_type", "content"),
                        **self._flatten_metadata(chunk.metadata)
                    }
                }
                
                vectors.append(vector_data)
                chunk_ids.append(chunk.chunk_id)
            
            # Batch upsert to Pinecone
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.debug(f"Upserted batch {i//batch_size + 1} for document {document_id}")
            
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
            
            # Prepare filters for Pinecone
            pinecone_filter = self._prepare_filters(filters)
            
            # Search in Pinecone
            search_results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            # Convert to SearchResult objects
            results = []
            for match in search_results.get("matches", []):
                result = SearchResult(
                    document_id=match["metadata"]["document_id"],
                    chunk_id=match["id"],
                    content=match["metadata"].get("full_content", match["metadata"].get("content", "")),
                    score=match["score"],
                    metadata=match["metadata"],
                    highlights=self._extract_highlights(query, match["metadata"].get("full_content", ""))
                )
                results.append(result)
            
            # Re-rank results
            results = self._rerank_results(query, results)
            
            logger.info(f"Found {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    def _create_chunks(self, content: str, document_id: str, metadata: Dict[str, Any]) -> List[DocumentChunk]:
        """Create document chunks with optimal sizing"""
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        
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
                "chunk_type": self._classify_chunk(text),
                "created_at": datetime.now().isoformat()
            }
            
            chunk = DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
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
    
    def _flatten_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested metadata for Pinecone"""
        flattened = {}
        
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                flattened[key] = value
            elif isinstance(value, list) and len(value) > 0:
                flattened[f"{key}_count"] = len(value)
                if isinstance(value[0], (str, int, float)):
                    flattened[f"{key}_first"] = str(value[0])
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (str, int, float, bool)):
                        flattened[f"{key}_{sub_key}"] = sub_value
        
        return flattened
    
    def _prepare_filters(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Prepare filters for Pinecone query"""
        if not filters:
            return None
        
        pinecone_filter = {}
        
        for key, value in filters.items():
            if key == "document_type" and value:
                pinecone_filter["document_type"] = {"$eq": value}
            elif key == "category" and value:
                pinecone_filter["category"] = {"$eq": value}
            elif key == "language" and value:
                pinecone_filter["language"] = {"$eq": value}
            elif key == "user_id" and value:
                pinecone_filter["user_id"] = {"$eq": value}
            elif key == "date_range" and isinstance(value, dict):
                if "start" in value and "end" in value:
                    pinecone_filter["timestamp"] = {
                        "$gte": value["start"],
                        "$lte": value["end"]
                    }
        
        return pinecone_filter if pinecone_filter else None
    
    def _extract_highlights(self, query: str, content: str) -> List[str]:
        """Extract relevant highlights from content"""
        if not content:
            return []
        
        query_words = query.lower().split()
        sentences = content.split('. ')
        
        highlights = []
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in query_words):
                highlight = sentence.strip()
                if len(highlight) > 150:
                    # Find matching word and create context
                    for word in query_words:
                        if word in sentence_lower:
                            word_pos = sentence_lower.find(word)
                            start = max(0, word_pos - 50)
                            end = min(len(sentence), word_pos + 100)
                            highlight = "..." + sentence[start:end] + "..."
                            break
                
                highlights.append(highlight)
                
                if len(highlights) >= 3:
                    break
        
        return highlights
    
    def _rerank_results(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Re-rank results using additional signals"""
        query_words = set(query.lower().split())
        
        for result in results:
            content_words = set(result.content.lower().split())
            
            # Calculate additional scoring factors
            word_overlap = len(query_words.intersection(content_words)) / len(query_words) if query_words else 0
            content_quality = min(1.0, len(result.content) / 500)
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
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            # Query for all chunks of this document
            query_results = self.index.query(
                vector=[0] * self.embedding_dimension,  # Dummy vector
                filter={"document_id": {"$eq": document_id}},
                top_k=1000,  # Adjust based on max chunks per document
                include_metadata=False
            )
            
            # Extract chunk IDs
            chunk_ids = [match["id"] for match in query_results.get("matches", [])]
            
            if chunk_ids:
                # Delete chunks in batches
                batch_size = 100
                for i in range(0, len(chunk_ids), batch_size):
                    batch = chunk_ids[i:i + batch_size]
                    self.index.delete(ids=batch)
                
                logger.info(f"Deleted {len(chunk_ids)} chunks for document {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get("total_vector_count", 0),
                "index_fullness": stats.get("index_fullness", 0),
                "dimension": stats.get("dimension", 0),
                "namespaces": stats.get("namespaces", {})
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def health_check(self) -> bool:
        """Check if the vector store is healthy"""
        try:
            # Simple query to test connectivity
            self.index.describe_index_stats()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

# Example usage
if __name__ == "__main__":
    import os
    import asyncio
    
    # Initialize store
    api_key = os.getenv("PINECONE_API_KEY")
    environment = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
    
    if api_key:
        store = PineconeVectorStore(api_key, environment)
        
        # Test storage and search
        async def test_store():
            # Store a test document
            doc_id = "test_doc_001"
            content = "Ini adalah dokumen contoh untuk testing sistem AI Indonesia."
            metadata = {"doc_type": "test", "language": "id"}
            
            chunk_ids = await store.store_document(doc_id, content, metadata)
            print(f"Stored document with {len(chunk_ids)} chunks")
            
            # Search
            results = await store.search("dokumen Indonesia", top_k=5)
            print(f"Found {len(results)} search results")
            
            # Statistics
            stats = await store.get_statistics()
            print(f"Index stats: {stats}")
        
        asyncio.run(test_store())
    else:
        print("Please set PINECONE_API_KEY environment variable")
