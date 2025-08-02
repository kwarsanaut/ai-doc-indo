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
            chunk_size = 800
            overlap = 100
        elif doc_type == "contract":
            chunk_size = 600
            overlap = 100
        else:
            chunk_size = 500
            overlap = 50
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        texts = text_splitter.split_text(content)
        chunks = []
        
        for i, text in enumerate(texts):
            chunk_id = f"{document_id}_chunk_{i}_{str(uuid.uuid4())[:8]}"
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
        elif len(text) > 300:
            return "main_content"
        else:
            return "fragment"
    
    def _prepare_where_clause(self, filters: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Prepare where clause for ChromaDB query"""
        if not filters:
            return None
        
        where_clause = {}
        
        for key, value in filters.items():
            if key == "document_type" and value:
                where_clause["document_type"] = {"$eq": value}
            elif key == "category" and value:
                where_clause["category"] = {"$eq": value}
            elif key == "language" and value:
                where_clause["language"] = {"$eq": value}
            elif key == "user_id" and value:
                where_clause["user_id"] = {"$eq": value}
            elif key == "chunk_type" and value:
                where_clause["chunk_type"] = {"$eq": value}
            elif key == "document_id" and value:
                where_clause["document_id"] = {"$eq": value}
        
        return where_clause if where_clause else None
    
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
                if len(highlight) > 100:
                    # Truncate long highlights
                    highlight = highlight[:100] + "..."
                
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
            content_quality = min(1.0, len(result.content) / 300)
            chunk_type_bonus = 0.1 if result.metadata.get("chunk_type") == "main_content" else 0
            
            # Adjust score
            boost = word_overlap * 0.2 + content_quality * 0.1 + chunk_type_bonus
            result.score = min(1.0, result.score + boost)
        
        return sorted(results, key=lambda x: x.score, reverse=True)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all chunks for a document"""
        try:
            # Get all chunks for this document
            results = self.collection.get(
                where={"document_id": {"$eq": document_id}},
                include=["metadatas"]
            )
            
            if results["ids"]:
                # Delete chunks
                self.collection.delete(ids=results["ids"])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics"""
        try:
            count = self.collection.count()
            
            # Get sample of documents to analyze
            sample_results = self.collection.get(
                limit=min(100, count),
                include=["metadatas"]
            )
            
            # Analyze document types
            doc_types = {}
            languages = {}
            categories = {}
            
            for metadata in sample_results.get("metadatas", []):
                # Document types
                doc_type = metadata.get("document_type", "unknown")
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
                
                # Languages
                language = metadata.get("language", "unknown")
                languages[language] = languages.get(language, 0) + 1
                
                # Categories
                category = metadata.get("category", "unknown")
                categories[category] = categories.get(category, 0) + 1
            
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "document_types": doc_types,
                "languages": languages,
                "categories": categories,
                "persist_directory": self.persist_directory
            }
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> bool:
        """Check if the vector store is healthy"""
        try:
            # Simple operation to test connectivity
            self.collection.count()
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def get_documents_by_ids(self, document_ids: List[str]) -> List[SearchResult]:
        """Get documents by their IDs"""
        try:
            results = self.collection.get(
                where={"document_id": {"$in": document_ids}},
                include=["documents", "metadatas"]
            )
            
            search_results = []
            if results["documents"]:
                for i, doc in enumerate(results["documents"]):
                    result = SearchResult(
                        document_id=results["metadatas"][i]["document_id"],
                        chunk_id=results["ids"][i],
                        content=doc,
                        score=1.0,  # Perfect match for direct retrieval
                        metadata=results["metadatas"][i],
                        highlights=[]
                    )
                    search_results.append(result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to get documents by IDs: {e}")
            return []
    
    async def search_similar_documents(self, document_id: str, top_k: int = 5) -> List[SearchResult]:
        """Find documents similar to a given document"""
        try:
            # Get the document chunks
            doc_results = self.collection.get(
                where={"document_id": {"$eq": document_id}},
                include=["embeddings", "documents", "metadatas"],
                limit=1
            )
            
            if not doc_results["embeddings"] or not doc_results["embeddings"][0]:
                return []
            
            # Use the first chunk's embedding to find similar documents
            query_embedding = doc_results["embeddings"][0]
            
            # Search for similar documents (exclude the same document)
            search_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k + 10,  # Get extra to filter out same document
                include=["documents", "metadatas", "distances"],
                where={"document_id": {"$ne": document_id}}  # Exclude same document
            )
            
            # Convert to SearchResult objects
            results = []
            seen_documents = set()
            
            if search_results["documents"] and search_results["documents"][0]:
                for i in range(len(search_results["documents"][0])):
                    doc_id = search_results["metadatas"][0][i]["document_id"]
                    
                    # Only include each document once
                    if doc_id not in seen_documents:
                        seen_documents.add(doc_id)
                        
                        distance = search_results["distances"][0][i]
                        similarity_score = 1 / (1 + distance)
                        
                        result = SearchResult(
                            document_id=doc_id,
                            chunk_id=search_results["ids"][0][i] if "ids" in search_results else f"chunk_{i}",
                            content=search_results["documents"][0][i],
                            score=similarity_score,
                            metadata=search_results["metadatas"][0][i],
                            highlights=[]
                        )
                        results.append(result)
                        
                        if len(results) >= top_k:
                            break
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to find similar documents for {document_id}: {e}")
            return []
    
    def reset_collection(self):
        """Reset the collection (delete all data)"""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={
                    "description": "AI Document Processor vector store",
                    "created_at": datetime.now().isoformat()
                }
            )
            logger.info(f"Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset collection: {e}")
            return False

# Example usage
if __name__ == "__main__":
    import asyncio
    
    # Initialize store
    store = ChromaDBVectorStore()
    
    async def test_store():
        # Test storage
        doc_id = "test_doc_001"
        content = "Ini adalah dokumen contoh untuk testing sistem AI Indonesia. Dokumen ini berisi informasi penting tentang penggunaan teknologi AI dalam pemrosesan dokumen bisnis Indonesia."
        metadata = {"doc_type": "test", "language": "id", "category": "example"}
        
        chunk_ids = await store.store_document(doc_id, content, metadata)
        print(f"Stored document with {len(chunk_ids)} chunks")
        
        # Test search
        results = await store.search("dokumen Indonesia AI", top_k=5)
        print(f"Found {len(results)} search results")
        for result in results:
            print(f"  Score: {result.score:.3f}, Content: {result.content[:100]}...")
        
        # Test statistics
        stats = await store.get_statistics()
        print(f"Collection stats: {stats}")
        
        # Test health check
        health = store.health_check()
        print(f"Health check: {'OK' if health else 'FAILED'}")
        
        # Test similar documents
        similar = await store.search_similar_documents(doc_id, top_k=3)
        print(f"Found {len(similar)} similar documents")
    
    asyncio.run(test_store())
