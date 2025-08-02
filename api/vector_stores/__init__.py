# api/vector_stores/__init__.py
"""
Vector Database Module

Vector storage and semantic search functionality using Pinecone and ChromaDB.
"""

# Import vector store implementations when they're created
try:
    from .pinecone_store import PineconeVectorStore
except ImportError:
    PineconeVectorStore = None

try:
    from .chroma_store import ChromaDBVectorStore  
except ImportError:
    ChromaDBVectorStore = None

# Common data structures and utilities
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum

@dataclass
class DocumentChunk:
    """Document chunk for vector storage"""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    similarity_score: Optional[float] = None

@dataclass 
class SearchResult:
    """Search result from vector database"""
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    highlights: List[str]

class VectorStoreType(Enum):
    """Supported vector store types"""
    PINECONE = "pinecone"
    CHROMADB = "chromadb"
    
class EmbeddingMethod(Enum):
    """Supported embedding methods"""
    SENTENCE_TRANSFORMER = "sentence_transformer"
    OPENAI = "openai"
    INDONESIAN = "indonesian"

__all__ = [
    # Vector Store Classes
    "PineconeVectorStore",
    "ChromaDBVectorStore",
    
    # Data Structures
    "DocumentChunk",
    "SearchResult", 
    "VectorStoreType",
    "EmbeddingMethod",
    
    # Utilities
    "get_vector_store",
    "initialize_all_stores",
    "get_available_stores"
]

# Vector store registry
VECTOR_STORE_REGISTRY = {}

if PineconeVectorStore is not None:
    VECTOR_STORE_REGISTRY[VectorStoreType.PINECONE] = PineconeVectorStore

if ChromaDBVectorStore is not None:
    VECTOR_STORE_REGISTRY[VectorStoreType.CHROMADB] = ChromaDBVectorStore

def get_vector_store(store_type: VectorStoreType, **kwargs):
    """
    Get vector store instance by type.
    
    Args:
        store_type: Type of vector store to create
        **kwargs: Configuration parameters for the store
        
    Returns:
        Vector store instance
        
    Raises:
        ValueError: If store type is not supported or available
    """
    if store_type not in VECTOR_STORE_REGISTRY:
        available_types = list(VECTOR_STORE_REGISTRY.keys())
        raise ValueError(f"Vector store type {store_type} not available. Available: {available_types}")
    
    store_class = VECTOR_STORE_REGISTRY[store_type]
    return store_class(**kwargs)

def get_available_stores() -> List[VectorStoreType]:
    """
    Get list of available vector store types.
    
    Returns:
        List of available vector store types
    """
    return list(VECTOR_STORE_REGISTRY.keys())

def initialize_all_stores(config: Dict[str, Any]) -> Dict[VectorStoreType, Any]:
    """
    Initialize all available vector stores with configuration.
    
    Args:
        config: Configuration dictionary with store-specific settings
        
    Returns:
        Dictionary mapping store types to initialized instances
    """
    initialized_stores = {}
    
    for store_type in VECTOR_STORE_REGISTRY:
        try:
            if store_type == VectorStoreType.PINECONE:
                if "pinecone_api_key" in config and "pinecone_environment" in config:
                    store = get_vector_store(
                        store_type,
                        api_key=config["pinecone_api_key"],
                        environment=config["pinecone_environment"],
                        index_name=config.get("pinecone_index_name", "document-processor")
                    )
                    initialized_stores[store_type] = store
            
            elif store_type == VectorStoreType.CHROMADB:
                store = get_vector_store(
                    store_type,
                    persist_directory=config.get("chroma_persist_directory", "./chroma_db")
                )
                initialized_stores[store_type] = store
                
        except Exception as e:
            print(f"Failed to initialize {store_type}: {e}")
            continue
    
    return initialized_stores

def get_default_store(initialized_stores: Dict[VectorStoreType, Any]):
    """
    Get the default/preferred vector store from initialized stores.
    
    Args:
        initialized_stores: Dictionary of initialized vector stores
        
    Returns:
        Default vector store instance or None
    """
    # Prefer Pinecone for production, ChromaDB for development
    if VectorStoreType.PINECONE in initialized_stores:
        return initialized_stores[VectorStoreType.PINECONE]
    elif VectorStoreType.CHROMADB in initialized_stores:
        return initialized_stores[VectorStoreType.CHROMADB]
    else:
        return None

# Configuration defaults
DEFAULT_CONFIG = {
    "chunk_size": 800,
    "chunk_overlap": 100,
    "embedding_model": "all-MiniLM-L6-v2",
    "similarity_threshold": 0.7,
    "max_results": 50
}

# Version info
__version__ = "1.0.0"
