# api/processors/__init__.py
"""
Document Processing Module

Core processors for document ingestion, AI extraction, and vector database operations.
"""

from .document_ingestion import DocumentProcessor, ProcessedDocument, DocumentType
from .ai_extraction import (
    AIExtractionEngine, 
    DocumentClassification,
    DocumentEntities,
    DocumentSummary,
    ExtractedEntity,
    EntityType,
    init_extraction_engine
)
from .vector_database import (
    EmbeddingManager,
    PineconeVectorStore,
    ChromaDBVectorStore,
    DocumentChunk,
    SearchResult,
    initialize_vector_stores
)

# Import production API components
try:
    from .production_api import (
        get_current_user,
        User,
        DocumentProcessingRequest,
        DocumentProcessingResponse,
        ProcessingJob,
        RateLimiter,
        rate_limiter,
        process_document_background,
        create_access_token,
        REQUEST_COUNT,
        REQUEST_DURATION,
        DOCUMENT_PROCESSING_COUNT,
        DOCUMENT_PROCESSING_DURATION,
        security
    )
except ImportError:
    # Production API components might not be available in all contexts
    pass

__all__ = [
    # Document Processing
    "DocumentProcessor",
    "ProcessedDocument", 
    "DocumentType",
    
    # AI Extraction
    "AIExtractionEngine",
    "DocumentClassification",
    "DocumentEntities", 
    "DocumentSummary",
    "ExtractedEntity",
    "EntityType",
    "init_extraction_engine",
    
    # Vector Database
    "EmbeddingManager",
    "PineconeVectorStore",
    "ChromaDBVectorStore", 
    "DocumentChunk",
    "SearchResult",
    "initialize_vector_stores",
    
    # Production API (if available)
    "get_current_user",
    "User",
    "DocumentProcessingRequest",
    "DocumentProcessingResponse",
    "ProcessingJob",
    "RateLimiter",
    "rate_limiter",
    "process_document_background",
    "create_access_token",
    "REQUEST_COUNT",
    "REQUEST_DURATION", 
    "DOCUMENT_PROCESSING_COUNT",
    "DOCUMENT_PROCESSING_DURATION",
    "security"
]

# Version info
__version__ = "1.0.0"
