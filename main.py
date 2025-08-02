# main.py - AI Document Processor FastAPI Application
import os
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

# Import our processors
from api.processors.document_ingestion import DocumentProcessor
from api.processors.ai_extraction import AIExtractionEngine, init_extraction_engine
from api.processors.vector_database import initialize_vector_stores
from api.processors.production_api import (
    get_current_user, User, DocumentProcessingRequest, 
    DocumentProcessingResponse, rate_limiter, process_document_background,
    REQUEST_COUNT, REQUEST_DURATION, DOCUMENT_PROCESSING_COUNT,
    security
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize processors
document_processor = DocumentProcessor()
extraction_engine = None
vector_stores_initialized = False

@asynccontextmanager  
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global extraction_engine, vector_stores_initialized
    
    logger.info("Starting AI Document Processor...")
    
    # Initialize AI extraction engine
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        init_extraction_engine(openai_key)
        logger.info("AI extraction engine initialized")
    else:
        logger.warning("OPENAI_API_KEY not found")
    
    # Initialize vector stores
    pinecone_key = os.getenv("PINECONE_API_KEY")
    pinecone_env = os.getenv("PINECONE_ENVIRONMENT", "us-west1-gcp-free")
    
    if pinecone_key:
        initialize_vector_stores(pinecone_key, pinecone_env)
        vector_stores_initialized = True
        logger.info("Vector stores initialized")
    else:
        logger.warning("PINECONE_API_KEY not found")
    
    # Create directories
    os.makedirs("temp", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    logger.info("AI Document Processor started successfully!")
    
    yield
    
    # Cleanup
    logger.info("Shutting down AI Document Processor...")

# FastAPI app with lifespan
app = FastAPI(
    title="AI Document Processor",
    description="Enterprise-grade AI document processing for Indonesian businesses",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure for production
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    REQUEST_DURATION.observe(duration)
    
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    
    return response

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path in ["/health", "/metrics", "/docs", "/redoc"]:
        return await call_next(request)
    
    user_id = request.headers.get("x-user-id", "anonymous")
    
    if not await rate_limiter.is_allowed(user_id, 100):
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"}
        )
    
    return await call_next(request)

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": "2024-01-01T00:00:00Z",
        "services": {
            "api": "healthy",
            "extraction_engine": "healthy" if extraction_engine else "not_configured",
            "vector_stores": "healthy" if vector_stores_initialized else "not_configured"
        },
        "version": "1.0.0"
    }

# Authentication endpoint (simplified)
@app.post("/auth/token")
async def login_for_access_token(email: str, password: str):
    """Get authentication token"""
    from datetime import timedelta
    from api.processors.production_api import create_access_token, ACCESS_TOKEN_EXPIRE_MINUTES
    
    # In production, verify against database
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

# Document processing endpoints
@app.post("/process-document")
async def process_document_endpoint(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """Process document with full AI extraction"""
    try:
        # Process document
        result = await document_processor.process_document(file)
        
        # AI extraction if available
        if extraction_engine:
            ai_result = await extraction_engine.extract_all(result.content, result.filename)
            result_data = {
                "document_info": {
                    "filename": result.filename,
                    "type": result.doc_type.value,
                    "pages": result.pages,
                    "processing_time": result.processing_time,
                    "confidence": result.confidence_score
                },
                "ai_extraction": ai_result
            }
        else:
            result_data = {
                "document_info": {
                    "filename": result.filename,
                    "type": result.doc_type.value,
                    "content": result.content[:1000] + "..." if len(result.content) > 1000 else result.content,
                    "pages": result.pages,
                    "processing_time": result.processing_time,
                    "confidence": result.confidence_score
                },
                "ai_extraction": {"status": "AI extraction not configured"}
            }
        
        DOCUMENT_PROCESSING_COUNT.labels(type=result.doc_type.value, status="success").inc()
        
        return {
            "status": "success",
            "data": result_data
        }
        
    except Exception as e:
        DOCUMENT_PROCESSING_COUNT.labels(type="unknown", status="error").inc()
        logger.error(f"Document processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Vector search endpoint
@app.post("/search")
async def search_documents(
    query: str,
    top_k: int = 10,
    current_user: User = Depends(get_current_user)
):
    """Search documents using semantic search"""
    if not vector_stores_initialized:
        raise HTTPException(
            status_code=503,
            detail="Vector search not available - configure PINECONE_API_KEY"
        )
    
    # Mock search results for now
    mock_results = [
        {
            "document_id": "doc_123",
            "filename": "contract_sample.pdf",
            "content": "Sample contract content matching your query...",
            "score": 0.95,
            "highlights": ["contract terms", "payment schedule"]
        }
    ]
    
    return {
        "query": query,
        "results": mock_results,
        "total": len(mock_results)
    }

# User profile
@app.get("/user/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get user profile and usage stats"""
    return {
        "user": {
            "email": current_user.email,
            "company": current_user.company,
            "plan": current_user.plan
        },
        "usage": {
            "documents_processed": 42,
            "storage_used": "1.2 GB",
            "api_calls": 156
        },
        "limits": {
            "monthly_documents": 1000 if current_user.plan == "basic" else 10000,
            "storage_limit": "10 GB" if current_user.plan == "basic" else "100 GB"
        }
    }

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics"""
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi import Response
    
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)} - {request.url}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
