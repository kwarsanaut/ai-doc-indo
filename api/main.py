# Production FastAPI - Authentication, Rate Limiting, Monitoring, Logging
# requirements.txt additions: fastapi-users, python-jose, passlib, python-multipart, redis, celery, prometheus-client

import asyncio
import time
import logging
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import redis
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from jose import JWTError, jwt
from passlib.context import CryptContext
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from celery import Celery
import uvicorn

# Structured logging setup
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
DOCUMENT_PROCESSING_COUNT = Counter('documents_processed_total', 'Total documents processed', ['type', 'status'])
DOCUMENT_PROCESSING_DURATION = Histogram('document_processing_duration_seconds', 'Document processing duration')

# Security
SECRET_KEY = "your-super-secret-key-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Redis for rate limiting and caching
redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)

# Celery for background tasks
celery_app = Celery(
    'document_processor',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Data Models
class User(BaseModel):
    id: str
    email: str
    company: str
    plan: str = "basic"  # basic, premium, enterprise
    rate_limit: int = 100  # requests per hour
    created_at: datetime

class TokenData(BaseModel):
    email: Optional[str] = None

class ProcessingJob(BaseModel):
    job_id: str
    status: str  # pending, processing, completed, failed
    document_id: str
    filename: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class DocumentProcessingRequest(BaseModel):
    extract_entities: bool = True
    extract_tables: bool = True
    generate_summary: bool = True
    store_in_vector_db: bool = True
    language: str = "id"  # Indonesian default
    classification_required: bool = True

class DocumentProcessingResponse(BaseModel):
    job_id: str
    status: str
    estimated_completion: Optional[datetime] = None
    document_info: Dict[str, Any]

# Authentication Functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Extract user from JWT token"""
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except JWTError:
        raise credentials_exception
    
    # In production, fetch user from database
    # For now, create a mock user
    user = User(
        id=str(uuid.uuid4()),
        email=token_data.email,
        company="Enterprise Corp",
        plan="enterprise",
        rate_limit=1000,
        created_at=datetime.utcnow()
    )
    
    return user

# Rate Limiting
class RateLimiter:
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def is_allowed(self, user_id: str, limit: int, window: int = 3600) -> bool:
        """Check if user is within rate limit"""
        key = f"rate_limit:{user_id}"
        current = self.redis.get(key)
        
        if current is None:
            # First request
            self.redis.setex(key, window, 1)
            return True
        
        if int(current) < limit:
            self.redis.incr(key)
            return True
        
        return False

rate_limiter = RateLimiter(redis_client)

# Middleware for monitoring and logging
@asynccontextmanager
async def request_middleware(request: Request):
    start_time = time.time()
    
    # Log request
    logger.info("request_started", 
                method=request.method, 
                url=str(request.url),
                user_agent=request.headers.get("user-agent"))
    
    yield
    
    # Log completion and metrics
    duration = time.time() - start_time
    REQUEST_DURATION.observe(duration)
    
    logger.info("request_completed", 
                method=request.method,
                url=str(request.url), 
                duration=duration)

# Background Tasks
@celery_app.task
def process_document_background(job_id: str, file_path: str, processing_options: Dict[str, Any], user_id: str):
    """Background document processing task"""
    
    try:
        logger.info("background_processing_started", job_id=job_id, user_id=user_id)
        
        # Update job status
        job_key = f"job:{job_id}"
        redis_client.hset(job_key, "status", "processing")
        redis_client.hset(job_key, "started_at", datetime.utcnow().isoformat())
        
        # Simulate processing (replace with actual processing logic)
        time.sleep(10)  # Simulate work
        
        # Mock result
        result = {
            "classification": {"category": "contract", "confidence": 0.95},
            "entities": {"people": ["John Doe"], "amounts": ["Rp 1,000,000"]},
            "summary": "This is a service contract with payment terms.",
            "processing_time": 10.0
        }
        
        # Update job with result
        redis_client.hset(job_key, "status", "completed")
        redis_client.hset(job_key, "completed_at", datetime.utcnow().isoformat())
        redis_client.hset(job_key, "result", str(result))
        
        # Metrics
        DOCUMENT_PROCESSING_COUNT.labels(type="contract", status="success").inc()
        DOCUMENT_PROCESSING_DURATION.observe(10.0)
        
        logger.info("background_processing_completed", job_id=job_id, user_id=user_id)
        
        return result
        
    except Exception as e:
        logger.error("background_processing_failed", job_id=job_id, error=str(e))
        
        # Update job with error
        redis_client.hset(job_key, "status", "failed")
        redis_client.hset(job_key, "error", str(e))
        
        DOCUMENT_PROCESSING_COUNT.labels(type="unknown", status="error").inc()
        
        raise

# FastAPI Application
app = FastAPI(
    title="Intelligent Document Processor API",
    description="Enterprise-grade document processing with AI extraction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["yourdomain.com", "localhost", "127.0.0.1"]
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.observe(duration)
    
    logger.info("request_processed",
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                duration=duration)
    
    return response

# Rate limiting middleware
@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    # Skip rate limiting for health checks and metrics
    if request.url.path in ["/health", "/metrics"]:
        return await call_next(request)
    
    # Extract user (simplified - in production, extract from token)
    user_id = request.headers.get("x-user-id", "anonymous")
    
    if not await rate_limiter.is_allowed(user_id, 100):  # 100 requests per hour
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded"}
        )
    
    return await call_next(request)

# API Endpoints
@app.post("/auth/token")
async def login_for_access_token(email: str, password: str):
    """Authenticate user and return JWT token"""
    # In production, verify credentials against database
    # For demo, accept any credentials
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": email}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
    }

@app.post("/process-document-async", response_model=DocumentProcessingResponse)
async def process_document_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    options: DocumentProcessingRequest = DocumentProcessingRequest(),
    current_user: User = Depends(get_current_user)
):
    """Asynchronous document processing endpoint"""
    
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Generate job ID
    job_id = str(uuid.uuid4())
    
    # Save file temporarily
    file_path = f"temp/{job_id}_{file.filename}"
    
    # Create job record
    job_data = {
        "job_id": job_id,
        "status": "pending",
        "document_id": str(uuid.uuid4()),
        "filename": file.filename,
        "user_id": current_user.id,
        "created_at": datetime.utcnow().isoformat(),
        "options": str(options.dict())
    }
    
    job_key = f"job:{job_id}"
    redis_client.hmset(job_key, job_data)
    redis_client.expire(job_key, 3600)  # Expire after 1 hour
    
    # Queue background processing
    process_document_background.delay(
        job_id=job_id,
        file_path=file_path,
        processing_options=options.dict(),
        user_id=current_user.id
    )
    
    logger.info("document_processing_queued", 
                job_id=job_id, 
                filename=file.filename,
                user_id=current_user.id)
    
    return DocumentProcessingResponse(
        job_id=job_id,
        status="pending",
        estimated_completion=datetime.utcnow() + timedelta(minutes=5),
        document_info={
            "filename": file.filename,
            "size": file.size if hasattr(file, 'size') else 0,
            "content_type": file.content_type
        }
    )

@app.get("/job/{job_id}")
async def get_job_status(job_id: str, current_user: User = Depends(get_current_user)):
    """Get processing job status and results"""
    
    job_key = f"job:{job_id}"
    job_data = redis_client.hgetall(job_key)
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Verify job belongs to user
    if job_data.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    response_data = {
        "job_id": job_id,
        "status": job_data.get("status", "unknown"),
        "filename": job_data.get("filename"),
        "created_at": job_data.get("created_at"),
        "started_at": job_data.get("started_at"),
        "completed_at": job_data.get("completed_at"),
        "error": job_data.get("error")
    }
    
    # Include result if completed
    if job_data.get("status") == "completed" and job_data.get("result"):
        try:
            response_data["result"] = eval(job_data["result"])  # In production, use proper JSON parsing
        except:
            response_data["result"] = {"error": "Failed to parse result"}
    
    return response_data

@app.post("/process-document-sync")
async def process_document_sync(
    file: UploadFile = File(...),
    options: DocumentProcessingRequest = DocumentProcessingRequest(),
    current_user: User = Depends(get_current_user)
):
    """Synchronous document processing for small files"""
    
    start_time = time.time()
    
    try:
        # File validation
        if file.size and file.size > 10 * 1024 * 1024:  # 10MB limit for sync processing
            raise HTTPException(
                status_code=413, 
                detail="File too large for synchronous processing. Use async endpoint."
            )
        
        # Read file content
        content = await file.read()
        
        # Here you would integrate with your document processor
        # For demo, return mock result
        processing_time = time.time() - start_time
        
        result = {
            "status": "completed",
            "filename": file.filename,
            "processing_time": round(processing_time, 2),
            "classification": {
                "category": "invoice",
                "confidence": 0.92,
                "subcategory": "tax_invoice"
            },
            "entities": {
                "people": ["PT. Example Company"],
                "amounts": ["Rp 5,500,000"],
                "dates": ["2024-01-15"],
                "locations": ["Jakarta"]
            },
            "summary": {
                "executive_summary": "Tax invoice for professional services rendered in January 2024.",
                "key_points": [
                    "Service period: January 1-31, 2024",
                    "Total amount: Rp 5,500,000",
                    "Payment due: February 15, 2024"
                ],
                "urgency_level": "MEDIUM"
            }
        }
        
        # Log metrics
        DOCUMENT_PROCESSING_COUNT.labels(type="invoice", status="success").inc()
        DOCUMENT_PROCESSING_DURATION.observe(processing_time)
        
        logger.info("sync_processing_completed",
                    filename=file.filename,
                    user_id=current_user.id,
                    processing_time=processing_time)
        
        return result
        
    except Exception as e:
        processing_time = time.time() - start_time
        
        DOCUMENT_PROCESSING_COUNT.labels(type="unknown", status="error").inc()
        
        logger.error("sync_processing_failed",
                     filename=file.filename,
                     user_id=current_user.id,
                     error=str(e),
                     processing_time=processing_time)
        
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/user/profile")
async def get_user_profile(current_user: User = Depends(get_current_user)):
    """Get current user profile and usage statistics"""
    
    # Get usage stats from Redis
    usage_key = f"usage:{current_user.id}"
    today = datetime.utcnow().strftime("%Y-%m-%d")
    today_usage = redis_client.get(f"{usage_key}:{today}") or 0
    
    # Get monthly usage
    month = datetime.utcnow().strftime("%Y-%m")
    monthly_usage = redis_client.get(f"{usage_key}:month:{month}") or 0
    
    return {
        "user": current_user,
        "usage": {
            "today": int(today_usage),
            "monthly": int(monthly_usage),
            "rate_limit": current_user.rate_limit,
            "plan": current_user.plan
        },
        "features": {
            "async_processing": True,
            "vector_search": current_user.plan in ["premium", "enterprise"],
            "batch_processing": current_user.plan == "enterprise",
            "custom_models": current_user.plan == "enterprise"
        }
    }

@app.post("/search/documents")
async def search_documents(
    query: str,
    top_k: int = 10,
    filters: Dict[str, Any] = None,
    current_user: User = Depends(get_current_user)
):
    """Search through processed documents using semantic search"""
    
    if current_user.plan not in ["premium", "enterprise"]:
        raise HTTPException(
            status_code=403, 
            detail="Semantic search requires premium or enterprise plan"
        )
    
    # Here you would integrate with your vector database
    # For demo, return mock search results
    mock_results = [
        {
            "document_id": str(uuid.uuid4()),
            "filename": "contract_2024_001.pdf",
            "content_preview": "This service agreement is entered into between...",
            "score": 0.95,
            "highlights": ["service agreement", "payment terms", "deliverables"],
            "metadata": {
                "category": "contract",
                "date_processed": "2024-01-15T10:30:00Z",
                "pages": 5
            }
        },
        {
            "document_id": str(uuid.uuid4()),
            "filename": "invoice_jan_2024.pdf", 
            "content_preview": "Invoice for professional services rendered...",
            "score": 0.87,
            "highlights": ["invoice", "professional services", "payment due"],
            "metadata": {
                "category": "invoice",
                "date_processed": "2024-01-20T14:15:00Z",
                "pages": 2
            }
        }
    ]
    
    logger.info("semantic_search_performed",
                query=query,
                user_id=current_user.id,
                results_count=len(mock_results))
    
    return {
        "query": query,
        "results": mock_results,
        "total_results": len(mock_results),
        "search_time": 0.15
    }

# Health and Monitoring Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    # Check Redis connectivity
    try:
        redis_client.ping()
        redis_status = "healthy"
    except:
        redis_status = "unhealthy"
    
    # Check Celery connectivity
    try:
        celery_inspect = celery_app.control.inspect()
        active_tasks = celery_inspect.active()
        celery_status = "healthy" if active_tasks is not None else "unhealthy"
    except:
        celery_status = "unhealthy"
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "api": "healthy",
            "redis": redis_status,
            "celery": celery_status
        },
        "version": "1.0.0"
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    from fastapi import Response
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/admin/stats")
async def get_admin_stats(current_user: User = Depends(get_current_user)):
    """Admin statistics (enterprise users only)"""
    
    if current_user.plan != "enterprise":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Get system statistics
    stats = {
        "total_documents_processed": 1543,
        "total_users": 87,
        "active_jobs": 12,
        "avg_processing_time": 8.3,
        "success_rate": 0.967,
        "popular_document_types": {
            "invoices": 45,
            "contracts": 32,
            "reports": 23
        },
        "usage_by_plan": {
            "basic": 45,
            "premium": 32,
            "enterprise": 10
        }
    }
    
    return stats

# Error Handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    
    logger.error("http_exception",
                 url=str(request.url),
                 method=request.method,
                 status_code=exc.status_code,
                 detail=exc.detail)
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    
    logger.error("unhandled_exception",
                 url=str(request.url),
                 method=request.method,
                 error=str(exc),
                 exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Startup/Shutdown Events
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    
    logger.info("application_starting")
    
    # Initialize services
    try:
        redis_client.ping()
        logger.info("redis_connected")
    except Exception as e:
        logger.error("redis_connection_failed", error=str(e))
    
    # Create necessary directories
    import os
    os.makedirs("temp", exist_ok=True)
    os.makedirs("uploads", exist_ok=True)
    
    logger.info("application_started")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    
    logger.info("application_shutting_down")
    
    # Cleanup tasks
    # Close Redis connections, cancel background tasks, etc.
    
    logger.info("application_shutdown_complete")

# Production Server Configuration
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        workers=4,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )
