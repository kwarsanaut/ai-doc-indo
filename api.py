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
                filename=file
