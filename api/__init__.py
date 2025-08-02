# api/__init__.py
"""
AI Document Processor API Package

Enterprise-grade AI document processing system optimized for Indonesian businesses.
"""

__version__ = "1.0.0"
__author__ = "AI Document Processor Team"
__email__ = "support@yourdomain.com"
__description__ = "AI-powered document processing with Indonesian language optimization"

# Package metadata
PACKAGE_INFO = {
    "name": "ai-doc-processor",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "email": __email__,
    "url": "https://github.com/yourusername/ai-doc-indo",
    "license": "MIT",
    "python_requires": ">=3.11",
    "keywords": ["ai", "document-processing", "indonesia", "ocr", "nlp"],
    "classifiers": [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Office/Business :: Office Suites",
        "Topic :: Text Processing :: General",
    ]
}

# Supported file formats
SUPPORTED_FORMATS = {
    "pdf": [".pdf"],
    "docx": [".docx", ".doc"], 
    "images": [".png", ".jpg", ".jpeg", ".tiff", ".bmp"]
}

# Indonesian language support
INDONESIAN_FEATURES = {
    "ocr_languages": ["id", "en"],
    "nlp_models": ["indobert", "multilingual"],
    "business_patterns": ["ktp", "npwp", "phone", "address", "currency"],
    "document_types": ["faktur_pajak", "kontrak_kerja", "surat_kuasa"]
}

# API configuration
API_CONFIG = {
    "title": "AI Document Processor",
    "description": "Enterprise-grade AI document processing for Indonesian businesses",
    "version": __version__,
    "docs_url": "/docs",
    "redoc_url": "/redoc",
    "openapi_url": "/openapi.json"
}

# Default settings
DEFAULT_SETTINGS = {
    "max_file_size": 50 * 1024 * 1024,  # 50MB
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "confidence_threshold": 0.8,
    "timeout_seconds": 300,
    "max_concurrent_jobs": 10
}

# Export main components
from api.processors import *
from api.extractors import *
from api.vector_stores import *

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    "PACKAGE_INFO",
    "SUPPORTED_FORMATS",
    "INDONESIAN_FEATURES",
    "API_CONFIG",
    "DEFAULT_SETTINGS"
]
