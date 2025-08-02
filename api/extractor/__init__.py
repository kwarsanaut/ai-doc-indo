# api/extractors/__init__.py
"""
Document Extractors Module

Specialized extractors for different document formats (PDF, DOCX, Images).
"""

from .pdf_processor import PDFProcessor, PDFExtractionResult
from .docx_processor import DOCXProcessor, DOCXExtractionResult

# OCR processor will be imported when created
try:
    from .ocr_processor import OCRProcessor, OCRExtractionResult
except ImportError:
    # OCR processor might not be available yet
    OCRProcessor = None
    OCRExtractionResult = None

__all__ = [
    # PDF Processing
    "PDFProcessor",
    "PDFExtractionResult",
    
    # DOCX Processing  
    "DOCXProcessor",
    "DOCXExtractionResult",
    
    # OCR Processing (if available)
    "OCRProcessor",
    "OCRExtractionResult"
]

# Extractor registry for dynamic loading
EXTRACTOR_REGISTRY = {
    "pdf": PDFProcessor,
    "docx": DOCXProcessor,
}

# Add OCR processor if available
if OCRProcessor is not None:
    EXTRACTOR_REGISTRY["image"] = OCRProcessor

def get_extractor(document_type: str):
    """
    Get appropriate extractor for document type.
    
    Args:
        document_type: Type of document ('pdf', 'docx', 'image')
        
    Returns:
        Extractor class instance
        
    Raises:
        ValueError: If document type is not supported
    """
    if document_type not in EXTRACTOR_REGISTRY:
        supported_types = list(EXTRACTOR_REGISTRY.keys())
        raise ValueError(f"Unsupported document type: {document_type}. Supported types: {supported_types}")
    
    return EXTRACTOR_REGISTRY[document_type]()

def get_supported_formats():
    """
    Get all supported file formats across all extractors.
    
    Returns:
        Dict mapping extractor type to supported file extensions
    """
    formats = {}
    
    for extractor_type, extractor_class in EXTRACTOR_REGISTRY.items():
        try:
            extractor = extractor_class()
            if hasattr(extractor, 'supported_formats'):
                formats[extractor_type] = extractor.supported_formats
        except Exception:
            # Skip if extractor cannot be instantiated
            continue
    
    return formats

def is_supported_file(filename: str):
    """
    Check if a file is supported by any extractor.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        Tuple of (is_supported: bool, extractor_type: str or None)
    """
    filename_lower = filename.lower()
    
    for extractor_type, extractor_class in EXTRACTOR_REGISTRY.items():
        try:
            extractor = extractor_class()
            if hasattr(extractor, 'is_supported_format') and extractor.is_supported_format(filename):
                return True, extractor_type
        except Exception:
            continue
    
    return False, None

# Supported file extensions for quick reference
SUPPORTED_EXTENSIONS = {
    '.pdf': 'pdf',
    '.docx': 'docx', 
    '.doc': 'docx',
    '.png': 'image',
    '.jpg': 'image',
    '.jpeg': 'image',
    '.tiff': 'image',
    '.bmp': 'image'
}

# Version info
__version__ = "1.0.0"
