# api/extractors/pdf_processor.py
import os
import fitz  # PyMuPDF
import pdfplumber
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PDFExtractionResult:
    text_content: str
    metadata: Dict[str, Any]
    tables: List[List[List[str]]]
    images_found: int
    confidence_score: float
    processing_time: float

class PDFProcessor:
    """Advanced PDF processing with multiple extraction methods"""
    
    def __init__(self):
        self.supported_formats = ['.pdf']
    
    def extract_content(self, file_path: str) -> PDFExtractionResult:
        """Extract all content from PDF using multiple methods"""
        import time
        start_time = time.time()
        
        try:
            # Method 1: PyMuPDF for text and metadata
            pymupdf_result = self._extract_with_pymupdf(file_path)
            
            # Method 2: pdfplumber for tables and structured data
            pdfplumber_result = self._extract_with_pdfplumber(file_path)
            
            # Combine results
            combined_text = self._combine_text_results(
                pymupdf_result['text'], 
                pdfplumber_result['text']
            )
            
            # Merge metadata
            metadata = {**pymupdf_result['metadata'], **pdfplumber_result['metadata']}
            
            # Calculate confidence based on extraction success
            confidence = self._calculate_confidence(
                pymupdf_result, pdfplumber_result
            )
            
            processing_time = time.time() - start_time
            
            return PDFExtractionResult(
                text_content=combined_text,
                metadata=metadata,
                tables=pdfplumber_result['tables'],
                images_found=pymupdf_result['images_count'],
                confidence_score=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return PDFExtractionResult(
                text_content="",
                metadata={"error": str(e)},
                tables=[],
                images_found=0,
                confidence_score=0.0,
                processing_time=processing_time
            )
    
    def _extract_with_pymupdf(self, file_path: str) -> Dict[str, Any]:
        """Extract using PyMuPDF for text and metadata"""
        content_parts = []
        images_count = 0
        
        with fitz.open(file_path) as doc:
            # Extract metadata
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'pages': doc.page_count,
                'encrypted': doc.is_encrypted,
                'pdf_version': doc.pdf_version()
            }
            
            # Extract text from each page
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                
                # Get text
                text = page.get_text()
                if text.strip():
                    content_parts.append(f"[Page {page_num + 1}]\n{text}")
                
                # Count images
                image_list = page.get_images()
                images_count += len(image_list)
                
                # Extract text from images if no direct text
                if not text.strip() and image_list:
                    # This would require OCR - handled in main processor
                    content_parts.append(f"[Page {page_num + 1} - Contains {len(image_list)} images]")
        
        return {
            'text': '\n\n'.join(content_parts),
            'metadata': metadata,
            'images_count': images_count
        }
    
    def _extract_with_pdfplumber(self, file_path: str) -> Dict[str, Any]:
        """Extract using pdfplumber for tables and structured data"""
        content_parts = []
        all_tables = []
        
        with pdfplumber.open(file_path) as pdf:
            metadata = {
                'pages_pdfplumber': len(pdf.pages),
                'pdf_info': pdf.metadata if hasattr(pdf, 'metadata') else {}
            }
            
            for page_num, page in enumerate(pdf.pages):
                # Extract text
                text = page.extract_text()
                if text and text.strip():
                    content_parts.append(f"[Page {page_num + 1} - pdfplumber]\n{text}")
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    all_tables.extend(tables)
                    
                    # Add table descriptions to content
                    for i, table in enumerate(tables):
                        if table and len(table) > 0:
                            content_parts.append(
                                f"[Page {page_num + 1} - Table {i + 1}]\n" +
                                self._format_table_for_text(table)
                            )
        
        return {
            'text': '\n\n'.join(content_parts),
            'metadata': metadata,
            'tables': all_tables
        }
    
    def _combine_text_results(self, pymupdf_text: str, pdfplumber_text: str) -> str:
        """Intelligently combine text from both extractors"""
        
        # If one extractor got significantly more text, prefer it
        pymupdf_length = len(pymupdf_text.strip())
        pdfplumber_length = len(pdfplumber_text.strip())
        
        if pymupdf_length == 0 and pdfplumber_length == 0:
            return ""
        
        if pymupdf_length == 0:
            return pdfplumber_text
        
        if pdfplumber_length == 0:
            return pymupdf_text
        
        # If lengths are similar, prefer PyMuPDF (usually better formatting)
        if abs(pymupdf_length - pdfplumber_length) / max(pymupdf_length, pdfplumber_length) < 0.1:
            return pymupdf_text
        
        # Otherwise, use the longer extraction
        return pymupdf_text if pymupdf_length > pdfplumber_length else pdfplumber_text
    
    def _format_table_for_text(self, table: List[List[str]]) -> str:
        """Format table data as readable text"""
        if not table or len(table) == 0:
            return ""
        
        formatted_rows = []
        for row in table:
            if row:
                # Clean and join cells
                clean_cells = [str(cell).strip() if cell else "" for cell in row]
                formatted_rows.append(" | ".join(clean_cells))
        
        return "\n".join(formatted_rows)
    
    def _calculate_confidence(self, pymupdf_result: Dict, pdfplumber_result: Dict) -> float:
        """Calculate extraction confidence score"""
        confidence_factors = []
        
        # Text extraction success
        pymupdf_text_length = len(pymupdf_result['text'].strip())
        pdfplumber_text_length = len(pdfplumber_result['text'].strip())
        
        if pymupdf_text_length > 0:
            confidence_factors.append(0.9)
        if pdfplumber_text_length > 0:
            confidence_factors.append(0.8)
        
        # Metadata extraction
        if pymupdf_result['metadata'].get('pages', 0) > 0:
            confidence_factors.append(0.7)
        
        # Table extraction
        if len(pdfplumber_result.get('tables', [])) > 0:
            confidence_factors.append(0.8)
        
        # Images found (lower confidence as may need OCR)
        if pymupdf_result.get('images_count', 0) > 0:
            confidence_factors.append(0.6)
        
        # Calculate average confidence
        if confidence_factors:
            return min(1.0, sum(confidence_factors) / len(confidence_factors))
        else:
            return 0.0
    
    def is_supported_format(self, filename: str) -> bool:
        """Check if file format is supported"""
        return any(filename.lower().endswith(fmt) for fmt in self.supported_formats)
    
    def get_pdf_info(self, file_path: str) -> Dict[str, Any]:
        """Get basic PDF information without full extraction"""
        try:
            with fitz.open(file_path) as doc:
                return {
                    'pages': doc.page_count,
                    'encrypted': doc.is_encrypted,
                    'pdf_version': doc.pdf_version(),
                    'title': doc.metadata.get('title', ''),
                    'author': doc.metadata.get('author', ''),
                    'file_size': os.path.getsize(file_path)
                }
        except Exception as e:
            logger.error(f"Failed to get PDF info: {str(e)}")
            return {'error': str(e)}

# Example usage
if __name__ == "__main__":
    processor = PDFProcessor()
    
    # Test with a sample PDF
    sample_file = "sample.pdf"
    if os.path.exists(sample_file):
        result = processor.extract_content(sample_file)
        print(f"Extracted {len(result.text_content)} characters")
        print(f"Found {len(result.tables)} tables")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Processing time: {result.processing_time:.2f}s")
    else:
        print("Sample PDF file not found")
