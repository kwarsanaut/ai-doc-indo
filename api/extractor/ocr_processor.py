# api/extractors/ocr_processor.py
import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import paddleocr
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import logging
import tempfile

logger = logging.getLogger(__name__)

@dataclass
class OCRExtractionResult:
    text_content: str
    confidence_score: float
    bounding_boxes: List[Dict[str, Any]]
    language_detected: str
    image_metadata: Dict[str, Any]
    processing_time: float
    ocr_method: str

class OCRProcessor:
    """Advanced OCR processing with Indonesian language optimization"""
    
    def __init__(self):
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp']
        
        # Initialize PaddleOCR for Indonesian
        try:
            self.paddle_ocr = paddleocr.PaddleOCR(
                use_angle_cls=True,
                lang='id',  # Indonesian
                use_gpu=False,  # Set to True if GPU available
                show_log=False
            )
            self.paddle_available = True
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.warning(f"PaddleOCR initialization failed: {e}")
            self.paddle_available = False
        
        # Configure Tesseract
        self.tesseract_config = '--oem 3 --psm 6 -l ind+eng'
        
        # Indonesian text patterns for validation
        self.indonesian_patterns = {
            'common_words': [
                'dan', 'atau', 'yang', 'ini', 'itu', 'adalah', 'akan', 'pada', 'untuk',
                'dari', 'ke', 'di', 'dengan', 'oleh', 'sebagai', 'tidak', 'sudah'
            ],
            'business_terms': [
                'perusahaan', 'kontrak', 'pembayaran', 'faktur', 'invoice', 'npwp',
                'ktp', 'alamat', 'telepon', 'email', 'tanggal', 'jumlah', 'total'
            ]
        }
    
    def extract_content(self, file_path: str) -> OCRExtractionResult:
        """Extract text from image using multiple OCR methods"""
        import time
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = self._load_and_preprocess_image(file_path)
            image_metadata = self._get_image_metadata(file_path)
            
            # Try multiple OCR methods
            ocr_results = []
            
            # Method 1: PaddleOCR (best for Indonesian)
            if self.paddle_available:
                paddle_result = self._extract_with_paddle(image)
                if paddle_result:
                    ocr_results.append(('PaddleOCR', paddle_result))
            
            # Method 2: Tesseract
            tesseract_result = self._extract_with_tesseract(image)
            if tesseract_result:
                ocr_results.append(('Tesseract', tesseract_result))
            
            # Select best result
            if not ocr_results:
                return self._create_empty_result(start_time, image_metadata)
            
            best_method, best_result = self._select_best_result(ocr_results)
            
            # Post-process text
            processed_text = self._post_process_text(best_result['text'])
            
            # Calculate confidence
            confidence = self._calculate_confidence(best_result, processed_text)
            
            # Detect language
            language = self._detect_language(processed_text)
            
            processing_time = time.time() - start_time
            
            return OCRExtractionResult(
                text_content=processed_text,
                confidence_score=confidence,
                bounding_boxes=best_result.get('boxes', []),
                language_detected=language,
                image_metadata=image_metadata,
                processing_time=processing_time,
                ocr_method=best_method
            )
            
        except Exception as e:
            logger.error(f"OCR processing failed: {str(e)}")
            processing_time = time.time() - start_time
            
            return OCRExtractionResult(
                text_content="",
                confidence_score=0.0,
                bounding_boxes=[],
                language_detected="unknown",
                image_metadata={"error": str(e)},
                processing_time=processing_time,
                ocr_method="none"
            )
    
    def _load_and_preprocess_image(self, file_path: str) -> np.ndarray:
        """Load and preprocess image for better OCR results"""
        
        # Load image
        image = cv2.imread(file_path)
        if image is None:
            # Try with PIL for different formats
            pil_image = Image.open(file_path)
            image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        # Preprocessing pipeline
        processed_image = self._apply_preprocessing(image)
        
        return processed_image
    
    def _apply_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Apply image preprocessing techniques"""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Noise reduction
        denoised = cv2.medianBlur(gray, 3)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Binarization (adaptive threshold)
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((1,1), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def _extract_with_paddle(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract text using PaddleOCR"""
        try:
            # Convert to RGB for PaddleOCR
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Run OCR
            results = self.paddle_ocr.ocr(rgb_image, cls=True)
            
            if not results or not results[0]:
                return None
            
            # Process results
            text_parts = []
            boxes = []
            confidences = []
            
            for line in results[0]:
                if len(line) >= 2:
                    bbox, (text, confidence) = line
                    text_parts.append(text)
                    boxes.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence
                    })
                    confidences.append(confidence)
            
            return {
                'text': '\n'.join(text_parts),
                'boxes': boxes,
                'avg_confidence': np.mean(confidences) if confidences else 0.0,
                'method': 'PaddleOCR'
            }
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return None
    
    def _extract_with_tesseract(self, image: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract text using Tesseract"""
        try:
            # Get text with confidence data
            data = pytesseract.image_to_data(
                image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Filter out low confidence results
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            
            if not confidences:
                return None
            
            # Extract text
            text_parts = []
            boxes = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Confidence threshold
                    text = data['text'][i].strip()
                    if text:
                        text_parts.append(text)
                        boxes.append({
                            'bbox': [
                                data['left'][i], 
                                data['top'][i],
                                data['left'][i] + data['width'][i],
                                data['top'][i] + data['height'][i]
                            ],
                            'text': text,
                            'confidence': data['conf'][i] / 100.0
                        })
            
            return {
                'text': ' '.join(text_parts),
                'boxes': boxes,
                'avg_confidence': np.mean(confidences) / 100.0,
                'method': 'Tesseract'
            }
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return None
    
    def _select_best_result(self, ocr_results: List[Tuple[str, Dict]]) -> Tuple[str, Dict]:
        """Select the best OCR result based on multiple criteria"""
        
        best_score = 0
        best_result = None
        best_method = None
        
        for method, result in ocr_results:
            score = 0
            
            # Base confidence score (40% weight)
            score += result.get('avg_confidence', 0) * 0.4
            
            # Text length score (20% weight)
            text_length = len(result.get('text', ''))
            if text_length > 0:
                score += min(text_length / 1000, 1.0) * 0.2
            
            # Indonesian content score (30% weight)
            indonesian_score = self._calculate_indonesian_score(result.get('text', ''))
            score += indonesian_score * 0.3
            
            # Method preference (10% weight)
            if method == 'PaddleOCR':
                score += 0.1  # Prefer PaddleOCR for Indonesian
            
            if score > best_score:
                best_score = score
                best_result = result
                best_method = method
        
        return best_method, best_result if best_result else ocr_results[0][1]
    
    def _calculate_indonesian_score(self, text: str) -> float:
        """Calculate how likely the text is Indonesian"""
        if not text:
            return 0.0
        
        text_lower = text.lower()
        word_count = len(text_lower.split())
        
        if word_count == 0:
            return 0.0
        
        # Count Indonesian words
        indonesian_words = 0
        
        for word_list in self.indonesian_patterns.values():
            for word in word_list:
                indonesian_words += text_lower.count(word)
        
        # Score based on Indonesian word density
        score = min(indonesian_words / word_count, 1.0)
        
        return score
    
    def _post_process_text(self, text: str) -> str:
        """Post-process extracted text"""
        if not text:
            return ""
        
        # Basic cleaning
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Remove extra spaces
                line = ' '.join(line.split())
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _calculate_confidence(self, ocr_result: Dict, processed_text: str) -> float:
        """Calculate overall confidence score"""
        confidence_factors = []
        
        # OCR confidence
        if 'avg_confidence' in ocr_result:
            confidence_factors.append(ocr_result['avg_confidence'])
        
        # Text quality indicators
        if processed_text:
            # Length factor
            length_factor = min(len(processed_text) / 100, 1.0)
            confidence_factors.append(length_factor * 0.3)
            
            # Indonesian content factor
            indonesian_factor = self._calculate_indonesian_score(processed_text)
            confidence_factors.append(indonesian_factor * 0.2)
        
        return np.mean(confidence_factors) if confidence_factors else 0.0
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of extracted text"""
        if not text:
            return "unknown"
        
        # Simple heuristic based on Indonesian words
        indonesian_score = self._calculate_indonesian_score(text)
        
        if indonesian_score > 0.3:
            return "indonesian"
        elif any(char.isascii() for char in text):
            return "english"
        else:
            return "unknown"
    
    def _get_image_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract image metadata"""
        try:
            with Image.open(file_path) as img:
                metadata = {
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'file_size': os.path.getsize(file_path)
                }
                
                # Add EXIF data if available
                if hasattr(img, '_getexif') and img._getexif():
                    metadata['exif'] = dict(img._getexif())
                
                return metadata
                
        except Exception as e:
            logger.error(f"Failed to extract image metadata: {e}")
            return {'error': str(e)}
    
    def _create_empty_result(self, start_time: float, image_metadata: Dict) -> OCRExtractionResult:
        """Create empty result for failed extractions"""
        import time
        processing_time = time.time() - start_time
        
        return OCRExtractionResult(
            text_content="",
            confidence_score=0.0,
            bounding_boxes=[],
            language_detected="unknown",
            image_metadata=image_metadata,
            processing_time=processing_time,
            ocr_method="none"
        )
    
    def is_supported_format(self, filename: str) -> bool:
        """Check if file format is supported"""
        return any(filename.lower().endswith(fmt) for fmt in self.supported_formats)
    
    def extract_specific_regions(self, file_path: str, regions: List[Dict]) -> List[OCRExtractionResult]:
        """Extract text from specific regions of an image"""
        results = []
        
        try:
            image = cv2.imread(file_path)
            
            for i, region in enumerate(regions):
                # Extract region coordinates
                x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 100), region.get('h', 100)
                
                # Crop region
                cropped = image[y:y+h, x:x+w]
                
                # Save temporary crop
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, cropped)
                    
                    # Extract from crop
                    result = self.extract_content(tmp_file.name)
                    result.image_metadata['region'] = region
                    results.append(result)
                    
                    # Cleanup
                    os.unlink(tmp_file.name)
        
        except Exception as e:
            logger.error(f"Region extraction failed: {e}")
        
        return results
    
    def extract_indonesian_business_patterns(self, text: str) -> Dict[str, List[str]]:
        """Extract Indonesian business-specific patterns from text"""
        import re
        
        patterns = {
            'ktp_numbers': [],
            'npwp_numbers': [],
            'phone_numbers': [],
            'email_addresses': [],
            'currency_amounts': [],
            'dates': [],
            'company_names': []
        }
        
        # KTP pattern (16 digits)
        ktp_pattern = r'\b\d{16}\b'
        patterns['ktp_numbers'] = re.findall(ktp_pattern, text)
        
        # NPWP pattern
        npwp_pattern = r'\d{2}\.\d{3}\.\d{3}\.\d{1}-\d{3}\.\d{3}'
        patterns['npwp_numbers'] = re.findall(npwp_pattern, text)
        
        # Indonesian phone patterns
        phone_patterns = [
            r'\+62\s?\d{2,3}\s?\d{3,4}\s?\d{3,4}',
            r'08\d{2}\s?\d{3,4}\s?\d{3,4}',
            r'\(021\)\s?\d{3,4}\s?\d{3,4}'
        ]
        
        for pattern in phone_patterns:
            patterns['phone_numbers'].extend(re.findall(pattern, text))
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        patterns['email_addresses'] = re.findall(email_pattern, text)
        
        # Currency patterns
        currency_patterns = [
            r'Rp\.?\s?[\d.,]+ juta',
            r'Rp\.?\s?[\d.,]+',
            r'IDR\s?[\d.,]+',
            r'USD\s?\$?[\d.,]+'
        ]
        
        for pattern in currency_patterns:
            patterns['currency_amounts'].extend(re.findall(pattern, text))
        
        # Date patterns
        date_patterns = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{4}',
            r'\d{1,2}\s+(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|Oktober|November|Desember)\s+\d{4}',
            r'\d{4}-\d{2}-\d{2}'
        ]
        
        for pattern in date_patterns:
            patterns['dates'].extend(re.findall(pattern, text))
        
        # Company name patterns
        company_patterns = [
            r'PT\.?\s+[A-Z][A-Za-z\s]+',
            r'CV\.?\s+[A-Z][A-Za-z\s]+',
            r'UD\.?\s+[A-Z][A-Za-z\s]+',
            r'Toko\s+[A-Z][A-Za-z\s]+'
        ]
        
        for pattern in company_patterns:
            patterns['company_names'].extend(re.findall(pattern, text))
        
        # Clean duplicates and empty results
        for key in patterns:
            patterns[key] = list(set([item.strip() for item in patterns[key] if item.strip()]))
        
        return patterns
    
    def enhance_image_quality(self, file_path: str, output_path: str = None) -> str:
        """Enhance image quality for better OCR results"""
        try:
            # Load image
            with Image.open(file_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1.5)
                
                # Enhance sharpness
                enhancer = ImageEnhance.Sharpness(img)
                img = enhancer.enhance(2.0)
                
                # Apply unsharp mask
                img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
                
                # Resize if too small (minimum 300 DPI equivalent)
                min_width, min_height = 1200, 1600
                if img.width < min_width or img.height < min_height:
                    scale_factor = max(min_width / img.width, min_height / img.height)
                    new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
                    img = img.resize(new_size, Image.Resampling.LANCZOS)
                
                # Save enhanced image
                if output_path is None:
                    output_path = file_path.replace('.', '_enhanced.')
                
                img.save(output_path, 'PNG', optimize=True)
                return output_path
                
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return file_path  # Return original if enhancement fails

# Example usage and testing
if __name__ == "__main__":
    processor = OCRProcessor()
    
    # Test with a sample image
    sample_file = "sample_document.png"
    if os.path.exists(sample_file):
        print("Processing sample image...")
        result = processor.extract_content(sample_file)
        
        print(f"Extracted text length: {len(result.text_content)}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Language: {result.language_detected}")
        print(f"Method: {result.ocr_method}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Bounding boxes: {len(result.bounding_boxes)}")
        
        # Extract Indonesian business patterns
        patterns = processor.extract_indonesian_business_patterns(result.text_content)
        print("\nIndonesian business patterns found:")
        for pattern_type, matches in patterns.items():
            if matches:
                print(f"  {pattern_type}: {matches}")
    else:
        print("Sample image file not found")
        print("Supported formats:", processor.supported_formats)
