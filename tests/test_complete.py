# tests/test_complete.py - Complete Test Suite
import pytest
import asyncio
import os
import tempfile
import json
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock
import io

# Import the main app
from main import app

# Test client
client = TestClient(app)

# Sample test data
SAMPLE_PDF_CONTENT = b"%PDF-1.4\n1 0 obj\n<<\n/Type /Catalog\n/Pages 2 0 R\n>>\nendobj\n"
SAMPLE_DOCX_CONTENT = b"PK\x03\x04" + b"\x00" * 100  # Minimal DOCX header

class TestAuthentication:
    """Test authentication endpoints"""
    
    def test_get_token_success(self):
        """Test successful token generation"""
        response = client.post(
            "/auth/token",
            data={"email": "test@example.com", "password": "testpass"}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert "expires_in" in data
    
    def test_get_token_missing_credentials(self):
        """Test token generation with missing credentials"""
        response = client.post("/auth/token", data={})
        
        assert response.status_code == 422  # Validation error
    
    def test_protected_endpoint_without_token(self):
        """Test accessing protected endpoint without token"""
        response = client.get("/user/profile")
        
        assert response.status_code == 403  # No auth header
    
    def test_protected_endpoint_with_invalid_token(self):
        """Test accessing protected endpoint with invalid token"""
        headers = {"Authorization": "Bearer invalid-token"}
        response = client.get("/user/profile", headers=headers)
        
        # Should return 401 or 403 depending on implementation
        assert response.status_code in [401, 403]

class TestHealthAndSystem:
    """Test system endpoints"""
    
    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "version" in data
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        # Should return Prometheus format
        assert "text/plain" in response.headers.get("content-type", "")

class TestDocumentProcessing:
    """Test document processing endpoints"""
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers for tests"""
        response = client.post(
            "/auth/token",
            data={"email": "test@example.com", "password": "testpass"}
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_process_document_pdf_success(self, auth_headers):
        """Test successful PDF processing"""
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(SAMPLE_PDF_CONTENT)
            tmp_file.flush()
            
            # Test upload
            with open(tmp_file.name, "rb") as f:
                files = {"file": ("test.pdf", f, "application/pdf")}
                response = client.post(
                    "/process-document",
                    headers=auth_headers,
                    files=files
                )
        
        # Cleanup
        os.unlink(tmp_file.name)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "document_info" in data["data"]
    
    def test_process_document_unsupported_format(self, auth_headers):
        """Test processing unsupported file format"""
        files = {"file": ("test.txt", io.BytesIO(b"plain text content"), "text/plain")}
        response = client.post(
            "/process-document",
            headers=auth_headers,
            files=files
        )
        
        # Should handle gracefully or return error
        assert response.status_code in [400, 500]
    
    def test_process_document_no_file(self, auth_headers):
        """Test processing without file"""
        response = client.post(
            "/process-document",
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_process_document_large_file(self, auth_headers):
        """Test processing file that's too large"""
        # Create large file content (mock)
        large_content = b"x" * (51 * 1024 * 1024)  # 51MB
        files = {"file": ("large.pdf", io.BytesIO(large_content), "application/pdf")}
        
        response = client.post(
            "/process-document",
            headers=auth_headers,
            files=files
        )
        
        # Should return error for large file
        assert response.status_code in [413, 400]

class TestSearch:
    """Test search functionality"""
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers for tests"""
        response = client.post(
            "/auth/token",
            data={"email": "test@example.com", "password": "testpass"}
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_search_documents_success(self, auth_headers):
        """Test successful document search"""
        search_data = {
            "query": "kontrak pembayaran",
            "top_k": 5
        }
        
        response = client.post(
            "/search",
            headers=auth_headers,
            params=search_data
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "query" in data
        assert "results" in data
        assert "total" in data
    
    def test_search_empty_query(self, auth_headers):
        """Test search with empty query"""
        response = client.post(
            "/search",
            headers=auth_headers,
            params={"query": ""}
        )
        
        # Should handle empty query gracefully
        assert response.status_code in [200, 400]
    
    def test_search_without_vector_store(self, auth_headers):
        """Test search when vector store is not configured"""
        with patch('main.vector_stores_initialized', False):
            response = client.post(
                "/search",
                headers=auth_headers,
                params={"query": "test query"}
            )
            
            assert response.status_code == 503  # Service unavailable

class TestUserProfile:
    """Test user profile endpoints"""
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers for tests"""
        response = client.post(
            "/auth/token",
            data={"email": "test@example.com", "password": "testpass"}
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_get_user_profile(self, auth_headers):
        """Test getting user profile"""
        response = client.get("/user/profile", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "user" in data
        assert "usage" in data
        assert "limits" in data

class TestProcessors:
    """Test individual processor components"""
    
    @pytest.mark.asyncio
    async def test_pdf_processor(self):
        """Test PDF processor directly"""
        from api.extractors.pdf_processor import PDFProcessor
        
        processor = PDFProcessor()
        
        # Test format support
        assert processor.is_supported_format("test.pdf")
        assert not processor.is_supported_format("test.txt")
        
        # Test with sample PDF
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(SAMPLE_PDF_CONTENT)
            tmp_file.flush()
            
            result = processor.extract_content(tmp_file.name)
            
            assert isinstance(result.text_content, str)
            assert isinstance(result.metadata, dict)
            assert isinstance(result.confidence_score, float)
            assert result.confidence_score >= 0.0
            assert result.confidence_score <= 1.0
        
        # Cleanup
        os.unlink(tmp_file.name)
    
    @pytest.mark.asyncio
    async def test_docx_processor(self):
        """Test DOCX processor directly"""
        from api.extractors.docx_processor import DOCXProcessor
        
        processor = DOCXProcessor()
        
        # Test format support
        assert processor.is_supported_format("test.docx")
        assert processor.is_supported_format("test.doc")
        assert not processor.is_supported_format("test.pdf")

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers for tests"""
        response = client.post(
            "/auth/token",
            data={"email": "test@example.com", "password": "testpass"}
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_rate_limiting_normal_usage(self, auth_headers):
        """Test normal usage within rate limits"""
        # Make several requests
        for i in range(5):
            response = client.get("/user/profile", headers=auth_headers)
            assert response.status_code == 200
    
    @pytest.mark.slow
    def test_rate_limiting_exceeded(self, auth_headers):
        """Test rate limiting when exceeded"""
        # This would require mocking the rate limiter
        # or making many requests (marked as slow test)
        pass

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_404_not_found(self):
        """Test 404 for non-existent endpoints"""
        response = client.get("/non-existent-endpoint")
        assert response.status_code == 404
    
    def test_method_not_allowed(self):
        """Test 405 for wrong methods"""
        response = client.delete("/health")  # Health only supports GET
        assert response.status_code == 405
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers for tests"""
        response = client.post(
            "/auth/token",
            data={"email": "test@example.com", "password": "testpass"}
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_malformed_json(self, auth_headers):
        """Test handling of malformed JSON"""
        response = client.post(
            "/search",
            headers={**auth_headers, "Content-Type": "application/json"},
            data="invalid json{"
        )
        
        assert response.status_code == 422

class TestIntegration:
    """Integration tests for complete workflows"""
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers for tests"""
        response = client.post(
            "/auth/token",
            data={"email": "test@example.com", "password": "testpass"}
        )
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    
    def test_full_document_processing_workflow(self, auth_headers):
        """Test complete document processing workflow"""
        # 1. Process a document
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
            tmp_file.write(SAMPLE_PDF_CONTENT)
            tmp_file.flush()
            
            with open(tmp_file.name, "rb") as f:
                files = {"file": ("test.pdf", f, "application/pdf")}
                process_response = client.post(
                    "/process-document",
                    headers=auth_headers,
                    files=files
                )
        
        os.unlink(tmp_file.name)
        
        assert process_response.status_code == 200
        
        # 2. Search for the processed document
        search_response = client.post(
            "/search",
            headers=auth_headers,
            params={"query": "test document"}
        )
        
        assert search_response.status_code == 200
        
        # 3. Check user profile for updated stats
        profile_response = client.get("/user/profile", headers=auth_headers)
        assert profile_response.status_code == 200

# Performance tests (marked as slow)
class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.slow
    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        import threading
        import time
        
        def make_request():
            response = client.get("/health")
            assert response.status_code == 200
        
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert end_time - start_time < 5.0

# Configuration for pytest
@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Setup test environment"""
    # Set test environment variables
    os.environ["ENVIRONMENT"] = "test"
    os.environ["LOG_LEVEL"] = "WARNING"
    
    yield
    
    # Cleanup after tests
    pass

# Custom pytest markers
pytestmark = [
    pytest.mark.asyncio,
]

# Run specific test categories with:
# pytest tests/test_complete.py::TestAuthentication -v
# pytest tests/test_complete.py -k "not slow" -v
# pytest tests/test_complete.py --cov=api --cov-report=html
