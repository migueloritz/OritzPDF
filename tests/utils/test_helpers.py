"""
Test utilities and helper functions for OritzPDF tests
"""
import asyncio
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union
import json
import uuid
import hashlib
from unittest.mock import AsyncMock, Mock, MagicMock
import io

from src.models.document import (
    Document, DocumentType, DocumentStatus, DocumentMetadata,
    DocumentContent, ExtractedText, DocumentUploadRequest
)
from src.models.analysis import (
    SummarizationResponse, QuestionAnswerResponse,
    TranslationResponse, DataExtractionResponse, ExtractedEntity, EntityType,
    ComparisonResponse, ComparisonDifference
)


class TestDataFactory:
    """Factory class for creating test data objects"""
    
    @staticmethod
    def create_document(
        doc_id: Optional[str] = None,
        filename: str = "test.pdf",
        file_type: DocumentType = DocumentType.PDF,
        file_size: int = 1024,
        status: DocumentStatus = DocumentStatus.COMPLETED,
        storage_path: Optional[str] = None,
        **kwargs
    ) -> Document:
        """Create a test document with optional custom properties"""
        return Document(
            id=doc_id or str(uuid.uuid4()),
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            status=status,
            upload_time=kwargs.get('upload_time', datetime.utcnow()),
            processing_time=kwargs.get('processing_time', 1.5),
            error_message=kwargs.get('error_message'),
            storage_path=storage_path or f"/path/to/{filename}",
            metadata=kwargs.get('metadata')
        )
    
    @staticmethod
    def create_document_metadata(
        title: str = "Test Document",
        author: str = "Test Author",
        pages: int = 1,
        **kwargs
    ) -> DocumentMetadata:
        """Create test document metadata"""
        return DocumentMetadata(
            title=title,
            author=author,
            subject=kwargs.get('subject', "Test Subject"),
            keywords=kwargs.get('keywords', ["test", "document"]),
            creation_date=kwargs.get('creation_date', datetime.utcnow()),
            modification_date=kwargs.get('modification_date', datetime.utcnow()),
            pages=pages,
            language=kwargs.get('language', "en")
        )
    
    @staticmethod
    def create_extracted_text(
        page_number: int = 1,
        text: str = "Sample extracted text",
        confidence: float = 0.95,
        **kwargs
    ) -> ExtractedText:
        """Create test extracted text"""
        return ExtractedText(
            page_number=page_number,
            text=text,
            confidence=confidence,
            bbox=kwargs.get('bbox')
        )
    
    @staticmethod
    def create_document_content(
        document_id: str = "test-doc-123",
        full_text: str = "This is test document content.",
        num_pages: int = 1,
        **kwargs
    ) -> DocumentContent:
        """Create test document content"""
        pages = []
        if num_pages > 0:
            if num_pages == 1:
                pages = [TestDataFactory.create_extracted_text(1, full_text)]
            else:
                # Split text across pages
                words = full_text.split()
                words_per_page = max(1, len(words) // num_pages)
                for i in range(num_pages):
                    start_idx = i * words_per_page
                    end_idx = min((i + 1) * words_per_page, len(words))
                    page_text = " ".join(words[start_idx:end_idx])
                    pages.append(TestDataFactory.create_extracted_text(i + 1, page_text))
        
        return DocumentContent(
            document_id=document_id,
            full_text=full_text,
            pages=pages,
            tables=kwargs.get('tables', []),
            images=kwargs.get('images', []),
            metadata=kwargs.get('metadata', TestDataFactory.create_document_metadata())
        )
    
    @staticmethod
    def create_upload_request(
        filename: str = "test.pdf",
        content_type: str = "application/pdf",
        file_size: int = 1024
    ) -> DocumentUploadRequest:
        """Create test upload request"""
        return DocumentUploadRequest(
            filename=filename,
            content_type=content_type,
            file_size=file_size
        )
    
    @staticmethod
    def create_summarization_response(
        document_id: str = "test-doc-123",
        summary: str = "This is a test summary.",
        pages: Optional[List[int]] = None,
        confidence: float = 0.85
    ) -> SummarizationResponse:
        """Create test summarization response"""
        return SummarizationResponse(
            document_id=document_id,
            summary=summary,
            pages_summarized=pages or [1],
            confidence_score=confidence
        )
    
    @staticmethod
    def create_qa_response(
        document_id: str = "test-doc-123",
        question: str = "What is this document about?",
        answer: str = "This document is about testing.",
        confidence: float = 0.75,
        source_pages: Optional[List[int]] = None
    ) -> QuestionAnswerResponse:
        """Create test Q&A response"""
        return QuestionAnswerResponse(
            document_id=document_id,
            question=question,
            answer=answer,
            confidence_score=confidence,
            source_pages=source_pages or [1],
            context="Sample context from the document"
        )
    
    @staticmethod
    def create_table_data(
        page: int = 1,
        table_number: int = 1,
        headers: Optional[List[str]] = None,
        rows: Optional[List[List[str]]] = None
    ) -> Dict[str, Any]:
        """Create test table data"""
        headers = headers or ["Column1", "Column2", "Column3"]
        rows = rows or [
            ["Row1Col1", "Row1Col2", "Row1Col3"],
            ["Row2Col1", "Row2Col2", "Row2Col3"]
        ]
        
        return {
            "page": page,
            "table_number": table_number,
            "data": [headers] + rows
        }
    
    @staticmethod
    def create_image_data(
        page: int = 1,
        index: int = 0,
        width: int = 100,
        height: int = 100,
        **kwargs
    ) -> Dict[str, Any]:
        """Create test image data"""
        return {
            "page": page,
            "index": index,
            "width": width,
            "height": height,
            "colorspace": kwargs.get("colorspace", "RGB"),
            "xref": kwargs.get("xref", 123)
        }


class MockFactory:
    """Factory class for creating mock objects"""
    
    @staticmethod
    def create_mock_redis() -> AsyncMock:
        """Create mock Redis client"""
        mock_redis = AsyncMock()
        mock_redis.get.return_value = None
        mock_redis.set.return_value = True
        mock_redis.setex.return_value = True
        mock_redis.delete.return_value = 1
        mock_redis.exists.return_value = False
        mock_redis.scan.return_value = (0, [])
        mock_redis.close.return_value = None
        return mock_redis
    
    @staticmethod
    def create_mock_s3_client() -> AsyncMock:
        """Create mock S3 client"""
        mock_client = AsyncMock()
        mock_client.put_object.return_value = {"ETag": "test-etag"}
        mock_client.get_object.return_value = {
            "Body": AsyncMock(read=AsyncMock(return_value=b"test content"))
        }
        mock_client.delete_object.return_value = {"DeleteMarker": False}
        mock_client.head_object.return_value = {"ContentLength": 100}
        return mock_client
    
    @staticmethod
    def create_mock_pdf_document(text_content: str = "Sample PDF text") -> Mock:
        """Create mock PyMuPDF document"""
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = text_content
        mock_page.get_text.side_effect = lambda arg=None: {
            None: text_content,
            "dict": {"blocks": []}
        }.get(arg, text_content)
        mock_page.get_images.return_value = []
        
        mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
        mock_doc.page_count = 1
        mock_doc.metadata = {
            "title": "Mock PDF",
            "author": "Test Author"
        }
        mock_doc.close = Mock()
        return mock_doc
    
    @staticmethod
    def create_mock_file_upload(
        filename: str = "test.pdf",
        content: bytes = b"fake pdf content",
        content_type: str = "application/pdf"
    ) -> Mock:
        """Create mock FastAPI UploadFile"""
        mock_file = Mock()
        mock_file.filename = filename
        mock_file.content_type = content_type
        mock_file.size = len(content)
        mock_file.read = AsyncMock(return_value=content)
        return mock_file


class PDFTestHelper:
    """Helper class for PDF-related testing"""
    
    @staticmethod
    def create_minimal_pdf_bytes() -> bytes:
        """Create minimal valid PDF bytes for testing"""
        return b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj

2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj

3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Resources <<
/Font <<
/F1 4 0 R
>>
>>
/Contents 5 0 R
>>
endobj

4 0 obj
<<
/Type /Font
/Subtype /Type1
/BaseFont /Times-Roman
>>
endobj

5 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
72 720 Td
(Hello World) Tj
ET
endstream
endobj

xref
0 6
0000000000 65535 f 
0000000010 00000 n 
0000000079 00000 n 
0000000173 00000 n 
0000000301 00000 n 
0000000380 00000 n 
trailer
<<
/Size 6
/Root 1 0 R
>>
startxref
492
%%EOF"""
    
    @staticmethod
    def create_pdf_with_text(text: str) -> bytes:
        """Create PDF bytes with specific text content"""
        # This is a simplified version - in reality, you'd need proper PDF generation
        base_pdf = PDFTestHelper.create_minimal_pdf_bytes()
        # Replace "Hello World" with custom text (simplified)
        return base_pdf.replace(b"Hello World", text.encode()[:11])  # PDF format limitations
    
    @staticmethod
    def create_multi_page_pdf_bytes(pages: List[str]) -> bytes:
        """Create multi-page PDF bytes"""
        # Simplified multi-page PDF - in reality, this would be more complex
        base = PDFTestHelper.create_minimal_pdf_bytes()
        # This is a placeholder - real implementation would need proper PDF library
        return base


class FileTestHelper:
    """Helper class for file-related testing"""
    
    @staticmethod
    def create_temp_file(content: bytes, suffix: str = ".pdf") -> Path:
        """Create temporary file with content"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(content)
        temp_file.close()
        return Path(temp_file.name)
    
    @staticmethod
    def create_temp_directory() -> Path:
        """Create temporary directory"""
        return Path(tempfile.mkdtemp())
    
    @staticmethod
    def cleanup_temp_path(path: Path):
        """Clean up temporary file or directory"""
        if path.is_file():
            path.unlink(missing_ok=True)
        elif path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


class AsyncTestHelper:
    """Helper class for async testing"""
    
    @staticmethod
    def run_async_test(coro):
        """Run async test function in new event loop"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()
    
    @staticmethod
    async def wait_for_condition(
        condition_func,
        timeout: float = 5.0,
        interval: float = 0.1
    ) -> bool:
        """Wait for a condition to become true"""
        start_time = asyncio.get_event_loop().time()
        while True:
            if condition_func():
                return True
            
            if asyncio.get_event_loop().time() - start_time > timeout:
                return False
            
            await asyncio.sleep(interval)
    
    @staticmethod
    def create_async_context_manager(mock_obj):
        """Create async context manager from mock object"""
        mock_obj.__aenter__ = AsyncMock(return_value=mock_obj)
        mock_obj.__aexit__ = AsyncMock(return_value=None)
        return mock_obj


class ValidationHelper:
    """Helper class for validation and assertions"""
    
    @staticmethod
    def assert_document_valid(document: Document):
        """Assert that a document object is valid"""
        assert document.id is not None
        assert document.filename is not None
        assert document.file_type is not None
        assert document.file_size > 0
        assert document.status in [status.value for status in DocumentStatus]
        assert document.upload_time is not None
        assert isinstance(document.upload_time, datetime)
    
    @staticmethod
    def assert_document_content_valid(content: DocumentContent):
        """Assert that document content is valid"""
        assert content.document_id is not None
        assert content.full_text is not None
        assert isinstance(content.pages, list)
        for page in content.pages:
            assert page.page_number > 0
            assert page.text is not None
            assert 0 <= page.confidence <= 1
    
    @staticmethod
    def assert_api_response_format(response_data: Dict[str, Any], expected_keys: List[str]):
        """Assert API response has expected format"""
        assert isinstance(response_data, dict)
        for key in expected_keys:
            assert key in response_data, f"Missing key: {key}"
    
    @staticmethod
    def assert_error_response(response_data: Dict[str, Any]):
        """Assert error response has correct format"""
        assert "detail" in response_data
        assert isinstance(response_data["detail"], str)


class CacheTestHelper:
    """Helper class for cache testing"""
    
    @staticmethod
    def create_cache_key(prefix: str, identifier: str) -> str:
        """Create cache key in expected format"""
        return f"oritzpdf:{prefix}:{identifier}"
    
    @staticmethod
    def create_question_hash(question: str) -> str:
        """Create question hash for Q&A caching"""
        return hashlib.md5(question.encode()).hexdigest()
    
    @staticmethod
    def serialize_for_cache(obj) -> str:
        """Serialize object for cache storage"""
        if hasattr(obj, 'model_dump_json'):
            return obj.model_dump_json()
        return json.dumps(obj)


class PerformanceTestHelper:
    """Helper class for performance testing"""
    
    @staticmethod
    def measure_execution_time(func, *args, **kwargs) -> tuple:
        """Measure function execution time"""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    async def measure_async_execution_time(coro) -> tuple:
        """Measure async function execution time"""
        import time
        start_time = time.time()
        result = await coro
        end_time = time.time()
        return result, end_time - start_time
    
    @staticmethod
    def create_large_content(size_mb: int) -> bytes:
        """Create large content for performance testing"""
        return b"x" * (size_mb * 1024 * 1024)


class TestScenarioHelper:
    """Helper class for creating test scenarios"""
    
    @staticmethod
    def create_error_scenarios() -> List[Dict[str, Any]]:
        """Create common error scenarios for testing"""
        return [
            {
                "name": "Network Error",
                "exception": ConnectionError("Network connection failed"),
                "expected_status": 500
            },
            {
                "name": "Timeout Error", 
                "exception": TimeoutError("Operation timed out"),
                "expected_status": 500
            },
            {
                "name": "Permission Error",
                "exception": PermissionError("Access denied"),
                "expected_status": 403
            },
            {
                "name": "File Not Found",
                "exception": FileNotFoundError("File not found"),
                "expected_status": 404
            }
        ]
    
    @staticmethod
    def create_document_types_scenarios() -> List[Dict[str, Any]]:
        """Create test scenarios for different document types"""
        return [
            {
                "filename": "document.pdf",
                "content_type": "application/pdf",
                "file_type": DocumentType.PDF,
                "content": PDFTestHelper.create_minimal_pdf_bytes()
            },
            {
                "filename": "document.docx",
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "file_type": DocumentType.DOCX,
                "content": b"PK\x03\x04" + b"0" * 100  # Minimal ZIP structure
            },
            {
                "filename": "document.txt",
                "content_type": "text/plain",
                "file_type": DocumentType.TXT,
                "content": b"Plain text content"
            }
        ]


# Utility functions that can be imported directly
def create_test_document(**kwargs) -> Document:
    """Convenience function to create test document"""
    return TestDataFactory.create_document(**kwargs)


def create_test_content(**kwargs) -> DocumentContent:
    """Convenience function to create test document content"""
    return TestDataFactory.create_document_content(**kwargs)


def create_minimal_pdf() -> bytes:
    """Convenience function to create minimal PDF"""
    return PDFTestHelper.create_minimal_pdf_bytes()


def assert_valid_document(document: Document):
    """Convenience function to validate document"""
    ValidationHelper.assert_document_valid(document)


def assert_valid_content(content: DocumentContent):
    """Convenience function to validate document content"""
    ValidationHelper.assert_document_content_valid(content)


# Context managers for testing
class TemporaryDirectory:
    """Context manager for temporary directory"""
    
    def __init__(self):
        self.path = None
    
    def __enter__(self) -> Path:
        self.path = FileTestHelper.create_temp_directory()
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path:
            FileTestHelper.cleanup_temp_path(self.path)


class TemporaryFile:
    """Context manager for temporary file"""
    
    def __init__(self, content: bytes, suffix: str = ".pdf"):
        self.content = content
        self.suffix = suffix
        self.path = None
    
    def __enter__(self) -> Path:
        self.path = FileTestHelper.create_temp_file(self.content, self.suffix)
        return self.path
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.path:
            FileTestHelper.cleanup_temp_path(self.path)