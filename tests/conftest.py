"""
Pytest configuration and shared fixtures for OritzPDF tests
"""
import pytest
import pytest_asyncio
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import AsyncGenerator, Generator, Optional
import io
import json
from datetime import datetime

# Import the application modules
from src.models.document import (
    Document, DocumentType, DocumentStatus, DocumentMetadata,
    DocumentContent, ExtractedText, DocumentUploadRequest
)
from src.models.analysis import (
    SummarizationResponse, QuestionAnswerResponse,
    DataExtractionResponse, TranslationResponse
)
from src.services.storage_service import LocalStorageService, S3StorageService
from src.services.cache_service import CacheService
from src.services.document_service import DocumentService
from src.services.pdf_processor import PyMuPDFProcessor, ComprehensivePDFProcessor
from src.config import Settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_settings() -> Settings:
    """Create test configuration settings"""
    return Settings(
        APP_NAME="OritzPDF-Test",
        DEBUG=True,
        LOG_LEVEL="DEBUG",
        DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/test",
        REDIS_URL="redis://localhost:6379/1",
        STORAGE_TYPE="local",
        LOCAL_STORAGE_PATH="./test_uploads",
        MAX_FILE_SIZE_MB=10,
        SUPPORTED_FORMATS="pdf,docx,txt",
        JWT_SECRET_KEY="test-secret-key"
    )


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files"""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_pdf_content() -> bytes:
    """Create sample PDF content for testing"""
    # This is a minimal PDF structure
    pdf_content = b"""%PDF-1.4
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
    return pdf_content


@pytest.fixture
def sample_docx_content() -> bytes:
    """Create sample DOCX content for testing"""
    # Minimal DOCX structure (would need python-docx to create properly)
    return b"PK\x03\x04\x14\x00\x00\x00\x08\x00" + b"0" * 100  # Minimal ZIP structure


@pytest.fixture
def sample_document() -> Document:
    """Create a sample document for testing"""
    return Document(
        id="test-doc-123",
        filename="test.pdf",
        file_type=DocumentType.PDF,
        file_size=1024,
        status=DocumentStatus.COMPLETED,
        upload_time=datetime.utcnow(),
        processing_time=1.5,
        storage_path="/path/to/test.pdf",
        metadata=DocumentMetadata(
            title="Test Document",
            author="Test Author",
            pages=1
        )
    )


@pytest.fixture
def sample_document_content() -> DocumentContent:
    """Create sample document content for testing"""
    return DocumentContent(
        document_id="test-doc-123",
        full_text="This is a test document with sample text.",
        pages=[
            ExtractedText(
                page_number=1,
                text="This is a test document with sample text.",
                confidence=0.95
            )
        ],
        tables=[
            {
                "page": 1,
                "table_number": 1,
                "data": [["Header1", "Header2"], ["Row1Col1", "Row1Col2"]]
            }
        ],
        images=[
            {
                "page": 1,
                "index": 0,
                "width": 100,
                "height": 100
            }
        ],
        metadata=DocumentMetadata(
            title="Test Document",
            author="Test Author",
            pages=1
        )
    )


@pytest.fixture
def sample_upload_request() -> DocumentUploadRequest:
    """Create a sample upload request"""
    return DocumentUploadRequest(
        filename="test.pdf",
        content_type="application/pdf",
        file_size=1024
    )


@pytest.fixture
async def mock_redis():
    """Mock Redis client for cache testing"""
    mock_redis = AsyncMock()
    mock_redis.get.return_value = None
    mock_redis.set.return_value = True
    mock_redis.setex.return_value = True
    mock_redis.delete.return_value = 1
    mock_redis.exists.return_value = False
    mock_redis.scan.return_value = (0, [])
    mock_redis.close.return_value = None
    
    with patch('redis.asyncio.from_url', return_value=mock_redis):
        yield mock_redis


@pytest.fixture
async def mock_s3_client():
    """Mock S3 client for storage testing"""
    mock_client = AsyncMock()
    mock_client.put_object.return_value = {"ETag": "test-etag"}
    mock_client.get_object.return_value = {
        "Body": AsyncMock(read=AsyncMock(return_value=b"test content"))
    }
    mock_client.delete_object.return_value = {"DeleteMarker": False}
    mock_client.head_object.return_value = {"ContentLength": 100}
    
    mock_session = AsyncMock()
    mock_session.client.return_value.__aenter__.return_value = mock_client
    mock_session.client.return_value.__aexit__.return_value = None
    
    with patch('aioboto3.Session', return_value=mock_session):
        yield mock_client


@pytest.fixture
async def local_storage_service(temp_dir: Path) -> LocalStorageService:
    """Create a local storage service for testing"""
    return LocalStorageService(str(temp_dir))


@pytest.fixture
async def s3_storage_service(mock_s3_client) -> S3StorageService:
    """Create an S3 storage service for testing"""
    return S3StorageService(
        bucket_name="test-bucket",
        access_key="test-access-key",
        secret_key="test-secret-key"
    )


@pytest.fixture
async def cache_service(mock_redis) -> CacheService:
    """Create a cache service for testing"""
    service = CacheService("redis://localhost:6379/1")
    await service.connect()
    return service


@pytest.fixture
async def mock_pdf_processor():
    """Mock PDF processor for testing"""
    mock_processor = AsyncMock(spec=ComprehensivePDFProcessor)
    
    # Mock the process_document method
    async def mock_process_document(file_path: str, document_id: str) -> DocumentContent:
        return DocumentContent(
            document_id=document_id,
            full_text="Mocked extracted text from PDF",
            pages=[
                ExtractedText(
                    page_number=1,
                    text="Mocked page 1 text",
                    confidence=0.95
                )
            ],
            tables=[],
            images=[],
            metadata=DocumentMetadata(
                title="Mocked PDF Title",
                pages=1
            )
        )
    
    mock_processor.process_document.side_effect = mock_process_document
    return mock_processor


@pytest.fixture
async def document_service(
    local_storage_service: LocalStorageService,
    cache_service: CacheService,
    mock_pdf_processor
) -> DocumentService:
    """Create a document service for testing"""
    service = DocumentService(
        storage_service=local_storage_service,
        cache_service=cache_service
    )
    service.pdf_processor = mock_pdf_processor
    return service


@pytest.fixture
def sample_summarization_response() -> SummarizationResponse:
    """Create a sample summarization response"""
    return SummarizationResponse(
        document_id="test-doc-123",
        summary="This is a test summary of the document.",
        pages_summarized=[1, 2],
        confidence_score=0.85
    )


@pytest.fixture
def sample_qa_response() -> QuestionAnswerResponse:
    """Create a sample Q&A response"""
    return QuestionAnswerResponse(
        document_id="test-doc-123",
        question="What is this document about?",
        answer="This document is about testing.",
        confidence_score=0.75,
        source_pages=[1],
        context="Sample context from the document"
    )


@pytest.fixture
def sample_translation_response() -> TranslationResponse:
    """Create a sample translation response"""
    return TranslationResponse(
        document_id="test-doc-123",
        source_language="en",
        target_language="es",
        translated_text="Este es un documento de prueba.",
        pages_translated=[1]
    )


@pytest.fixture
def sample_data_extraction_response() -> DataExtractionResponse:
    """Create a sample data extraction response"""
    return DataExtractionResponse(
        document_id="test-doc-123",
        entities=[],
        tables=[{"page": 1, "data": [["A", "B"], ["1", "2"]]}],
        numerical_data={"total": 100, "average": 50}
    )


@pytest.fixture
def mock_file_upload():
    """Mock FastAPI UploadFile for testing"""
    mock_file = MagicMock()
    mock_file.filename = "test.pdf"
    mock_file.content_type = "application/pdf"
    mock_file.size = 1024
    mock_file.read = AsyncMock(return_value=b"test pdf content")
    return mock_file


@pytest.fixture
async def test_pdf_file(temp_dir: Path, sample_pdf_content: bytes) -> Path:
    """Create a test PDF file"""
    pdf_path = temp_dir / "test.pdf"
    pdf_path.write_bytes(sample_pdf_content)
    return pdf_path


@pytest.fixture
def mock_huggingface_models():
    """Mock Hugging Face models for NLP tasks"""
    with patch('transformers.pipeline') as mock_pipeline:
        # Mock summarization pipeline
        mock_summarizer = MagicMock()
        mock_summarizer.return_value = [{"summary_text": "Mocked summary"}]
        
        # Mock Q&A pipeline
        mock_qa = MagicMock()
        mock_qa.return_value = {
            "answer": "Mocked answer",
            "score": 0.75,
            "start": 0,
            "end": 10
        }
        
        # Return different mocks based on task
        def pipeline_side_effect(task, **kwargs):
            if task == "summarization":
                return mock_summarizer
            elif task == "question-answering":
                return mock_qa
            else:
                return MagicMock()
        
        mock_pipeline.side_effect = pipeline_side_effect
        yield mock_pipeline


@pytest.fixture
def mock_spacy_nlp():
    """Mock spaCy NLP model for entity extraction"""
    mock_nlp = MagicMock()
    mock_doc = MagicMock()
    mock_doc.ents = []
    mock_nlp.return_value = mock_doc
    
    with patch('spacy.load', return_value=mock_nlp):
        yield mock_nlp


@pytest.fixture
async def cleanup_test_files():
    """Cleanup test files after tests"""
    yield
    # Cleanup code here if needed
    pass


# Helper functions for tests
def create_test_pdf_bytes() -> bytes:
    """Create minimal valid PDF bytes for testing"""
    return b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]>>endobj
xref
0 4
0000000000 65535 f 
0000000009 00000 n 
0000000074 00000 n 
0000000120 00000 n 
trailer<</Size 4/Root 1 0 R>>
startxref
185
%%EOF"""


def assert_document_valid(document: Document) -> None:
    """Assert that a document object is valid"""
    assert document.id is not None
    assert document.filename is not None
    assert document.file_type is not None
    assert document.file_size > 0
    assert document.status in [status.value for status in DocumentStatus]
    assert document.upload_time is not None


def assert_document_content_valid(content: DocumentContent) -> None:
    """Assert that document content is valid"""
    assert content.document_id is not None
    assert content.full_text is not None
    assert isinstance(content.pages, list)
    for page in content.pages:
        assert page.page_number > 0
        assert page.text is not None