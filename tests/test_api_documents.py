"""
Tests for documents API endpoints
"""
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
import json
from datetime import datetime
import io

from src.api.documents import router, get_document_service
from src.services.document_service import DocumentService
from src.models.document import (
    Document, DocumentType, DocumentStatus, DocumentMetadata,
    DocumentContent, ExtractedText, DocumentUploadResponse
)


@pytest.fixture
def app():
    """Create FastAPI app with documents router"""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/documents")
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_document_service():
    """Create mock document service"""
    return AsyncMock(spec=DocumentService)


@pytest.fixture
def sample_document():
    """Create sample document for testing"""
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
            pages=2
        )
    )


@pytest.fixture
def sample_document_content():
    """Create sample document content for testing"""
    return DocumentContent(
        document_id="test-doc-123",
        full_text="This is the full text of the document.\nSecond page content.",
        pages=[
            ExtractedText(
                page_number=1,
                text="This is the full text of the document.",
                confidence=0.95
            ),
            ExtractedText(
                page_number=2,
                text="Second page content.",
                confidence=0.92
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
            pages=2
        )
    )


class TestUploadDocument:
    """Test cases for document upload endpoint"""

    def test_upload_document_success(self, client, mock_document_service, sample_document):
        """Test successful document upload"""
        # Mock the dependency
        def mock_get_service():
            return mock_document_service
        
        # Mock service responses
        mock_document_service.create_document.return_value = sample_document
        
        # Override dependency
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        # Create test file
        test_file_content = b"fake pdf content"
        files = {"file": ("test.pdf", io.BytesIO(test_file_content), "application/pdf")}
        
        response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["document_id"] == "test-doc-123"
        assert response_data["status"] == DocumentStatus.PENDING
        
        # Verify service was called
        mock_document_service.create_document.assert_called_once()
        mock_document_service.process_document.assert_not_called()  # Called in background

    def test_upload_document_validation_error(self, client, mock_document_service):
        """Test upload with validation error"""
        def mock_get_service():
            return mock_document_service
        
        # Mock service to raise validation error
        mock_document_service.create_document.side_effect = ValueError("File too large")
        
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        test_file_content = b"fake pdf content"
        files = {"file": ("test.pdf", io.BytesIO(test_file_content), "application/pdf")}
        
        response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 400
        assert "File too large" in response.json()["detail"]

    def test_upload_document_server_error(self, client, mock_document_service):
        """Test upload with server error"""
        def mock_get_service():
            return mock_document_service
        
        # Mock service to raise unexpected error
        mock_document_service.create_document.side_effect = Exception("Unexpected error")
        
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        test_file_content = b"fake pdf content"
        files = {"file": ("test.pdf", io.BytesIO(test_file_content), "application/pdf")}
        
        response = client.post("/api/v1/documents/upload", files=files)
        
        assert response.status_code == 500
        assert "Internal server error" in response.json()["detail"]

    def test_upload_document_no_file(self, client, mock_document_service):
        """Test upload without file"""
        def mock_get_service():
            return mock_document_service
        
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.post("/api/v1/documents/upload")
        
        assert response.status_code == 422  # Unprocessable Entity

    def test_upload_document_empty_file(self, client, mock_document_service):
        """Test upload with empty file"""
        def mock_get_service():
            return mock_document_service
        
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        files = {"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")}
        
        response = client.post("/api/v1/documents/upload", files=files)
        
        # Should handle empty file gracefully
        assert response.status_code in [200, 400]  # Depends on validation


class TestGetDocument:
    """Test cases for get document endpoint"""

    def test_get_document_success(self, client, mock_document_service, sample_document):
        """Test successful document retrieval"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document.return_value = sample_document
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/test-doc-123")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["id"] == "test-doc-123"
        assert response_data["filename"] == "test.pdf"
        assert response_data["file_type"] == "pdf"
        assert response_data["status"] == "completed"

    def test_get_document_not_found(self, client, mock_document_service):
        """Test get document when document doesn't exist"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document.return_value = None
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/nonexistent-doc")
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_get_document_service_error(self, client, mock_document_service):
        """Test get document with service error"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document.side_effect = Exception("Database error")
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/test-doc-123")
        
        # Should handle error gracefully
        assert response.status_code == 500


class TestGetDocumentContent:
    """Test cases for get document content endpoint"""

    def test_get_document_content_success(self, client, mock_document_service, sample_document_content):
        """Test successful document content retrieval"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/test-doc-123/content")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["document_id"] == "test-doc-123"
        assert response_data["full_text"] == sample_document_content.full_text
        assert len(response_data["pages"]) == 2
        assert len(response_data["tables"]) == 1

    def test_get_document_content_not_found(self, client, mock_document_service):
        """Test get document content when content doesn't exist"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = None
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/nonexistent-doc/content")
        
        assert response.status_code == 404
        assert "Document content not found" in response.json()["detail"]

    def test_get_document_content_with_page_range(self, client, mock_document_service, sample_document_content):
        """Test get document content with page range filtering"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        # Request only page 1
        response = client.get("/api/v1/documents/test-doc-123/content?page_start=1&page_end=1")
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Should only include page 1
        assert len(response_data["pages"]) == 1
        assert response_data["pages"][0]["page_number"] == 1
        assert response_data["full_text"] == "This is the full text of the document."

    def test_get_document_content_page_range_start_only(self, client, mock_document_service, sample_document_content):
        """Test get document content with only page_start parameter"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        # Request from page 2 onwards
        response = client.get("/api/v1/documents/test-doc-123/content?page_start=2")
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Should only include page 2
        assert len(response_data["pages"]) == 1
        assert response_data["pages"][0]["page_number"] == 2

    def test_get_document_content_page_range_end_only(self, client, mock_document_service, sample_document_content):
        """Test get document content with only page_end parameter"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        # Request up to page 1
        response = client.get("/api/v1/documents/test-doc-123/content?page_end=1")
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Should only include page 1
        assert len(response_data["pages"]) == 1
        assert response_data["pages"][0]["page_number"] == 1


class TestGetDocumentText:
    """Test cases for get document plain text endpoint"""

    def test_get_document_text_success(self, client, mock_document_service, sample_document_content):
        """Test successful document text retrieval"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/test-doc-123/text")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "text" in response_data
        assert response_data["text"] == sample_document_content.full_text

    def test_get_document_text_with_page_range(self, client, mock_document_service, sample_document_content):
        """Test get document text with page range filtering"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        # Request only page 2
        response = client.get("/api/v1/documents/test-doc-123/text?page_start=2&page_end=2")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["text"] == "Second page content."

    def test_get_document_text_not_found(self, client, mock_document_service):
        """Test get document text when document doesn't exist"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = None
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/nonexistent-doc/text")
        
        assert response.status_code == 404
        assert "Document content not found" in response.json()["detail"]


class TestGetDocumentMetadata:
    """Test cases for get document metadata endpoint"""

    def test_get_document_metadata_success(self, client, mock_document_service, sample_document_content):
        """Test successful document metadata retrieval"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/test-doc-123/metadata")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["title"] == "Test Document"
        assert response_data["author"] == "Test Author"
        assert response_data["pages"] == 2

    def test_get_document_metadata_not_found(self, client, mock_document_service):
        """Test get document metadata when document doesn't exist"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = None
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/nonexistent-doc/metadata")
        
        assert response.status_code == 404
        assert "Document metadata not found" in response.json()["detail"]

    def test_get_document_metadata_no_metadata(self, client, mock_document_service, sample_document_content):
        """Test get document metadata when document has no metadata"""
        def mock_get_service():
            return mock_document_service
        
        # Remove metadata from sample content
        sample_document_content.metadata = None
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/test-doc-123/metadata")
        
        assert response.status_code == 404
        assert "Document metadata not found" in response.json()["detail"]


class TestGetDocumentTables:
    """Test cases for get document tables endpoint"""

    def test_get_document_tables_success(self, client, mock_document_service, sample_document_content):
        """Test successful document tables retrieval"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/test-doc-123/tables")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "tables" in response_data
        assert len(response_data["tables"]) == 1
        assert response_data["tables"][0]["page"] == 1

    def test_get_document_tables_with_page_filter(self, client, mock_document_service, sample_document_content):
        """Test get document tables with page filtering"""
        def mock_get_service():
            return mock_document_service
        
        # Add table on page 2
        sample_document_content.tables.append({
            "page": 2,
            "table_number": 1,
            "data": [["Col1", "Col2"], ["A", "B"]]
        })
        
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        # Request only tables from page 2
        response = client.get("/api/v1/documents/test-doc-123/tables?page=2")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert len(response_data["tables"]) == 1
        assert response_data["tables"][0]["page"] == 2

    def test_get_document_tables_no_tables(self, client, mock_document_service, sample_document_content):
        """Test get document tables when document has no tables"""
        def mock_get_service():
            return mock_document_service
        
        # Remove tables
        sample_document_content.tables = []
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/test-doc-123/tables")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["tables"] == []

    def test_get_document_tables_not_found(self, client, mock_document_service):
        """Test get document tables when document doesn't exist"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = None
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/nonexistent-doc/tables")
        
        assert response.status_code == 404
        assert "Document content not found" in response.json()["detail"]


class TestDeleteDocument:
    """Test cases for delete document endpoint"""

    def test_delete_document_success(self, client, mock_document_service):
        """Test successful document deletion"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.delete_document.return_value = True
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.delete("/api/v1/documents/test-doc-123")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["message"] == "Document deleted successfully"
        mock_document_service.delete_document.assert_called_once_with("test-doc-123")

    def test_delete_document_not_found(self, client, mock_document_service):
        """Test delete document when document doesn't exist"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.delete_document.return_value = False
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.delete("/api/v1/documents/nonexistent-doc")
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_delete_document_service_error(self, client, mock_document_service):
        """Test delete document with service error"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.delete_document.side_effect = Exception("Deletion failed")
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.delete("/api/v1/documents/test-doc-123")
        
        # Should handle error gracefully
        assert response.status_code == 500


class TestListDocuments:
    """Test cases for list documents endpoint"""

    def test_list_documents_placeholder(self, client, mock_document_service):
        """Test list documents endpoint (currently returns empty list)"""
        def mock_get_service():
            return mock_document_service
        
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/")
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Currently returns empty list as it's not implemented
        assert response_data == []

    def test_list_documents_with_parameters(self, client, mock_document_service):
        """Test list documents with query parameters"""
        def mock_get_service():
            return mock_document_service
        
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.get("/api/v1/documents/?status=completed&limit=10&offset=0")
        
        assert response.status_code == 200
        # Currently returns empty list regardless of parameters


class TestReprocessDocument:
    """Test cases for reprocess document endpoint"""

    def test_reprocess_document_success(self, client, mock_document_service, sample_document):
        """Test successful document reprocessing"""
        def mock_get_service():
            return mock_document_service
        
        # Mock storage service
        mock_storage = AsyncMock()
        mock_storage.get_file.return_value = b"file content"
        mock_document_service.storage = mock_storage
        
        mock_document_service.get_document.return_value = sample_document
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.post("/api/v1/documents/test-doc-123/reprocess")
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["message"] == "Document reprocessing started"
        assert response_data["document_id"] == "test-doc-123"

    def test_reprocess_document_not_found(self, client, mock_document_service):
        """Test reprocess document when document doesn't exist"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document.return_value = None
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.post("/api/v1/documents/nonexistent-doc/reprocess")
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_reprocess_document_no_storage_path(self, client, mock_document_service, sample_document):
        """Test reprocess document when document has no storage path"""
        def mock_get_service():
            return mock_document_service
        
        # Remove storage path
        sample_document.storage_path = None
        mock_document_service.get_document.return_value = sample_document
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.post("/api/v1/documents/test-doc-123/reprocess")
        
        assert response.status_code == 400
        assert "Document file not found in storage" in response.json()["detail"]

    def test_reprocess_document_storage_error(self, client, mock_document_service, sample_document):
        """Test reprocess document with storage error"""
        def mock_get_service():
            return mock_document_service
        
        # Mock storage service with error
        mock_storage = AsyncMock()
        mock_storage.get_file.side_effect = Exception("Storage error")
        mock_document_service.storage = mock_storage
        
        mock_document_service.get_document.return_value = sample_document
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        response = client.post("/api/v1/documents/test-doc-123/reprocess")
        
        # Should handle error gracefully
        assert response.status_code == 500


class TestAPIIntegration:
    """Integration tests for document API endpoints"""

    def test_api_error_handling_consistency(self, client, mock_document_service):
        """Test that all endpoints handle errors consistently"""
        def mock_get_service():
            return mock_document_service
        
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        # Test various endpoints with service errors
        endpoints = [
            ("/api/v1/documents/test-doc", "GET"),
            ("/api/v1/documents/test-doc/content", "GET"),
            ("/api/v1/documents/test-doc/text", "GET"),
            ("/api/v1/documents/test-doc/metadata", "GET"),
            ("/api/v1/documents/test-doc/tables", "GET"),
            ("/api/v1/documents/test-doc", "DELETE"),
        ]
        
        for endpoint, method in endpoints:
            # Configure service to raise error
            if method == "GET":
                if "content" in endpoint or "text" in endpoint or "metadata" in endpoint or "tables" in endpoint:
                    mock_document_service.get_document_content.side_effect = Exception("Service error")
                else:
                    mock_document_service.get_document.side_effect = Exception("Service error")
            elif method == "DELETE":
                mock_document_service.delete_document.side_effect = Exception("Service error")
            
            if method == "GET":
                response = client.get(endpoint)
            elif method == "DELETE":
                response = client.delete(endpoint)
            
            # All should return 500 for unexpected errors
            assert response.status_code == 500

    def test_api_response_formats(self, client, mock_document_service, sample_document, sample_document_content):
        """Test that API responses follow consistent format"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document.return_value = sample_document
        mock_document_service.get_document_content.return_value = sample_document_content
        mock_document_service.delete_document.return_value = True
        
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        # Test successful responses
        endpoints_and_expected_keys = [
            ("/api/v1/documents/test-doc-123", ["id", "filename", "file_type", "status"]),
            ("/api/v1/documents/test-doc-123/content", ["document_id", "full_text", "pages"]),
            ("/api/v1/documents/test-doc-123/text", ["text"]),
            ("/api/v1/documents/test-doc-123/metadata", ["title", "author", "pages"]),
            ("/api/v1/documents/test-doc-123/tables", ["tables"]),
        ]
        
        for endpoint, expected_keys in endpoints_and_expected_keys:
            response = client.get(endpoint)
            assert response.status_code == 200
            
            response_data = response.json()
            for key in expected_keys:
                assert key in response_data

    def test_api_parameter_validation(self, client, mock_document_service, sample_document_content):
        """Test API parameter validation"""
        def mock_get_service():
            return mock_document_service
        
        mock_document_service.get_document_content.return_value = sample_document_content
        client.app.dependency_overrides[get_document_service] = mock_get_service
        
        # Test invalid parameter types
        response = client.get("/api/v1/documents/test-doc/content?page_start=invalid")
        assert response.status_code == 422  # Validation error
        
        response = client.get("/api/v1/documents/test-doc/content?page_end=invalid")
        assert response.status_code == 422  # Validation error

    def test_api_cors_and_headers(self, client):
        """Test API headers and CORS behavior"""
        response = client.get("/api/v1/documents/test-doc")
        
        # Should have appropriate headers
        assert "content-type" in response.headers
        assert response.headers["content-type"] == "application/json"