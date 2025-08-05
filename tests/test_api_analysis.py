"""
Tests for analysis API endpoints
"""
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, Mock, patch
import json
import hashlib

from src.api.analysis import router, get_services
from src.services.document_service import DocumentService
from src.services.cache_service import CacheService
from src.models.document import DocumentContent, ExtractedText, DocumentMetadata
from src.models.analysis import (
    SummarizationRequest, SummarizationResponse,
    QuestionAnswerRequest, QuestionAnswerResponse,
    TranslationRequest, TranslationResponse,
    DataExtractionRequest, DataExtractionResponse, EntityType,
    ComparisonRequest, ComparisonResponse, ComparisonDifference
)


@pytest.fixture
def app():
    """Create FastAPI app with analysis router"""
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/analysis")
    return app


@pytest.fixture
def client(app):
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_services():
    """Create mock services tuple"""
    document_service = AsyncMock(spec=DocumentService)
    cache_service = AsyncMock(spec=CacheService)
    return document_service, cache_service


@pytest.fixture
def sample_document_content():
    """Create sample document content for testing"""
    return DocumentContent(
        document_id="test-doc-123",
        full_text="This is a comprehensive test document. It contains multiple paragraphs with various information. The document discusses important topics and provides detailed analysis.",
        pages=[
            ExtractedText(
                page_number=1,
                text="This is a comprehensive test document. It contains multiple paragraphs with various information.",
                confidence=0.95
            ),
            ExtractedText(
                page_number=2,
                text="The document discusses important topics and provides detailed analysis.",
                confidence=0.93
            )
        ],
        tables=[
            {
                "page": 1,
                "table_number": 1,
                "data": [["Name", "Value"], ["Item1", "100"], ["Item2", "200"]]
            }
        ],
        images=[],
        metadata=DocumentMetadata(
            title="Test Analysis Document",
            author="Test Author",
            pages=2
        )
    )


class TestSummarizeEndpoint:
    """Test cases for document summarization endpoint"""

    def test_summarize_document_success(self, client, mock_services, sample_document_content):
        """Test successful document summarization"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        # Mock service responses
        document_service.get_document_content.return_value = sample_document_content
        cache_service.get_summary.return_value = None  # Cache miss
        
        # Mock cache set
        cache_service.set_summary.return_value = True
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "max_length": 200,
            "min_length": 50
        }
        
        response = client.post("/api/v1/analysis/summarize", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["document_id"] == "test-doc-123"
        assert "summary" in response_data
        assert "pages_summarized" in response_data
        assert "confidence_score" in response_data
        
        # Verify service was called
        document_service.get_document_content.assert_called_once_with("test-doc-123")

    def test_summarize_document_with_page_range(self, client, mock_services, sample_document_content):
        """Test document summarization with specific page range"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        cache_service.get_summary.return_value = None
        cache_service.set_summary.return_value = True
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "page_range": [1, 2],
            "max_length": 150,
            "min_length": 75
        }
        
        response = client.post("/api/v1/analysis/summarize", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["pages_summarized"] == [1, 2]

    def test_summarize_document_cache_hit(self, client, mock_services, sample_summarization_response):
        """Test document summarization with cache hit"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        # Mock cache hit
        cache_service.get_summary.return_value = sample_summarization_response
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "max_length": 200
        }
        
        response = client.post("/api/v1/analysis/summarize", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["summary"] == sample_summarization_response.summary
        # Should not call document service if cache hit
        document_service.get_document_content.assert_not_called()

    def test_summarize_document_not_found(self, client, mock_services):
        """Test summarization when document doesn't exist"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = None
        cache_service.get_summary.return_value = None
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "nonexistent-doc",
            "max_length": 200
        }
        
        response = client.post("/api/v1/analysis/summarize", json=request_data)
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_summarize_document_invalid_parameters(self, client, mock_services):
        """Test summarization with invalid parameters"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        # Test max_length too small
        request_data = {
            "document_id": "test-doc-123",
            "max_length": 30,  # Below minimum of 50
            "min_length": 50
        }
        
        response = client.post("/api/v1/analysis/summarize", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_summarize_document_missing_required_fields(self, client, mock_services):
        """Test summarization with missing required fields"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        # Missing document_id
        request_data = {
            "max_length": 200
        }
        
        response = client.post("/api/v1/analysis/summarize", json=request_data)
        
        assert response.status_code == 422


class TestQuestionAnswerEndpoint:
    """Test cases for question answering endpoint"""

    def test_answer_question_success(self, client, mock_services, sample_document_content):
        """Test successful question answering"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        cache_service.get_qa_result.return_value = None  # Cache miss
        cache_service.set_qa_result.return_value = True
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "question": "What is this document about?"
        }
        
        response = client.post("/api/v1/analysis/qa", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["document_id"] == "test-doc-123"
        assert response_data["question"] == "What is this document about?"
        assert "answer" in response_data
        assert "confidence_score" in response_data
        assert "source_pages" in response_data

    def test_answer_question_with_context_pages(self, client, mock_services, sample_document_content):
        """Test question answering with specific context pages"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        cache_service.get_qa_result.return_value = None
        cache_service.set_qa_result.return_value = True
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "question": "What information is on page 1?",
            "context_pages": [1]
        }
        
        response = client.post("/api/v1/analysis/qa", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["question"] == "What information is on page 1?"

    def test_answer_question_cache_hit(self, client, mock_services, sample_qa_response):
        """Test question answering with cache hit"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        # Mock cache hit
        question_hash = hashlib.md5("What is this document about?".encode()).hexdigest()
        cache_service.get_qa_result.return_value = sample_qa_response
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "question": "What is this document about?"
        }
        
        response = client.post("/api/v1/analysis/qa", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["answer"] == sample_qa_response.answer
        # Should not call document service if cache hit
        document_service.get_document_content.assert_not_called()

    def test_answer_question_document_not_found(self, client, mock_services):
        """Test question answering when document doesn't exist"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = None
        cache_service.get_qa_result.return_value = None
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "nonexistent-doc",
            "question": "What is this document about?"
        }
        
        response = client.post("/api/v1/analysis/qa", json=request_data)
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_answer_question_empty_question(self, client, mock_services):
        """Test question answering with empty question"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "question": ""
        }
        
        response = client.post("/api/v1/analysis/qa", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestTranslationEndpoint:
    """Test cases for document translation endpoint"""

    def test_translate_document_success(self, client, mock_services, sample_document_content):
        """Test successful document translation"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "target_language": "es",
            "source_language": "en"
        }
        
        response = client.post("/api/v1/analysis/translate", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["document_id"] == "test-doc-123"
        assert response_data["target_language"] == "es"
        assert response_data["source_language"] == "en"
        assert "translated_text" in response_data
        assert "pages_translated" in response_data

    def test_translate_document_auto_detect_source(self, client, mock_services, sample_document_content):
        """Test document translation with auto-detected source language"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "target_language": "fr"
            # No source_language - should auto-detect
        }
        
        response = client.post("/api/v1/analysis/translate", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["source_language"] == "en"  # Default detected
        assert response_data["target_language"] == "fr"

    def test_translate_document_with_page_range(self, client, mock_services, sample_document_content):
        """Test document translation with specific page range"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "target_language": "de",
            "page_range": [1]
        }
        
        response = client.post("/api/v1/analysis/translate", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["pages_translated"] == [1]

    def test_translate_document_not_found(self, client, mock_services):
        """Test translation when document doesn't exist"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = None
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "nonexistent-doc",
            "target_language": "es"
        }
        
        response = client.post("/api/v1/analysis/translate", json=request_data)
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_translate_document_invalid_language_code(self, client, mock_services):
        """Test translation with invalid language code"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "target_language": "invalid_language_code"
        }
        
        response = client.post("/api/v1/analysis/translate", json=request_data)
        
        # Should accept any string for now (validation would be done by translation service)
        assert response.status_code in [200, 400, 422]


class TestDataExtractionEndpoint:
    """Test cases for data extraction endpoint"""

    def test_extract_data_success(self, client, mock_services, sample_document_content):
        """Test successful data extraction"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "entity_types": ["person", "organization"],
            "extract_tables": True,
            "extract_numbers": True
        }
        
        response = client.post("/api/v1/analysis/extract", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["document_id"] == "test-doc-123"
        assert "entities" in response_data
        assert "tables" in response_data
        assert "numerical_data" in response_data

    def test_extract_data_tables_only(self, client, mock_services, sample_document_content):
        """Test data extraction with tables only"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "extract_tables": True,
            "extract_numbers": False
        }
        
        response = client.post("/api/v1/analysis/extract", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["tables"] is not None
        assert response_data["numerical_data"] is None

    def test_extract_data_numbers_only(self, client, mock_services, sample_document_content):
        """Test data extraction with numbers only"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "extract_tables": False,
            "extract_numbers": True
        }
        
        response = client.post("/api/v1/analysis/extract", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["tables"] is None
        assert response_data["numerical_data"] is not None

    def test_extract_data_specific_entities(self, client, mock_services, sample_document_content):
        """Test data extraction with specific entity types"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "entity_types": ["money", "date", "email"]
        }
        
        response = client.post("/api/v1/analysis/extract", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert "entities" in response_data

    def test_extract_data_document_not_found(self, client, mock_services):
        """Test data extraction when document doesn't exist"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = None
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "nonexistent-doc",
            "extract_tables": True
        }
        
        response = client.post("/api/v1/analysis/extract", json=request_data)
        
        assert response.status_code == 404
        assert "Document not found" in response.json()["detail"]

    def test_extract_data_invalid_entity_type(self, client, mock_services):
        """Test data extraction with invalid entity type"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id": "test-doc-123",
            "entity_types": ["invalid_entity_type"]
        }
        
        response = client.post("/api/v1/analysis/extract", json=request_data)
        
        assert response.status_code == 422  # Validation error


class TestComparisonEndpoint:
    """Test cases for document comparison endpoint"""

    def test_compare_documents_success(self, client, mock_services, sample_document_content):
        """Test successful document comparison"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        # Mock both documents exist
        document_service.get_document_content.return_value = sample_document_content
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id_1": "doc-1",
            "document_id_2": "doc-2",
            "comparison_type": "both",
            "detailed_diff": True
        }
        
        response = client.post("/api/v1/analysis/compare", json=request_data)
        
        assert response.status_code == 200
        response_data = response.json()
        
        assert response_data["document_id_1"] == "doc-1"
        assert response_data["document_id_2"] == "doc-2"
        assert "similarity_score" in response_data
        assert "total_differences" in response_data
        assert "differences" in response_data
        assert "summary" in response_data

    def test_compare_documents_text_only(self, client, mock_services, sample_document_content):
        """Test document comparison with text only"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id_1": "doc-1",
            "document_id_2": "doc-2",
            "comparison_type": "text"
        }
        
        response = client.post("/api/v1/analysis/compare", json=request_data)
        
        assert response.status_code == 200

    def test_compare_documents_visual_only(self, client, mock_services, sample_document_content):
        """Test document comparison with visual only"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id_1": "doc-1",
            "document_id_2": "doc-2",
            "comparison_type": "visual"
        }
        
        response = client.post("/api/v1/analysis/compare", json=request_data)
        
        assert response.status_code == 200

    def test_compare_documents_first_not_found(self, client, mock_services, sample_document_content):
        """Test comparison when first document doesn't exist"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        # First call returns None, second returns document
        document_service.get_document_content.side_effect = [None, sample_document_content]
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id_1": "nonexistent-doc",
            "document_id_2": "doc-2"
        }
        
        response = client.post("/api/v1/analysis/compare", json=request_data)
        
        assert response.status_code == 404
        assert "One or both documents not found" in response.json()["detail"]

    def test_compare_documents_second_not_found(self, client, mock_services, sample_document_content):
        """Test comparison when second document doesn't exist"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        # First call returns document, second returns None
        document_service.get_document_content.side_effect = [sample_document_content, None]
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id_1": "doc-1",
            "document_id_2": "nonexistent-doc"
        }
        
        response = client.post("/api/v1/analysis/compare", json=request_data)
        
        assert response.status_code == 404
        assert "One or both documents not found" in response.json()["detail"]

    def test_compare_documents_invalid_comparison_type(self, client, mock_services):
        """Test comparison with invalid comparison type"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id_1": "doc-1",
            "document_id_2": "doc-2",
            "comparison_type": "invalid_type"
        }
        
        response = client.post("/api/v1/analysis/compare", json=request_data)
        
        assert response.status_code == 422  # Validation error

    def test_compare_same_document(self, client, mock_services, sample_document_content):
        """Test comparing document with itself"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        request_data = {
            "document_id_1": "same-doc",
            "document_id_2": "same-doc"
        }
        
        response = client.post("/api/v1/analysis/compare", json=request_data)
        
        assert response.status_code == 200
        # Should handle comparison of same document gracefully


class TestAnalysisAPIIntegration:
    """Integration tests for analysis API endpoints"""

    def test_all_endpoints_require_document_id(self, client, mock_services):
        """Test that all endpoints require document_id parameter"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        # Test each endpoint without document_id
        endpoints_and_data = [
            ("/api/v1/analysis/summarize", {"max_length": 200}),
            ("/api/v1/analysis/qa", {"question": "What is this?"}),
            ("/api/v1/analysis/translate", {"target_language": "es"}),
            ("/api/v1/analysis/extract", {"extract_tables": True}),
        ]
        
        for endpoint, data in endpoints_and_data:
            response = client.post(endpoint, json=data)
            assert response.status_code == 422  # Validation error

    def test_error_handling_consistency(self, client, mock_services):
        """Test that all endpoints handle errors consistently"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        # Make service raise an exception
        document_service.get_document_content.side_effect = Exception("Service error")
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        endpoints_and_data = [
            ("/api/v1/analysis/summarize", {"document_id": "test", "max_length": 200}),
            ("/api/v1/analysis/qa", {"document_id": "test", "question": "What?"}),
            ("/api/v1/analysis/translate", {"document_id": "test", "target_language": "es"}),
            ("/api/v1/analysis/extract", {"document_id": "test", "extract_tables": True}),
        ]
        
        for endpoint, data in endpoints_and_data:
            response = client.post(endpoint, json=data)
            # Should handle errors gracefully
            assert response.status_code in [404, 500]  # Either not found or server error

    def test_response_format_consistency(self, client, mock_services, sample_document_content):
        """Test that all endpoints return consistent response formats"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        cache_service.get_summary.return_value = None
        cache_service.get_qa_result.return_value = None
        cache_service.set_summary.return_value = True
        cache_service.set_qa_result.return_value = True
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        endpoints_and_expected = [
            ("/api/v1/analysis/summarize", {"document_id": "test", "max_length": 200}, ["document_id", "summary"]),
            ("/api/v1/analysis/qa", {"document_id": "test", "question": "What?"}, ["document_id", "question", "answer"]),
            ("/api/v1/analysis/translate", {"document_id": "test", "target_language": "es"}, ["document_id", "target_language"]),
            ("/api/v1/analysis/extract", {"document_id": "test"}, ["document_id", "entities"]),
        ]
        
        for endpoint, data, expected_keys in endpoints_and_expected:
            response = client.post(endpoint, json=data)
            assert response.status_code == 200
            
            response_data = response.json()
            for key in expected_keys:
                assert key in response_data

    def test_content_type_handling(self, client, mock_services):
        """Test API content type handling"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        # Test with wrong content type
        response = client.post(
            "/api/v1/analysis/summarize",
            data="document_id=test&max_length=200",  # form data instead of JSON
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        assert response.status_code == 422  # Should expect JSON

    def test_large_request_handling(self, client, mock_services, sample_document_content):
        """Test handling of large requests"""
        document_service, cache_service = mock_services
        
        def mock_get_services():
            return document_service, cache_service
        
        document_service.get_document_content.return_value = sample_document_content
        cache_service.get_qa_result.return_value = None
        cache_service.set_qa_result.return_value = True
        
        client.app.dependency_overrides[get_services] = mock_get_services
        
        # Test with very long question
        long_question = "What is this document about? " * 1000  # Very long question
        
        request_data = {
            "document_id": "test-doc-123",
            "question": long_question
        }
        
        response = client.post("/api/v1/analysis/qa", json=request_data)
        
        # Should handle large requests gracefully
        assert response.status_code == 200