"""
Tests for document service with integration scenarios
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from pathlib import Path
from datetime import datetime
import tempfile
import uuid

from src.services.document_service import DocumentService
from src.services.storage_service import LocalStorageService
from src.services.cache_service import CacheService
from src.models.document import (
    Document, DocumentType, DocumentStatus, DocumentMetadata,
    DocumentContent, ExtractedText, DocumentUploadRequest
)


class TestDocumentServiceInitialization:
    """Test cases for DocumentService initialization"""

    def test_init_with_default_services(self):
        """Test initialization with default services"""
        service = DocumentService()
        
        assert service.storage is not None
        assert service.cache is None  # Default is None
        assert service.pdf_processor is not None

    def test_init_with_custom_services(self, local_storage_service, cache_service):
        """Test initialization with custom services"""
        service = DocumentService(
            storage_service=local_storage_service,
            cache_service=cache_service
        )
        
        assert service.storage is local_storage_service
        assert service.cache is cache_service

    def test_pdf_processor_uses_storage_service(self, local_storage_service):
        """Test that PDF processor is initialized with storage service"""
        service = DocumentService(storage_service=local_storage_service)
        
        assert service.pdf_processor.storage is local_storage_service


class TestDocumentServiceValidation:
    """Test cases for document validation"""

    @pytest.fixture
    def service(self, local_storage_service, cache_service):
        """Create document service for testing"""
        return DocumentService(
            storage_service=local_storage_service,
            cache_service=cache_service
        )

    @pytest.mark.asyncio
    async def test_validate_upload_success(self, service):
        """Test successful upload validation"""
        # Should not raise any exception
        await service.validate_upload("document.pdf", 1024, "application/pdf")

    @pytest.mark.asyncio
    async def test_validate_upload_file_too_large(self, service):
        """Test validation fails for files that are too large"""
        large_size = 50 * 1024 * 1024  # 50MB (assuming max is smaller)
        
        with pytest.raises(ValueError, match="File size exceeds maximum"):
            await service.validate_upload("document.pdf", large_size, "application/pdf")

    @pytest.mark.asyncio
    async def test_validate_upload_unsupported_format(self, service):
        """Test validation fails for unsupported file formats"""
        with pytest.raises(ValueError, match="File type 'exe' not supported"):
            await service.validate_upload("malware.exe", 1024, "application/octet-stream")

    @pytest.mark.asyncio
    async def test_validate_upload_content_type_mismatch(self, service):
        """Test validation with mismatched content type (should warn but not fail)"""
        # This should log a warning but not raise an exception
        await service.validate_upload("document.pdf", 1024, "text/plain")

    @pytest.mark.asyncio
    async def test_validate_upload_various_formats(self, service):
        """Test validation for all supported formats"""
        test_cases = [
            ("document.pdf", "application/pdf"),
            ("document.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
            ("document.txt", "text/plain"),
            ("data.csv", "text/csv"),
            ("page.html", "text/html"),
            ("spreadsheet.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        ]
        
        for filename, content_type in test_cases:
            await service.validate_upload(filename, 1024, content_type)

    @pytest.mark.asyncio
    async def test_validate_upload_no_extension(self, service):
        """Test validation for file without extension"""
        with pytest.raises(ValueError, match="File type '' not supported"):
            await service.validate_upload("document", 1024, "application/pdf")

    @pytest.mark.asyncio
    async def test_validate_upload_case_insensitive_extension(self, service):
        """Test validation handles case-insensitive extensions"""
        await service.validate_upload("document.PDF", 1024, "application/pdf")
        await service.validate_upload("document.Pdf", 1024, "application/pdf")


class TestDocumentServiceCreation:
    """Test cases for document creation"""

    @pytest.fixture
    def service(self, local_storage_service, cache_service):
        """Create document service for testing"""
        return DocumentService(
            storage_service=local_storage_service,
            cache_service=cache_service
        )

    @pytest.mark.asyncio
    async def test_create_document_success(self, service, sample_upload_request):
        """Test successful document creation"""
        document = await service.create_document(sample_upload_request)
        
        assert document.id is not None
        assert len(document.id) > 0
        assert document.filename == sample_upload_request.filename
        assert document.file_type == DocumentType.PDF
        assert document.file_size == sample_upload_request.file_size
        assert document.status == DocumentStatus.PENDING
        assert document.upload_time is not None
        assert isinstance(document.upload_time, datetime)

    @pytest.mark.asyncio
    async def test_create_document_different_types(self, service):
        """Test document creation for different file types"""
        test_cases = [
            ("test.pdf", "application/pdf", DocumentType.PDF),
            ("test.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", DocumentType.DOCX),
            ("test.txt", "text/plain", DocumentType.TXT),
            ("test.csv", "text/csv", DocumentType.CSV),
            ("test.html", "text/html", DocumentType.HTML),
            ("test.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", DocumentType.XLSX)
        ]
        
        for filename, content_type, expected_type in test_cases:
            upload_request = DocumentUploadRequest(
                filename=filename,
                content_type=content_type,
                file_size=1024
            )
            
            document = await service.create_document(upload_request)
            assert document.file_type == expected_type

    @pytest.mark.asyncio
    async def test_create_document_unique_ids(self, service, sample_upload_request):
        """Test that created documents have unique IDs"""
        doc1 = await service.create_document(sample_upload_request)
        doc2 = await service.create_document(sample_upload_request)
        
        assert doc1.id != doc2.id

    @pytest.mark.asyncio
    async def test_create_document_validation_error(self, service):
        """Test document creation with validation error"""
        invalid_request = DocumentUploadRequest(
            filename="malware.exe",
            content_type="application/octet-stream",
            file_size=1024
        )
        
        with pytest.raises(ValueError, match="not supported"):
            await service.create_document(invalid_request)


class TestDocumentServiceProcessing:
    """Test cases for document processing"""

    @pytest.fixture
    def service(self, local_storage_service, cache_service, mock_pdf_processor):
        """Create document service with mocked PDF processor"""
        service = DocumentService(
            storage_service=local_storage_service,
            cache_service=cache_service
        )
        service.pdf_processor = mock_pdf_processor
        return service

    @pytest.mark.asyncio
    async def test_process_document_pdf_success(self, service, sample_pdf_content):
        """Test successful PDF document processing"""
        document_id = str(uuid.uuid4())
        
        document = await service.process_document(document_id, sample_pdf_content)
        
        assert document.id == document_id
        assert document.status == DocumentStatus.COMPLETED
        assert document.processing_time is not None
        assert document.processing_time > 0
        assert document.storage_path is not None
        
        # Verify PDF processor was called
        service.pdf_processor.process_document.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_caches_content(self, service, sample_pdf_content, cache_service):
        """Test that processed document content is cached"""
        document_id = str(uuid.uuid4())
        
        await service.process_document(document_id, sample_pdf_content)
        
        # Verify cache was called
        cache_service.set_document_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_without_cache(self, local_storage_service, mock_pdf_processor, sample_pdf_content):
        """Test document processing without cache service"""
        service = DocumentService(
            storage_service=local_storage_service,
            cache_service=None  # No cache
        )
        service.pdf_processor = mock_pdf_processor
        
        document_id = str(uuid.uuid4())
        
        document = await service.process_document(document_id, sample_pdf_content)
        
        assert document.status == DocumentStatus.COMPLETED
        # Should not crash without cache service

    @pytest.mark.asyncio
    async def test_process_document_pdf_processor_error(self, service, sample_pdf_content):
        """Test document processing when PDF processor fails"""
        document_id = str(uuid.uuid4())
        service.pdf_processor.process_document.side_effect = Exception("PDF processing failed")
        
        document = await service.process_document(document_id, sample_pdf_content)
        
        assert document.status == DocumentStatus.FAILED
        assert document.error_message == "PDF processing failed"
        assert document.processing_time is not None

    @pytest.mark.asyncio
    async def test_process_document_storage_error(self, cache_service, mock_pdf_processor, sample_pdf_content):
        """Test document processing when storage fails"""
        # Create mock storage that fails
        mock_storage = AsyncMock()
        mock_storage.save_file.side_effect = Exception("Storage failed")
        
        service = DocumentService(
            storage_service=mock_storage,
            cache_service=cache_service
        )
        service.pdf_processor = mock_pdf_processor
        
        document_id = str(uuid.uuid4())
        
        document = await service.process_document(document_id, sample_pdf_content)
        
        assert document.status == DocumentStatus.FAILED
        assert "Storage failed" in document.error_message

    @pytest.mark.asyncio
    async def test_process_document_unsupported_type(self, service, sample_docx_content):
        """Test processing unsupported document type (DOCX not implemented)"""
        document_id = str(uuid.uuid4())
        
        # Mock document creation to return DOCX type
        with patch.object(service, 'create_document') as mock_create:
            mock_doc = Document(
                id=document_id,
                filename="test.docx",
                file_type=DocumentType.DOCX,
                file_size=len(sample_docx_content),
                status=DocumentStatus.PENDING,
                upload_time=datetime.utcnow()
            )
            mock_create.return_value = mock_doc
            
            document = await service.process_document(document_id, sample_docx_content)
            
            assert document.status == DocumentStatus.FAILED
            assert "Unsupported file type" in document.error_message

    @pytest.mark.asyncio
    async def test_process_document_metadata_extraction(self, service, sample_pdf_content):
        """Test that document metadata is extracted and stored"""
        document_id = str(uuid.uuid4())
        
        # Configure mock to return metadata
        mock_content = DocumentContent(
            document_id=document_id,
            full_text="Sample text",
            pages=[],
            tables=[],
            images=[],
            metadata=DocumentMetadata(
                title="Test Document",
                author="Test Author",
                pages=1
            )
        )
        service.pdf_processor.process_document.return_value = mock_content
        
        document = await service.process_document(document_id, sample_pdf_content)
        
        assert document.metadata is not None
        assert document.metadata.title == "Test Document"
        assert document.metadata.author == "Test Author"


class TestDocumentServiceRetrieval:
    """Test cases for document retrieval"""

    @pytest.fixture
    def service(self, local_storage_service, cache_service, mock_pdf_processor):
        """Create document service with mocked components"""
        service = DocumentService(
            storage_service=local_storage_service,
            cache_service=cache_service
        )
        service.pdf_processor = mock_pdf_processor
        return service

    @pytest.mark.asyncio
    async def test_get_document_content_from_cache(self, service, sample_document_content, cache_service):
        """Test retrieving document content from cache"""
        document_id = "test-doc-123"
        cache_service.get_document_content.return_value = sample_document_content
        
        content = await service.get_document_content(document_id)
        
        assert content is not None
        assert content.document_id == document_id
        cache_service.get_document_content.assert_called_once_with(document_id)

    @pytest.mark.asyncio
    async def test_get_document_content_cache_miss(self, service, cache_service):
        """Test handling cache miss for document content"""
        document_id = "test-doc-123"
        cache_service.get_document_content.return_value = None
        
        # Mock get_document to return None (not implemented in service)
        with patch.object(service, 'get_document', return_value=None):
            content = await service.get_document_content(document_id)
            
            assert content is None

    @pytest.mark.asyncio
    async def test_get_document_content_reprocess_from_storage(self, service, sample_pdf_content, cache_service):
        """Test reprocessing document from storage when not in cache"""
        document_id = "test-doc-123"
        
        # Cache miss
        cache_service.get_document_content.return_value = None
        
        # Mock document with storage path
        mock_document = Document(
            id=document_id,
            filename="test.pdf",
            file_type=DocumentType.PDF,
            file_size=len(sample_pdf_content),
            status=DocumentStatus.COMPLETED,
            upload_time=datetime.utcnow(),
            storage_path="/path/to/test.pdf"
        )
        
        with patch.object(service, 'get_document', return_value=mock_document), \
             patch.object(service.storage, 'get_file', return_value=sample_pdf_content), \
             patch('tempfile.NamedTemporaryFile') as mock_temp:
            
            # Mock temporary file
            mock_temp_file = Mock()
            mock_temp_file.name = "/tmp/test.pdf"
            mock_temp_file.__enter__ = Mock(return_value=mock_temp_file)
            mock_temp_file.__exit__ = Mock(return_value=None)
            mock_temp.return_value = mock_temp_file
            
            # Mock Path.unlink
            with patch('pathlib.Path.unlink'):
                content = await service.get_document_content(document_id)
                
                assert content is not None
                service.pdf_processor.process_document.assert_called_once()
                cache_service.set_document_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_document_content_no_storage_path(self, service, cache_service):
        """Test get_document_content when document has no storage path"""
        document_id = "test-doc-123"
        
        cache_service.get_document_content.return_value = None
        
        mock_document = Document(
            id=document_id,
            filename="test.pdf",
            file_type=DocumentType.PDF,
            file_size=1024,
            status=DocumentStatus.FAILED,
            upload_time=datetime.utcnow(),
            storage_path=None  # No storage path
        )
        
        with patch.object(service, 'get_document', return_value=mock_document):
            content = await service.get_document_content(document_id)
            
            assert content is None

    @pytest.mark.asyncio
    async def test_get_document_placeholder(self, service):
        """Test get_document method (currently returns None - placeholder)"""
        document_id = "test-doc-123"
        
        result = await service.get_document(document_id)
        
        # Currently returns None as it's not implemented
        assert result is None


class TestDocumentServiceDeletion:
    """Test cases for document deletion"""

    @pytest.fixture
    def service(self, local_storage_service, cache_service):
        """Create document service for testing"""
        return DocumentService(
            storage_service=local_storage_service,
            cache_service=cache_service
        )

    @pytest.mark.asyncio
    async def test_delete_document_success(self, service, local_storage_service, cache_service):
        """Test successful document deletion"""
        document_id = "test-doc-123"
        
        mock_document = Document(
            id=document_id,
            filename="test.pdf",
            file_type=DocumentType.PDF,
            file_size=1024,
            status=DocumentStatus.COMPLETED,
            upload_time=datetime.utcnow(),
            storage_path="/path/to/test.pdf"
        )
        
        with patch.object(service, 'get_document', return_value=mock_document):
            result = await service.delete_document(document_id)
            
            assert result is True
            local_storage_service.delete_file.assert_called_once_with("/path/to/test.pdf")
            cache_service.delete_document.assert_called_once_with(document_id)

    @pytest.mark.asyncio
    async def test_delete_document_not_found(self, service):
        """Test deleting non-existent document"""
        document_id = "nonexistent-doc"
        
        with patch.object(service, 'get_document', return_value=None):
            result = await service.delete_document(document_id)
            
            assert result is False

    @pytest.mark.asyncio
    async def test_delete_document_no_storage_path(self, service, cache_service):
        """Test deleting document without storage path"""
        document_id = "test-doc-123"
        
        mock_document = Document(
            id=document_id,
            filename="test.pdf",
            file_type=DocumentType.PDF,
            file_size=1024,
            status=DocumentStatus.FAILED,
            upload_time=datetime.utcnow(),
            storage_path=None
        )
        
        with patch.object(service, 'get_document', return_value=mock_document):
            result = await service.delete_document(document_id)
            
            assert result is True
            cache_service.delete_document.assert_called_once_with(document_id)

    @pytest.mark.asyncio
    async def test_delete_document_without_cache(self, local_storage_service):
        """Test document deletion without cache service"""
        service = DocumentService(
            storage_service=local_storage_service,
            cache_service=None
        )
        
        document_id = "test-doc-123"
        mock_document = Document(
            id=document_id,
            filename="test.pdf",
            file_type=DocumentType.PDF,
            file_size=1024,
            status=DocumentStatus.COMPLETED,
            upload_time=datetime.utcnow(),
            storage_path="/path/to/test.pdf"
        )
        
        with patch.object(service, 'get_document', return_value=mock_document):
            result = await service.delete_document(document_id)
            
            assert result is True
            local_storage_service.delete_file.assert_called_once_with("/path/to/test.pdf")


class TestDocumentServiceIntegration:
    """Integration tests for document service"""

    @pytest.mark.asyncio
    async def test_full_document_lifecycle(self, temp_dir, sample_pdf_content):
        """Test complete document lifecycle from creation to deletion"""
        # Set up services
        storage = LocalStorageService(str(temp_dir))
        cache = AsyncMock(spec=CacheService)
        cache.set_document_content.return_value = True
        cache.delete_document.return_value = True
        
        service = DocumentService(storage_service=storage, cache_service=cache)
        
        # Mock PDF processor
        mock_content = DocumentContent(
            document_id="",  # Will be set during processing
            full_text="Sample extracted text",
            pages=[ExtractedText(page_number=1, text="Sample text", confidence=0.95)],
            tables=[],
            images=[],
            metadata=DocumentMetadata(title="Test PDF", pages=1)
        )
        
        with patch.object(service.pdf_processor, 'process_document', return_value=mock_content):
            # 1. Create document
            upload_request = DocumentUploadRequest(
                filename="test.pdf",
                content_type="application/pdf",
                file_size=len(sample_pdf_content)
            )
            
            document = await service.create_document(upload_request)
            assert document.status == DocumentStatus.PENDING
            
            # 2. Process document
            processed_doc = await service.process_document(document.id, sample_pdf_content)
            assert processed_doc.status == DocumentStatus.COMPLETED
            assert processed_doc.storage_path is not None
            
            # Verify file was saved to storage
            assert Path(processed_doc.storage_path).exists()
            
            # 3. Delete document
            with patch.object(service, 'get_document', return_value=processed_doc):
                delete_result = await service.delete_document(document.id)
                assert delete_result is True
                
                # Verify file was deleted from storage
                assert not Path(processed_doc.storage_path).exists()

    @pytest.mark.asyncio
    async def test_concurrent_document_processing(self, temp_dir, sample_pdf_content):
        """Test concurrent processing of multiple documents"""
        import asyncio
        
        storage = LocalStorageService(str(temp_dir))
        cache = AsyncMock(spec=CacheService)
        service = DocumentService(storage_service=storage, cache_service=cache)
        
        # Mock PDF processor
        async def mock_process_document(file_path: str, document_id: str):
            # Simulate some processing time
            await asyncio.sleep(0.1)
            return DocumentContent(
                document_id=document_id,
                full_text=f"Content for {document_id}",
                pages=[],
                tables=[],
                images=[],
                metadata=DocumentMetadata()
            )
        
        service.pdf_processor.process_document = mock_process_document
        
        # Process multiple documents concurrently
        tasks = []
        for i in range(5):
            document_id = f"doc-{i}"
            task = service.process_document(document_id, sample_pdf_content)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        # All should complete successfully
        for i, result in enumerate(results):
            assert result.status == DocumentStatus.COMPLETED
            assert result.id == f"doc-{i}"

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, local_storage_service, cache_service):
        """Test various error recovery scenarios"""
        service = DocumentService(
            storage_service=local_storage_service,
            cache_service=cache_service
        )
        
        document_id = "error-test-doc"
        test_content = b"test content"
        
        # Test 1: Storage succeeds, PDF processing fails
        local_storage_service.save_file.return_value = "/path/to/file.pdf"
        service.pdf_processor.process_document.side_effect = Exception("Processing failed")
        
        document = await service.process_document(document_id, test_content)
        
        assert document.status == DocumentStatus.FAILED
        assert "Processing failed" in document.error_message
        
        # Test 2: Storage fails immediately
        local_storage_service.save_file.side_effect = Exception("Storage failed")
        
        document = await service.process_document(document_id, test_content)
        
        assert document.status == DocumentStatus.FAILED
        assert "Storage failed" in document.error_message

    @pytest.mark.asyncio
    async def test_performance_with_large_content(self, local_storage_service, cache_service):
        """Test processing performance with large content"""
        import time
        
        service = DocumentService(
            storage_service=local_storage_service,
            cache_service=cache_service
        )
        
        # Create large content (1MB)
        large_content = b"x" * (1024 * 1024)
        document_id = "large-doc"
        
        # Mock quick processing
        mock_content = DocumentContent(
            document_id=document_id,
            full_text="Large document processed",
            pages=[],
            tables=[],
            images=[],
            metadata=DocumentMetadata()
        )
        
        with patch.object(service.pdf_processor, 'process_document', return_value=mock_content):
            start_time = time.time()
            document = await service.process_document(document_id, large_content)
            end_time = time.time()
            
            assert document.status == DocumentStatus.COMPLETED
            # Should complete reasonably quickly (storage + processing)
            assert end_time - start_time < 5.0  # Less than 5 seconds