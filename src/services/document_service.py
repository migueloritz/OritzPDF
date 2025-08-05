import uuid
import time
import logging
from typing import Optional, BinaryIO
from datetime import datetime
from pathlib import Path

from src.models.document import (
    Document, DocumentStatus, DocumentType,
    DocumentUploadRequest, DocumentUploadResponse,
    DocumentContent
)
from src.services.storage_service import StorageService, get_storage_service
from src.services.pdf_processor import ComprehensivePDFProcessor
from src.services.cache_service import CacheService
from src.config import settings

logger = logging.getLogger(__name__)


class DocumentService:
    """Main service for document operations"""
    
    def __init__(self, 
                 storage_service: Optional[StorageService] = None,
                 cache_service: Optional[CacheService] = None):
        self.storage = storage_service or get_storage_service()
        self.cache = cache_service
        self.pdf_processor = ComprehensivePDFProcessor(self.storage)
        # Add more processors for other formats later
    
    async def validate_upload(self, filename: str, file_size: int, content_type: str) -> None:
        """Validate file upload parameters"""
        # Check file size
        if file_size > settings.max_file_size_bytes:
            raise ValueError(f"File size exceeds maximum allowed size of {settings.MAX_FILE_SIZE_MB}MB")
        
        # Check file type
        extension = Path(filename).suffix.lower()[1:]  # Remove the dot
        if extension not in settings.supported_formats_list:
            raise ValueError(f"File type '{extension}' not supported. Supported formats: {settings.SUPPORTED_FORMATS}")
        
        # Additional content type validation
        expected_types = {
            'pdf': ['application/pdf'],
            'docx': ['application/vnd.openxmlformats-officedocument.wordprocessingml.document'],
            'txt': ['text/plain'],
            'csv': ['text/csv', 'application/csv'],
            'html': ['text/html'],
            'xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet']
        }
        
        if extension in expected_types and content_type not in expected_types[extension]:
            logger.warning(f"Unexpected content type {content_type} for {extension} file")
    
    async def create_document(self, upload_request: DocumentUploadRequest) -> Document:
        """Create a new document record"""
        # Validate upload
        await self.validate_upload(
            upload_request.filename,
            upload_request.file_size,
            upload_request.content_type
        )
        
        # Determine file type
        extension = Path(upload_request.filename).suffix.lower()[1:]
        file_type = DocumentType(extension)
        
        # Create document record
        document = Document(
            id=str(uuid.uuid4()),
            filename=upload_request.filename,
            file_type=file_type,
            file_size=upload_request.file_size,
            status=DocumentStatus.PENDING,
            upload_time=datetime.utcnow()
        )
        
        # Store in database (placeholder - implement with SQLAlchemy)
        # await self.db.save_document(document)
        
        return document
    
    async def process_document(self, document_id: str, file_content: bytes) -> Document:
        """Process uploaded document"""
        start_time = time.time()
        
        try:
            # Get document record
            # document = await self.db.get_document(document_id)
            # For now, create a placeholder
            document = Document(
                id=document_id,
                filename="temp.pdf",
                file_type=DocumentType.PDF,
                file_size=len(file_content),
                status=DocumentStatus.PROCESSING,
                upload_time=datetime.utcnow()
            )
            
            # Update status to processing
            document.status = DocumentStatus.PROCESSING
            # await self.db.update_document(document)
            
            # Save file to storage
            storage_path = await self.storage.save_file(file_content, document.filename)
            document.storage_path = storage_path
            
            # Process based on file type
            if document.file_type == DocumentType.PDF:
                content = await self.pdf_processor.process_document(storage_path, document_id)
                
                # Extract metadata
                if content.metadata:
                    document.metadata = content.metadata
                
                # Cache the processed content
                if self.cache:
                    await self.cache.set_document_content(document_id, content)
            
            elif document.file_type == DocumentType.DOCX:
                # TODO: Implement DOCX processor
                pass
            
            elif document.file_type == DocumentType.TXT:
                # TODO: Implement TXT processor
                pass
            
            else:
                raise ValueError(f"Unsupported file type: {document.file_type}")
            
            # Update document status
            document.status = DocumentStatus.COMPLETED
            document.processing_time = time.time() - start_time
            
            logger.info(f"Document {document_id} processed successfully in {document.processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Document processing failed for {document_id}: {e}")
            document.status = DocumentStatus.FAILED
            document.error_message = str(e)
            document.processing_time = time.time() - start_time
        
        # Update in database
        # await self.db.update_document(document)
        
        return document
    
    async def get_document(self, document_id: str) -> Optional[Document]:
        """Get document by ID"""
        # TODO: Implement database retrieval
        # return await self.db.get_document(document_id)
        pass
    
    async def get_document_content(self, document_id: str) -> Optional[DocumentContent]:
        """Get processed document content"""
        # Check cache first
        if self.cache:
            cached_content = await self.cache.get_document_content(document_id)
            if cached_content:
                logger.info(f"Retrieved document {document_id} from cache")
                return cached_content
        
        # Get from storage and reprocess if needed
        document = await self.get_document(document_id)
        if not document or not document.storage_path:
            return None
        
        # Reprocess the document
        file_content = await self.storage.get_file(document.storage_path)
        
        # Save to temporary file for processing
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=f".{document.file_type.value}", delete=False) as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        try:
            if document.file_type == DocumentType.PDF:
                content = await self.pdf_processor.process_document(tmp_path, document_id)
                
                # Cache the result
                if self.cache:
                    await self.cache.set_document_content(document_id, content)
                
                return content
        finally:
            # Clean up temp file
            Path(tmp_path).unlink(missing_ok=True)
        
        return None
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document and its associated data"""
        document = await self.get_document(document_id)
        if not document:
            return False
        
        # Delete from storage
        if document.storage_path:
            await self.storage.delete_file(document.storage_path)
        
        # Delete from cache
        if self.cache:
            await self.cache.delete_document(document_id)
        
        # Delete from database
        # await self.db.delete_document(document_id)
        
        logger.info(f"Document {document_id} deleted successfully")
        return True