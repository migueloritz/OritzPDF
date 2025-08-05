from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Optional, List
import logging

from src.models.document import (
    Document, DocumentUploadRequest, DocumentUploadResponse,
    DocumentContent, DocumentStatus
)
from src.services.document_service import DocumentService
from src.services.cache_service import get_cache_service
from src.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()

# Dependency injection
async def get_document_service() -> DocumentService:
    cache_service = get_cache_service()
    await cache_service.connect()
    return DocumentService(cache_service=cache_service)


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    service: DocumentService = Depends(get_document_service)
):
    """
    Upload a document for processing
    
    Supported formats: PDF, DOCX, TXT, CSV, HTML, XLSX
    Maximum file size: 32MB
    """
    try:
        # Validate file input
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Get file size from file object or estimate
        file_size = 0
        if hasattr(file, 'size') and file.size is not None:
            file_size = file.size
        else:
            # Read file to get size (this consumes the stream)
            file_content = await file.read()
            file_size = len(file_content)
            # Reset file pointer for later reading
            await file.seek(0)
        
        # Validate content type
        content_type = file.content_type or "application/octet-stream"
        
        # Create upload request
        upload_request = DocumentUploadRequest(
            filename=file.filename,
            content_type=content_type,
            file_size=file_size
        )
        
        # Create document record
        document = await service.create_document(upload_request)
        
        # Read file content if not already read
        if 'file_content' not in locals():
            file_content = await file.read()
        
        # Validate file content
        if not file_content:
            raise HTTPException(status_code=400, detail="File content is empty")
        
        # Process document in background
        background_tasks.add_task(
            service.process_document,
            document.id,
            file_content
        )
        
        return DocumentUploadResponse(
            document_id=document.id,
            status=document.status
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        # Ensure file is properly closed
        if hasattr(file, 'close'):
            try:
                await file.close()
            except Exception as close_error:
                logger.warning(f"Error closing uploaded file: {close_error}")


@router.get("/{document_id}", response_model=Document)
async def get_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Get document metadata and status"""
    document = await service.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return document


@router.get("/{document_id}/content", response_model=DocumentContent)
async def get_document_content(
    document_id: str,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    service: DocumentService = Depends(get_document_service)
):
    """
    Get extracted document content
    
    Optional parameters:
    - page_start: Starting page number (1-indexed)
    - page_end: Ending page number (inclusive)
    """
    # Validate document ID
    if not document_id or not isinstance(document_id, str):
        raise HTTPException(status_code=400, detail="Invalid document ID")
    
    # Validate page parameters
    if page_start is not None and page_start < 1:
        raise HTTPException(status_code=400, detail="page_start must be greater than 0")
    
    if page_end is not None and page_end < 1:
        raise HTTPException(status_code=400, detail="page_end must be greater than 0")
    
    if page_start is not None and page_end is not None and page_start > page_end:
        raise HTTPException(status_code=400, detail="page_start cannot be greater than page_end")
    
    content = await service.get_document_content(document_id)
    
    if not content:
        raise HTTPException(status_code=404, detail="Document content not found")
    
    # Filter pages if requested
    if page_start or page_end:
        filtered_pages = []
        for page in content.pages:
            if page_start and page.page_number < page_start:
                continue
            if page_end and page.page_number > page_end:
                continue
            filtered_pages.append(page)
        
        content.pages = filtered_pages
        
        # Reconstruct full text from filtered pages
        if filtered_pages:
            content.full_text = "\n".join([p.text for p in filtered_pages])
        else:
            content.full_text = ""
    
    return content


@router.get("/{document_id}/text")
async def get_document_text(
    document_id: str,
    page_start: Optional[int] = None,
    page_end: Optional[int] = None,
    service: DocumentService = Depends(get_document_service)
):
    """Get plain text content of document"""
    content = await service.get_document_content(document_id)
    
    if not content:
        raise HTTPException(status_code=404, detail="Document content not found")
    
    # Filter pages if requested
    if page_start or page_end:
        text_parts = []
        for page in content.pages:
            if page_start and page.page_number < page_start:
                continue
            if page_end and page.page_number > page_end:
                continue
            text_parts.append(page.text)
        
        text = "\n".join(text_parts)
    else:
        text = content.full_text
    
    return {"text": text}


@router.get("/{document_id}/metadata")
async def get_document_metadata(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Get document metadata"""
    content = await service.get_document_content(document_id)
    
    if not content or not content.metadata:
        raise HTTPException(status_code=404, detail="Document metadata not found")
    
    return content.metadata


@router.get("/{document_id}/tables")
async def get_document_tables(
    document_id: str,
    page: Optional[int] = None,
    service: DocumentService = Depends(get_document_service)
):
    """Get extracted tables from document"""
    content = await service.get_document_content(document_id)
    
    if not content:
        raise HTTPException(status_code=404, detail="Document content not found")
    
    tables = content.tables or []
    
    # Filter by page if requested
    if page:
        tables = [t for t in tables if t.get("page") == page]
    
    return {"tables": tables}


@router.delete("/{document_id}")
async def delete_document(
    document_id: str,
    service: DocumentService = Depends(get_document_service)
):
    """Delete a document and all associated data"""
    success = await service.delete_document(document_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {"message": "Document deleted successfully"}


@router.get("/", response_model=List[Document])
async def list_documents(
    status: Optional[DocumentStatus] = None,
    limit: int = 20,
    offset: int = 0,
    service: DocumentService = Depends(get_document_service)
):
    """
    List documents with optional filtering
    
    Parameters:
    - status: Filter by document status
    - limit: Maximum number of documents to return (default: 20, max: 100)
    - offset: Number of documents to skip
    """
    # TODO: Implement pagination and filtering in document service
    return []


@router.post("/{document_id}/reprocess")
async def reprocess_document(
    document_id: str,
    background_tasks: BackgroundTasks,
    service: DocumentService = Depends(get_document_service)
):
    """Reprocess a document"""
    document = await service.get_document(document_id)
    
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")
    
    if not document.storage_path:
        raise HTTPException(status_code=400, detail="Document file not found in storage")
    
    # Get file content from storage
    file_content = await service.storage.get_file(document.storage_path)
    
    # Reprocess in background
    background_tasks.add_task(
        service.process_document,
        document_id,
        file_content
    )
    
    return {"message": "Document reprocessing started", "document_id": document_id}