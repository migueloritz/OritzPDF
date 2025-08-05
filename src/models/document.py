from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    CSV = "csv"
    HTML = "html"
    XLSX = "xlsx"


class DocumentMetadata(BaseModel):
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    keywords: Optional[List[str]] = None
    creation_date: Optional[datetime] = None
    modification_date: Optional[datetime] = None
    pages: Optional[int] = None
    language: Optional[str] = None


class DocumentUploadRequest(BaseModel):
    filename: str
    content_type: str
    file_size: int


class DocumentUploadResponse(BaseModel):
    document_id: str
    upload_url: Optional[str] = None
    status: DocumentStatus


class Document(BaseModel):
    id: str
    filename: str
    file_type: DocumentType
    file_size: int
    status: DocumentStatus
    metadata: Optional[DocumentMetadata] = None
    upload_time: datetime
    processing_time: Optional[float] = None
    error_message: Optional[str] = None
    storage_path: Optional[str] = None


class ExtractedText(BaseModel):
    page_number: int
    text: str
    confidence: Optional[float] = None
    bbox: Optional[Dict[str, float]] = None  # bounding box coordinates


class DocumentContent(BaseModel):
    document_id: str
    full_text: str
    pages: List[ExtractedText]
    tables: Optional[List[Dict[str, Any]]] = None
    images: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[DocumentMetadata] = None