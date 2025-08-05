import fitz  # PyMuPDF
from pypdf import PdfReader
import pdfplumber
from typing import List, Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
import logging
from pathlib import Path
from datetime import datetime

from src.models.document import DocumentContent, ExtractedText, DocumentMetadata
from src.services.storage_service import StorageService

logger = logging.getLogger(__name__)


class PDFProcessor(ABC):
    """Abstract base class for PDF processors"""
    
    @abstractmethod
    async def extract_text(self, file_path: str) -> Tuple[str, List[ExtractedText]]:
        """Extract text from PDF. Returns full text and page-wise text"""
        pass
    
    @abstractmethod
    async def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract metadata from PDF"""
        pass
    
    @abstractmethod
    async def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF"""
        pass
    
    @abstractmethod
    async def extract_images(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract images from PDF"""
        pass


class PyMuPDFProcessor(PDFProcessor):
    """High-performance PDF processor using PyMuPDF"""
    
    async def extract_text(self, file_path: str) -> Tuple[str, List[ExtractedText]]:
        """Extract text using PyMuPDF"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        doc = None
        try:
            doc = fitz.open(file_path)
            full_text = ""
            pages = []
            
            for page_num, page in enumerate(doc):
                try:
                    text = page.get_text()
                    if text:  # Only add non-empty text
                        full_text += text + "\n"
                    
                    # Get text with bounding boxes
                    blocks = page.get_text("dict")
                    page_data = ExtractedText(
                        page_number=page_num + 1,
                        text=text,
                        confidence=0.95  # PyMuPDF doesn't provide confidence
                    )
                    pages.append(page_data)
                except Exception as page_error:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {page_error}")
                    # Add empty page data to maintain page numbering
                    pages.append(ExtractedText(
                        page_number=page_num + 1,
                        text="",
                        confidence=0.0
                    ))
            
            return full_text.strip(), pages
            
        except Exception as e:
            logger.error(f"PyMuPDF text extraction failed for {file_path}: {e}")
            raise
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as close_error:
                    logger.warning(f"Error closing PyMuPDF document: {close_error}")
    
    async def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract metadata using PyMuPDF"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        doc = None
        try:
            doc = fitz.open(file_path)
            metadata = doc.metadata
            
            # Validate and parse metadata safely
            title = metadata.get("title", "").strip() if metadata.get("title") else None
            author = metadata.get("author", "").strip() if metadata.get("author") else None
            subject = metadata.get("subject", "").strip() if metadata.get("subject") else None
            keywords_str = metadata.get("keywords", "")
            keywords = [k.strip() for k in keywords_str.split(",") if k.strip()] if keywords_str else None
            
            result = DocumentMetadata(
                title=title,
                author=author,
                subject=subject,
                keywords=keywords,
                creation_date=self._parse_date(metadata.get("creationDate")),
                modification_date=self._parse_date(metadata.get("modDate")),
                pages=doc.page_count
            )
            
            return result
            
        except Exception as e:
            logger.error(f"PyMuPDF metadata extraction failed for {file_path}: {e}")
            raise
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as close_error:
                    logger.warning(f"Error closing PyMuPDF document: {close_error}")
    
    async def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables using PyMuPDF"""
        tables = []
        try:
            doc = fitz.open(file_path)
            
            for page_num, page in enumerate(doc):
                # PyMuPDF doesn't have built-in table extraction
                # Would need to implement custom logic or use pdfplumber
                pass
            
            doc.close()
            return tables
            
        except Exception as e:
            logger.error(f"PyMuPDF table extraction failed: {e}")
            return []
    
    async def extract_images(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract images using PyMuPDF"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        images = []
        doc = None
        try:
            doc = fitz.open(file_path)
            
            for page_num, page in enumerate(doc):
                try:
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        xref = img[0]
                        try:
                            pix = fitz.Pixmap(doc, xref)
                            
                            images.append({
                                "page": page_num + 1,
                                "index": img_index,
                                "width": pix.width,
                                "height": pix.height,
                                "colorspace": pix.colorspace.name if pix.colorspace else "unknown",
                                "xref": xref
                            })
                            
                            # Free memory immediately
                            pix = None
                        except Exception as img_error:
                            logger.warning(f"Failed to process image {img_index} on page {page_num + 1}: {img_error}")
                            continue
                            
                except Exception as page_error:
                    logger.warning(f"Failed to extract images from page {page_num + 1}: {page_error}")
                    continue
            
            return images
            
        except Exception as e:
            logger.error(f"PyMuPDF image extraction failed for {file_path}: {e}")
            return []  # Return empty list instead of raising for non-critical errors
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as close_error:
                    logger.warning(f"Error closing PyMuPDF document: {close_error}")
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse PDF date string safely"""
        if not date_str or not isinstance(date_str, str):
            return None
        try:
            # PDF date format: D:YYYYMMDDHHmmSSOHH'mm
            if date_str.startswith("D:"):
                date_str = date_str[2:]
            
            # Handle different date format lengths
            if len(date_str) >= 8:
                # Try basic YYYYMMDD format first
                date_part = date_str[:8]
                if date_part.isdigit():
                    return datetime.strptime(date_part, "%Y%m%d")
            
            # If that fails, try more flexible parsing
            return None
        except (ValueError, TypeError) as e:
            logger.debug(f"Could not parse date string '{date_str}': {e}")
            return None


class PyPDFProcessor(PDFProcessor):
    """PDF processor using pypdf (pure Python)"""
    
    async def extract_text(self, file_path: str) -> Tuple[str, List[ExtractedText]]:
        """Extract text using pypdf"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        try:
            reader = PdfReader(file_path)
            full_text = ""
            pages = []
            
            for page_num, page in enumerate(reader.pages):
                try:
                    text = page.extract_text()
                    if text:  # Only add non-empty text
                        full_text += text + "\n"
                    
                    page_data = ExtractedText(
                        page_number=page_num + 1,
                        text=text,
                        confidence=0.90  # pypdf doesn't provide confidence
                    )
                    pages.append(page_data)
                except Exception as page_error:
                    logger.warning(f"Failed to extract text from page {page_num + 1}: {page_error}")
                    # Add empty page data to maintain page numbering
                    pages.append(ExtractedText(
                        page_number=page_num + 1,
                        text="",
                        confidence=0.0
                    ))
            
            return full_text.strip(), pages
            
        except Exception as e:
            logger.error(f"pypdf text extraction failed for {file_path}: {e}")
            raise
    
    async def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract metadata using pypdf"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        try:
            reader = PdfReader(file_path)
            info = reader.metadata
            
            # Safely extract metadata with validation
            title = info.get("/Title")
            if title and isinstance(title, str):
                title = title.strip() or None
            else:
                title = None
                
            author = info.get("/Author")
            if author and isinstance(author, str):
                author = author.strip() or None
            else:
                author = None
                
            subject = info.get("/Subject")
            if subject and isinstance(subject, str):
                subject = subject.strip() or None
            else:
                subject = None
            
            return DocumentMetadata(
                title=title,
                author=author,
                subject=subject,
                pages=len(reader.pages)
            )
            
        except Exception as e:
            logger.error(f"pypdf metadata extraction failed for {file_path}: {e}")
            raise
    
    async def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """pypdf doesn't support table extraction"""
        return []
    
    async def extract_images(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract images using pypdf"""
        images = []
        try:
            reader = PdfReader(file_path)
            
            for page_num, page in enumerate(reader.pages):
                if "/XObject" in page["/Resources"]:
                    xobjects = page["/Resources"]["/XObject"].get_object()
                    
                    for obj_name in xobjects:
                        if xobjects[obj_name]["/Subtype"] == "/Image":
                            images.append({
                                "page": page_num + 1,
                                "name": obj_name
                            })
            
            return images
            
        except Exception as e:
            logger.error(f"pypdf image extraction failed: {e}")
            return []


class PDFPlumberProcessor(PDFProcessor):
    """PDF processor using pdfplumber (best for tables)"""
    
    async def extract_text(self, file_path: str) -> Tuple[str, List[ExtractedText]]:
        """Extract text using pdfplumber"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        try:
            full_text = ""
            pages = []
            
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        text = page.extract_text() or ""
                        if text:  # Only add non-empty text
                            full_text += text + "\n"
                        
                        page_data = ExtractedText(
                            page_number=page_num + 1,
                            text=text,
                            confidence=0.92
                        )
                        pages.append(page_data)
                    except Exception as page_error:
                        logger.warning(f"Failed to extract text from page {page_num + 1}: {page_error}")
                        # Add empty page data to maintain page numbering
                        pages.append(ExtractedText(
                            page_number=page_num + 1,
                            text="",
                            confidence=0.0
                        ))
            
            return full_text.strip(), pages
            
        except Exception as e:
            logger.error(f"pdfplumber text extraction failed for {file_path}: {e}")
            raise
    
    async def extract_metadata(self, file_path: str) -> DocumentMetadata:
        """Extract metadata using pdfplumber"""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
            
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata = pdf.metadata or {}
                
                # Safely extract metadata with validation
                title = metadata.get("Title")
                if title and isinstance(title, str):
                    title = title.strip() or None
                else:
                    title = None
                    
                author = metadata.get("Author")
                if author and isinstance(author, str):
                    author = author.strip() or None
                else:
                    author = None
                    
                subject = metadata.get("Subject")
                if subject and isinstance(subject, str):
                    subject = subject.strip() or None
                else:
                    subject = None
                
                return DocumentMetadata(
                    title=title,
                    author=author,
                    subject=subject,
                    pages=len(pdf.pages)
                )
                
        except Exception as e:
            logger.error(f"pdfplumber metadata extraction failed for {file_path}: {e}")
            raise
    
    async def extract_tables(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract tables using pdfplumber (best feature)"""
        tables = []
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(page_tables):
                        if table:
                            tables.append({
                                "page": page_num + 1,
                                "table_number": table_num + 1,
                                "data": table
                            })
            
            return tables
            
        except Exception as e:
            logger.error(f"pdfplumber table extraction failed: {e}")
            return []
    
    async def extract_images(self, file_path: str) -> List[Dict[str, Any]]:
        """pdfplumber doesn't support image extraction well"""
        return []


class ComprehensivePDFProcessor:
    """Main PDF processor that uses multiple backends with fallback"""
    
    def __init__(self, storage_service: StorageService):
        self.storage = storage_service
        self.processors = [
            PyMuPDFProcessor(),     # Fastest, most features
            PDFPlumberProcessor(),  # Best for tables
            PyPDFProcessor()        # Fallback, pure Python
        ]
    
    async def process_document(self, file_path: str, document_id: str) -> DocumentContent:
        """Process PDF document with fallback mechanism"""
        full_text = ""
        pages = []
        tables = []
        images = []
        metadata = None
        
        # Try text extraction with fallback
        for processor in self.processors:
            try:
                full_text, pages = await processor.extract_text(file_path)
                logger.info(f"Text extracted successfully using {processor.__class__.__name__}")
                break
            except Exception as e:
                logger.warning(f"{processor.__class__.__name__} failed for text: {e}")
                if processor == self.processors[-1]:
                    raise Exception("All processors failed for text extraction")
        
        # Try metadata extraction
        for processor in self.processors:
            try:
                metadata = await processor.extract_metadata(file_path)
                logger.info(f"Metadata extracted successfully using {processor.__class__.__name__}")
                break
            except Exception as e:
                logger.warning(f"{processor.__class__.__name__} failed for metadata: {e}")
        
        # Try table extraction (prefer pdfplumber)
        processors_for_tables = [self.processors[1]] + [p for p in self.processors if p != self.processors[1]]
        for processor in processors_for_tables:
            try:
                tables = await processor.extract_tables(file_path)
                if tables:
                    logger.info(f"Tables extracted successfully using {processor.__class__.__name__}")
                    break
            except Exception as e:
                logger.warning(f"{processor.__class__.__name__} failed for tables: {e}")
        
        # Try image extraction (prefer PyMuPDF)
        for processor in self.processors:
            try:
                images = await processor.extract_images(file_path)
                if images:
                    logger.info(f"Images extracted successfully using {processor.__class__.__name__}")
                    break
            except Exception as e:
                logger.warning(f"{processor.__class__.__name__} failed for images: {e}")
        
        return DocumentContent(
            document_id=document_id,
            full_text=full_text,
            pages=pages,
            tables=tables,
            images=images,
            metadata=metadata
        )