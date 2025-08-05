"""
Tests for PDF processor implementations
"""
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from datetime import datetime
import tempfile

from src.services.pdf_processor import (
    PyMuPDFProcessor, PyPDFProcessor, PDFPlumberProcessor,
    ComprehensivePDFProcessor
)
from src.services.storage_service import LocalStorageService
from src.models.document import DocumentContent, ExtractedText, DocumentMetadata


class TestPyMuPDFProcessor:
    """Test cases for PyMuPDFProcessor"""

    @pytest.fixture
    def processor(self):
        """Create PyMuPDFProcessor instance"""
        return PyMuPDFProcessor()

    @pytest.mark.asyncio
    async def test_extract_text_success(self, processor, test_pdf_file):
        """Test successful text extraction using PyMuPDF"""
        with patch('fitz.open') as mock_fitz:
            # Mock PyMuPDF document
            mock_doc = Mock()
            mock_page = Mock()
            mock_page.get_text.return_value = "Sample text from page 1"
            mock_page.get_text.side_effect = lambda arg=None: {
                None: "Sample text from page 1",
                "dict": {"blocks": []}
            }.get(arg, "Sample text from page 1")
            
            mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            full_text, pages = await processor.extract_text(str(test_pdf_file))
            
            assert full_text == "Sample text from page 1"
            assert len(pages) == 1
            assert pages[0].page_number == 1
            assert pages[0].text == "Sample text from page 1"
            assert pages[0].confidence == 0.95
            mock_doc.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_text_multiple_pages(self, processor, test_pdf_file):
        """Test text extraction from multiple pages"""
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            
            # Create multiple mock pages
            pages_data = [
                "Text from page 1",
                "Text from page 2",
                "Text from page 3"
            ]
            mock_pages = []
            for text in pages_data:
                mock_page = Mock()
                mock_page.get_text.return_value = text
                mock_page.get_text.side_effect = lambda arg=None, txt=text: {
                    None: txt,
                    "dict": {"blocks": []}
                }.get(arg, txt)
                mock_pages.append(mock_page)
            
            mock_doc.__iter__ = Mock(return_value=iter(mock_pages))
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            full_text, pages = await processor.extract_text(str(test_pdf_file))
            
            expected_full_text = "Text from page 1\nText from page 2\nText from page 3"
            assert full_text == expected_full_text
            assert len(pages) == 3
            
            for i, page in enumerate(pages):
                assert page.page_number == i + 1
                assert page.text == pages_data[i]

    @pytest.mark.asyncio
    async def test_extract_text_error_handling(self, processor, test_pdf_file):
        """Test error handling in text extraction"""
        with patch('fitz.open') as mock_fitz:
            mock_fitz.side_effect = Exception("PDF parsing error")
            
            with pytest.raises(Exception, match="PDF parsing error"):
                await processor.extract_text(str(test_pdf_file))

    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, processor, test_pdf_file):
        """Test successful metadata extraction"""
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_doc.metadata = {
                "title": "Test Document",
                "author": "Test Author",
                "subject": "Test Subject",
                "keywords": "test,document,pdf",
                "creationDate": "D:20240101120000",
                "modDate": "D:20240101120000"
            }
            mock_doc.page_count = 5
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            metadata = await processor.extract_metadata(str(test_pdf_file))
            
            assert metadata.title == "Test Document"
            assert metadata.author == "Test Author"
            assert metadata.subject == "Test Subject"
            assert metadata.keywords == ["test", "document", "pdf"]
            assert metadata.pages == 5
            assert isinstance(metadata.creation_date, datetime)
            mock_doc.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_metadata_minimal(self, processor, test_pdf_file):
        """Test metadata extraction with minimal data"""
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_doc.metadata = {}
            mock_doc.page_count = 1
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            metadata = await processor.extract_metadata(str(test_pdf_file))
            
            assert metadata.title is None
            assert metadata.author is None
            assert metadata.subject is None
            assert metadata.keywords is None
            assert metadata.pages == 1

    @pytest.mark.asyncio
    async def test_extract_images_success(self, processor, test_pdf_file):
        """Test successful image extraction"""
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_page = Mock()
            
            # Mock image list
            mock_page.get_images.return_value = [(123, 0, 100, 100, 8, "DeviceRGB", "", "", "")]
            
            # Mock Pixmap
            mock_pixmap = Mock()
            mock_pixmap.width = 100
            mock_pixmap.height = 100
            mock_pixmap.colorspace.name = "DeviceRGB"
            
            with patch('fitz.Pixmap', return_value=mock_pixmap):
                mock_doc.__iter__ = Mock(return_value=iter([mock_page]))
                mock_doc.close = Mock()
                mock_fitz.return_value = mock_doc
                
                images = await processor.extract_images(str(test_pdf_file))
                
                assert len(images) == 1
                assert images[0]["page"] == 1
                assert images[0]["width"] == 100
                assert images[0]["height"] == 100
                assert images[0]["xref"] == 123

    @pytest.mark.asyncio
    async def test_extract_tables_placeholder(self, processor, test_pdf_file):
        """Test that extract_tables returns empty list (placeholder implementation)"""
        with patch('fitz.open') as mock_fitz:
            mock_doc = Mock()
            mock_doc.__iter__ = Mock(return_value=iter([Mock()]))
            mock_doc.close = Mock()
            mock_fitz.return_value = mock_doc
            
            tables = await processor.extract_tables(str(test_pdf_file))
            
            assert tables == []

    def test_parse_date_valid(self, processor):
        """Test _parse_date with valid PDF date string"""
        date_str = "D:20240101120000"
        parsed_date = processor._parse_date(date_str)
        
        assert parsed_date is not None
        assert parsed_date.year == 2024
        assert parsed_date.month == 1
        assert parsed_date.day == 1

    def test_parse_date_invalid(self, processor):
        """Test _parse_date with invalid date string"""
        assert processor._parse_date("") is None
        assert processor._parse_date("invalid") is None
        assert processor._parse_date(None) is None


class TestPyPDFProcessor:
    """Test cases for PyPDFProcessor"""

    @pytest.fixture
    def processor(self):
        """Create PyPDFProcessor instance"""
        return PyPDFProcessor()

    @pytest.mark.asyncio
    async def test_extract_text_success(self, processor, test_pdf_file):
        """Test successful text extraction using pypdf"""
        with patch('pypdf.PdfReader') as mock_reader_class:
            mock_reader = Mock()
            mock_page = Mock()
            mock_page.extract_text.return_value = "Sample text from pypdf"
            mock_reader.pages = [mock_page]
            mock_reader_class.return_value = mock_reader
            
            full_text, pages = await processor.extract_text(str(test_pdf_file))
            
            assert full_text == "Sample text from pypdf"
            assert len(pages) == 1
            assert pages[0].page_number == 1
            assert pages[0].text == "Sample text from pypdf"
            assert pages[0].confidence == 0.90

    @pytest.mark.asyncio
    async def test_extract_metadata_success(self, processor, test_pdf_file):
        """Test successful metadata extraction using pypdf"""
        with patch('pypdf.PdfReader') as mock_reader_class:
            mock_reader = Mock()
            mock_reader.metadata = {
                "/Title": "Test Document",
                "/Author": "Test Author",
                "/Subject": "Test Subject"
            }
            mock_reader.pages = [Mock(), Mock()]  # 2 pages
            mock_reader_class.return_value = mock_reader
            
            metadata = await processor.extract_metadata(str(test_pdf_file))
            
            assert metadata.title == "Test Document"
            assert metadata.author == "Test Author"
            assert metadata.subject == "Test Subject"
            assert metadata.pages == 2

    @pytest.mark.asyncio
    async def test_extract_tables_not_supported(self, processor, test_pdf_file):
        """Test that pypdf processor doesn't support table extraction"""
        tables = await processor.extract_tables(str(test_pdf_file))
        assert tables == []

    @pytest.mark.asyncio
    async def test_extract_images_success(self, processor, test_pdf_file):
        """Test image extraction using pypdf"""
        with patch('pypdf.PdfReader') as mock_reader_class:
            mock_reader = Mock()
            mock_page = Mock()
            
            # Mock page resources structure
            mock_xobjects = {
                "/Image1": {"/Subtype": "/Image"},
                "/NonImage": {"/Subtype": "/Form"}
            }
            mock_page.__getitem__.return_value = {
                "/Resources": {
                    "/XObject": Mock(get_object=Mock(return_value=mock_xobjects))
                }
            }
            mock_reader.pages = [mock_page]
            mock_reader_class.return_value = mock_reader
            
            images = await processor.extract_images(str(test_pdf_file))
            
            assert len(images) == 1
            assert images[0]["page"] == 1
            assert images[0]["name"] == "/Image1"


class TestPDFPlumberProcessor:
    """Test cases for PDFPlumberProcessor"""

    @pytest.fixture
    def processor(self):
        """Create PDFPlumberProcessor instance"""
        return PDFPlumberProcessor()

    @pytest.mark.asyncio
    async def test_extract_text_success(self, processor, test_pdf_file):
        """Test successful text extraction using pdfplumber"""
        with patch('pdfplumber.open') as mock_open:
            mock_pdf = Mock()
            mock_page = Mock()
            mock_page.extract_text.return_value = "Sample text from pdfplumber"
            mock_pdf.pages = [mock_page]
            mock_pdf.__enter__ = Mock(return_value=mock_pdf)
            mock_pdf.__exit__ = Mock(return_value=None)
            mock_open.return_value = mock_pdf
            
            full_text, pages = await processor.extract_text(str(test_pdf_file))
            
            assert full_text == "Sample text from pdfplumber"
            assert len(pages) == 1
            assert pages[0].confidence == 0.92

    @pytest.mark.asyncio
    async def test_extract_text_none_handling(self, processor, test_pdf_file):
        """Test handling of None text from pdfplumber"""
        with patch('pdfplumber.open') as mock_open:
            mock_pdf = Mock()
            mock_page = Mock()
            mock_page.extract_text.return_value = None  # pdfplumber can return None
            mock_pdf.pages = [mock_page]
            mock_pdf.__enter__ = Mock(return_value=mock_pdf)
            mock_pdf.__exit__ = Mock(return_value=None)
            mock_open.return_value = mock_pdf
            
            full_text, pages = await processor.extract_text(str(test_pdf_file))
            
            assert full_text == ""
            assert pages[0].text == ""

    @pytest.mark.asyncio
    async def test_extract_tables_success(self, processor, test_pdf_file):
        """Test successful table extraction using pdfplumber"""
        with patch('pdfplumber.open') as mock_open:
            mock_pdf = Mock()
            mock_page = Mock()
            
            # Mock table data
            table_data = [
                ["Header1", "Header2"],
                ["Row1Col1", "Row1Col2"],
                ["Row2Col1", "Row2Col2"]
            ]
            mock_page.extract_tables.return_value = [table_data]
            mock_pdf.pages = [mock_page]
            mock_pdf.__enter__ = Mock(return_value=mock_pdf)
            mock_pdf.__exit__ = Mock(return_value=None)
            mock_open.return_value = mock_pdf
            
            tables = await processor.extract_tables(str(test_pdf_file))
            
            assert len(tables) == 1
            assert tables[0]["page"] == 1
            assert tables[0]["table_number"] == 1
            assert tables[0]["data"] == table_data

    @pytest.mark.asyncio
    async def test_extract_tables_multiple_pages(self, processor, test_pdf_file):
        """Test table extraction from multiple pages"""
        with patch('pdfplumber.open') as mock_open:
            mock_pdf = Mock()
            
            # Page 1 with one table
            mock_page1 = Mock()
            mock_page1.extract_tables.return_value = [
                [["P1T1H1", "P1T1H2"], ["P1T1R1C1", "P1T1R1C2"]]
            ]
            
            # Page 2 with two tables
            mock_page2 = Mock()
            mock_page2.extract_tables.return_value = [
                [["P2T1H1", "P2T1H2"], ["P2T1R1C1", "P2T1R1C2"]],
                [["P2T2H1", "P2T2H2"], ["P2T2R1C1", "P2T2R1C2"]]
            ]
            
            mock_pdf.pages = [mock_page1, mock_page2]
            mock_pdf.__enter__ = Mock(return_value=mock_pdf)
            mock_pdf.__exit__ = Mock(return_value=None)
            mock_open.return_value = mock_pdf
            
            tables = await processor.extract_tables(str(test_pdf_file))
            
            assert len(tables) == 3
            assert tables[0]["page"] == 1
            assert tables[1]["page"] == 2
            assert tables[2]["page"] == 2
            assert tables[1]["table_number"] == 1
            assert tables[2]["table_number"] == 2

    @pytest.mark.asyncio
    async def test_extract_images_not_supported(self, processor, test_pdf_file):
        """Test that pdfplumber processor doesn't support image extraction well"""
        images = await processor.extract_images(str(test_pdf_file))
        assert images == []


class TestComprehensivePDFProcessor:
    """Test cases for ComprehensivePDFProcessor"""

    @pytest.fixture
    def storage_service(self, temp_dir):
        """Create storage service for processor"""
        return LocalStorageService(str(temp_dir))

    @pytest.fixture
    def processor(self, storage_service):
        """Create ComprehensivePDFProcessor instance"""
        return ComprehensivePDFProcessor(storage_service)

    @pytest.mark.asyncio
    async def test_process_document_success(self, processor, test_pdf_file):
        """Test successful document processing with fallback mechanism"""
        document_id = "test-doc-123"
        
        # Mock all processors to succeed
        with patch.object(processor.processors[0], 'extract_text') as mock_text, \
             patch.object(processor.processors[0], 'extract_metadata') as mock_metadata, \
             patch.object(processor.processors[1], 'extract_tables') as mock_tables, \
             patch.object(processor.processors[0], 'extract_images') as mock_images:
            
            mock_text.return_value = ("Sample text", [
                ExtractedText(page_number=1, text="Sample text", confidence=0.95)
            ])
            mock_metadata.return_value = DocumentMetadata(title="Test", pages=1)
            mock_tables.return_value = [{"page": 1, "data": [["A", "B"]]}]
            mock_images.return_value = [{"page": 1, "width": 100}]
            
            content = await processor.process_document(str(test_pdf_file), document_id)
            
            assert content.document_id == document_id
            assert content.full_text == "Sample text"
            assert len(content.pages) == 1
            assert len(content.tables) == 1
            assert len(content.images) == 1
            assert content.metadata.title == "Test"

    @pytest.mark.asyncio
    async def test_process_document_fallback_text(self, processor, test_pdf_file):
        """Test fallback mechanism for text extraction"""
        document_id = "test-doc-123"
        
        with patch.object(processor.processors[0], 'extract_text') as mock_text1, \
             patch.object(processor.processors[1], 'extract_text') as mock_text2, \
             patch.object(processor.processors[2], 'extract_text') as mock_text3:
            
            # First processor fails
            mock_text1.side_effect = Exception("PyMuPDF failed")
            
            # Second processor succeeds
            mock_text2.return_value = ("Fallback text", [
                ExtractedText(page_number=1, text="Fallback text", confidence=0.92)
            ])
            
            # Mock other methods to avoid errors
            with patch.object(processor.processors[0], 'extract_metadata', return_value=DocumentMetadata()), \
                 patch.object(processor.processors[1], 'extract_tables', return_value=[]), \
                 patch.object(processor.processors[0], 'extract_images', return_value=[]):
                
                content = await processor.process_document(str(test_pdf_file), document_id)
                
                assert content.full_text == "Fallback text"
                mock_text1.assert_called_once()
                mock_text2.assert_called_once()
                mock_text3.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_document_all_processors_fail(self, processor, test_pdf_file):
        """Test when all processors fail for text extraction"""
        document_id = "test-doc-123"
        
        # All processors fail for text extraction
        for proc in processor.processors:
            with patch.object(proc, 'extract_text', side_effect=Exception("Failed")):
                pass
        
        with patch.object(processor.processors[0], 'extract_text', side_effect=Exception("Failed")), \
             patch.object(processor.processors[1], 'extract_text', side_effect=Exception("Failed")), \
             patch.object(processor.processors[2], 'extract_text', side_effect=Exception("Failed")):
            
            with pytest.raises(Exception, match="All processors failed for text extraction"):
                await processor.process_document(str(test_pdf_file), document_id)

    @pytest.mark.asyncio
    async def test_process_document_table_fallback(self, processor, test_pdf_file):
        """Test that pdfplumber is preferred for table extraction"""
        document_id = "test-doc-123"
        
        with patch.object(processor.processors[0], 'extract_text', return_value=("", [])), \
             patch.object(processor.processors[0], 'extract_metadata', return_value=DocumentMetadata()), \
             patch.object(processor.processors[1], 'extract_tables') as mock_pdfplumber_tables, \
             patch.object(processor.processors[0], 'extract_tables') as mock_pymupdf_tables, \
             patch.object(processor.processors[0], 'extract_images', return_value=[]):
            
            # pdfplumber succeeds with tables
            mock_pdfplumber_tables.return_value = [{"page": 1, "data": [["A", "B"]]}]
            
            content = await processor.process_document(str(test_pdf_file), document_id)
            
            # Should call pdfplumber first for tables
            mock_pdfplumber_tables.assert_called_once()
            mock_pymupdf_tables.assert_not_called()
            assert len(content.tables) == 1

    @pytest.mark.asyncio
    async def test_process_document_metadata_fallback(self, processor, test_pdf_file):
        """Test metadata extraction with fallback"""
        document_id = "test-doc-123"
        
        with patch.object(processor.processors[0], 'extract_text', return_value=("", [])), \
             patch.object(processor.processors[0], 'extract_metadata') as mock_meta1, \
             patch.object(processor.processors[1], 'extract_metadata') as mock_meta2, \
             patch.object(processor.processors[1], 'extract_tables', return_value=[]), \
             patch.object(processor.processors[0], 'extract_images', return_value=[]):
            
            # First processor fails, second succeeds
            mock_meta1.side_effect = Exception("Metadata failed")
            mock_meta2.return_value = DocumentMetadata(title="Fallback metadata")
            
            content = await processor.process_document(str(test_pdf_file), document_id)
            
            assert content.metadata.title == "Fallback metadata"
            mock_meta1.assert_called_once()
            mock_meta2.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_document_graceful_failure(self, processor, test_pdf_file):
        """Test that processor continues even if some extractions fail"""
        document_id = "test-doc-123"
        
        with patch.object(processor.processors[0], 'extract_text', return_value=("Text works", [])), \
             patch.object(processor.processors[0], 'extract_metadata', side_effect=Exception("Meta fails")), \
             patch.object(processor.processors[1], 'extract_metadata', side_effect=Exception("Meta fails")), \
             patch.object(processor.processors[2], 'extract_metadata', side_effect=Exception("Meta fails")), \
             patch.object(processor.processors[1], 'extract_tables', side_effect=Exception("Tables fail")), \
             patch.object(processor.processors[0], 'extract_tables', side_effect=Exception("Tables fail")), \
             patch.object(processor.processors[2], 'extract_tables', side_effect=Exception("Tables fail")), \
             patch.object(processor.processors[0], 'extract_images', side_effect=Exception("Images fail")), \
             patch.object(processor.processors[1], 'extract_images', side_effect=Exception("Images fail")), \
             patch.object(processor.processors[2], 'extract_images', side_effect=Exception("Images fail")):
            
            # Should complete with text extraction only
            content = await processor.process_document(str(test_pdf_file), document_id)
            
            assert content.full_text == "Text works"
            assert content.metadata is None
            assert content.tables == []
            assert content.images == []

    def test_processor_initialization(self, processor, storage_service):
        """Test that processor initializes with correct storage service"""
        assert processor.storage == storage_service
        assert len(processor.processors) == 3
        assert isinstance(processor.processors[0], PyMuPDFProcessor)
        assert isinstance(processor.processors[1], PDFPlumberProcessor)
        assert isinstance(processor.processors[2], PyPDFProcessor)