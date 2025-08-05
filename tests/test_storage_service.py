"""
Tests for storage service implementations (local and S3)
"""
import pytest
import pytest_asyncio
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock
import tempfile
import shutil

from src.services.storage_service import (
    LocalStorageService, S3StorageService, get_storage_service
)
from src.config import Settings


class TestLocalStorageService:
    """Test cases for LocalStorageService"""

    @pytest.mark.asyncio
    async def test_init_creates_directory(self, temp_dir: Path):
        """Test that LocalStorageService creates the base directory"""
        storage_path = temp_dir / "storage"
        assert not storage_path.exists()
        
        service = LocalStorageService(str(storage_path))
        assert storage_path.exists()
        assert service.base_path == storage_path

    @pytest.mark.asyncio
    async def test_save_file_success(self, local_storage_service: LocalStorageService):
        """Test successful file saving"""
        test_content = b"test file content"
        filename = "test.pdf"
        
        storage_path = await local_storage_service.save_file(test_content, filename)
        
        assert storage_path is not None
        assert Path(storage_path).exists()
        assert Path(storage_path).suffix == ".pdf"
        
        # Verify content
        with open(storage_path, 'rb') as f:
            saved_content = f.read()
        assert saved_content == test_content

    @pytest.mark.asyncio
    async def test_save_file_generates_unique_paths(self, local_storage_service: LocalStorageService):
        """Test that save_file generates unique paths for same filename"""
        test_content = b"test content"
        filename = "test.pdf"
        
        path1 = await local_storage_service.save_file(test_content, filename)
        path2 = await local_storage_service.save_file(test_content, filename)
        
        assert path1 != path2
        assert Path(path1).exists()
        assert Path(path2).exists()

    @pytest.mark.asyncio
    async def test_get_file_success(self, local_storage_service: LocalStorageService):
        """Test successful file retrieval"""
        test_content = b"test file content"
        filename = "test.pdf"
        
        storage_path = await local_storage_service.save_file(test_content, filename)
        retrieved_content = await local_storage_service.get_file(storage_path)
        
        assert retrieved_content == test_content

    @pytest.mark.asyncio
    async def test_get_file_not_found(self, local_storage_service: LocalStorageService):
        """Test get_file raises FileNotFoundError for non-existent file"""
        with pytest.raises(FileNotFoundError):
            await local_storage_service.get_file("/nonexistent/path.pdf")

    @pytest.mark.asyncio
    async def test_delete_file_success(self, local_storage_service: LocalStorageService):
        """Test successful file deletion"""
        test_content = b"test file content"
        filename = "test.pdf"
        
        storage_path = await local_storage_service.save_file(test_content, filename)
        assert Path(storage_path).exists()
        
        result = await local_storage_service.delete_file(storage_path)
        assert result is True
        assert not Path(storage_path).exists()

    @pytest.mark.asyncio
    async def test_delete_file_not_found(self, local_storage_service: LocalStorageService):
        """Test delete_file returns False for non-existent file"""
        result = await local_storage_service.delete_file("/nonexistent/path.pdf")
        assert result is False

    @pytest.mark.asyncio
    async def test_file_exists_true(self, local_storage_service: LocalStorageService):
        """Test file_exists returns True for existing file"""
        test_content = b"test file content"
        filename = "test.pdf"
        
        storage_path = await local_storage_service.save_file(test_content, filename)
        exists = await local_storage_service.file_exists(storage_path)
        assert exists is True

    @pytest.mark.asyncio
    async def test_file_exists_false(self, local_storage_service: LocalStorageService):
        """Test file_exists returns False for non-existent file"""
        exists = await local_storage_service.file_exists("/nonexistent/path.pdf")
        assert exists is False

    @pytest.mark.asyncio
    async def test_generate_path_preserves_extension(self, local_storage_service: LocalStorageService):
        """Test that _generate_path preserves file extension"""
        service = local_storage_service
        
        pdf_path = service._generate_path("document.pdf")
        docx_path = service._generate_path("document.docx")
        txt_path = service._generate_path("document.txt")
        
        assert pdf_path.suffix == ".pdf"
        assert docx_path.suffix == ".docx"
        assert txt_path.suffix == ".txt"

    @pytest.mark.asyncio
    async def test_generate_path_handles_no_extension(self, local_storage_service: LocalStorageService):
        """Test that _generate_path handles files without extension"""
        service = local_storage_service
        path = service._generate_path("document")
        
        assert path.suffix == ""
        assert path.name != "document"  # Should have UUID prefix

    @pytest.mark.asyncio
    async def test_save_file_permission_error(self, temp_dir: Path):
        """Test save_file handles permission errors gracefully"""
        # Create a read-only directory
        readonly_dir = temp_dir / "readonly"
        readonly_dir.mkdir()
        readonly_dir.chmod(0o444)
        
        service = LocalStorageService(str(readonly_dir))
        
        with pytest.raises(Exception):  # Should raise PermissionError or similar
            await service.save_file(b"test content", "test.pdf")


class TestS3StorageService:
    """Test cases for S3StorageService"""

    def test_init_with_all_params(self):
        """Test S3StorageService initialization with all parameters"""
        service = S3StorageService(
            bucket_name="test-bucket",
            endpoint_url="http://localhost:9000",
            access_key="test-access",
            secret_key="test-secret"
        )
        
        assert service.bucket_name == "test-bucket"
        assert service.endpoint_url == "http://localhost:9000"
        assert service.access_key == "test-access"
        assert service.secret_key == "test-secret"

    def test_init_missing_required_params(self):
        """Test S3StorageService raises ValueError for missing required params"""
        with pytest.raises(ValueError, match="S3 storage requires"):
            S3StorageService(bucket_name="test-bucket")  # Missing access_key and secret_key

    def test_generate_key_format(self):
        """Test that _generate_key creates proper S3 key format"""
        service = S3StorageService(
            bucket_name="test-bucket",
            access_key="test-access",
            secret_key="test-secret"
        )
        
        key = service._generate_key("document.pdf")
        
        assert key.startswith("documents/")
        assert key.endswith(".pdf")
        assert len(key.split("/")[1]) > 10  # UUID should be longer than 10 chars

    @pytest.mark.asyncio
    async def test_save_file_success(self, s3_storage_service: S3StorageService, mock_s3_client):
        """Test successful S3 file saving"""
        test_content = b"test file content"
        filename = "test.pdf"
        
        storage_path = await s3_storage_service.save_file(test_content, filename)
        
        assert storage_path.startswith("s3://test-bucket/documents/")
        assert storage_path.endswith(".pdf")
        
        # Verify S3 client was called correctly
        mock_s3_client.put_object.assert_called_once()
        call_args = mock_s3_client.put_object.call_args
        assert call_args[1]["Bucket"] == "test-bucket"
        assert call_args[1]["Body"] == test_content

    @pytest.mark.asyncio
    async def test_get_file_success(self, s3_storage_service: S3StorageService, mock_s3_client):
        """Test successful S3 file retrieval"""
        test_content = b"test file content"
        storage_path = "s3://test-bucket/documents/test-file.pdf"
        
        # Mock the response body
        mock_s3_client.get_object.return_value = {
            "Body": AsyncMock(read=AsyncMock(return_value=test_content))
        }
        
        retrieved_content = await s3_storage_service.get_file(storage_path)
        
        assert retrieved_content == test_content
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="documents/test-file.pdf"
        )

    @pytest.mark.asyncio
    async def test_get_file_with_key_only(self, s3_storage_service: S3StorageService, mock_s3_client):
        """Test get_file with key only (no s3:// prefix)"""
        test_content = b"test file content"
        storage_path = "documents/test-file.pdf"
        
        mock_s3_client.get_object.return_value = {
            "Body": AsyncMock(read=AsyncMock(return_value=test_content))
        }
        
        retrieved_content = await s3_storage_service.get_file(storage_path)
        
        assert retrieved_content == test_content
        mock_s3_client.get_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="documents/test-file.pdf"
        )

    @pytest.mark.asyncio
    async def test_delete_file_success(self, s3_storage_service: S3StorageService, mock_s3_client):
        """Test successful S3 file deletion"""
        storage_path = "s3://test-bucket/documents/test-file.pdf"
        
        result = await s3_storage_service.delete_file(storage_path)
        
        assert result is True
        mock_s3_client.delete_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="documents/test-file.pdf"
        )

    @pytest.mark.asyncio
    async def test_file_exists_true(self, s3_storage_service: S3StorageService, mock_s3_client):
        """Test file_exists returns True for existing S3 file"""
        storage_path = "s3://test-bucket/documents/test-file.pdf"
        
        # head_object succeeds for existing files
        mock_s3_client.head_object.return_value = {"ContentLength": 100}
        
        exists = await s3_storage_service.file_exists(storage_path)
        
        assert exists is True
        mock_s3_client.head_object.assert_called_once_with(
            Bucket="test-bucket",
            Key="documents/test-file.pdf"
        )

    @pytest.mark.asyncio
    async def test_file_exists_false(self, s3_storage_service: S3StorageService, mock_s3_client):
        """Test file_exists returns False for non-existent S3 file"""
        storage_path = "s3://test-bucket/documents/nonexistent.pdf"
        
        # head_object raises exception for non-existent files
        mock_s3_client.head_object.side_effect = Exception("Not found")
        
        exists = await s3_storage_service.file_exists(storage_path)
        
        assert exists is False

    @pytest.mark.asyncio
    async def test_save_file_error_handling(self, s3_storage_service: S3StorageService, mock_s3_client):
        """Test save_file handles S3 errors gracefully"""
        mock_s3_client.put_object.side_effect = Exception("S3 Error")
        
        with pytest.raises(Exception, match="S3 Error"):
            await s3_storage_service.save_file(b"test content", "test.pdf")

    @pytest.mark.asyncio
    async def test_get_file_error_handling(self, s3_storage_service: S3StorageService, mock_s3_client):
        """Test get_file handles S3 errors gracefully"""
        mock_s3_client.get_object.side_effect = Exception("S3 Error")
        
        with pytest.raises(Exception, match="S3 Error"):
            await s3_storage_service.get_file("s3://test-bucket/test.pdf")

    @pytest.mark.asyncio
    async def test_delete_file_error_handling(self, s3_storage_service: S3StorageService, mock_s3_client):
        """Test delete_file handles S3 errors gracefully"""
        mock_s3_client.delete_object.side_effect = Exception("S3 Error")
        
        with pytest.raises(Exception, match="S3 Error"):
            await s3_storage_service.delete_file("s3://test-bucket/test.pdf")


class TestStorageServiceFactory:
    """Test cases for storage service factory function"""

    def test_get_storage_service_local(self, test_settings):
        """Test factory returns LocalStorageService for local storage type"""
        test_settings.STORAGE_TYPE = "local"
        
        with patch('src.services.storage_service.settings', test_settings):
            service = get_storage_service()
            assert isinstance(service, LocalStorageService)

    def test_get_storage_service_s3(self, test_settings):
        """Test factory returns S3StorageService for s3 storage type"""
        test_settings.STORAGE_TYPE = "s3"
        test_settings.S3_BUCKET_NAME = "test-bucket"
        test_settings.S3_ACCESS_KEY = "test-access"
        test_settings.S3_SECRET_KEY = "test-secret"
        
        with patch('src.services.storage_service.settings', test_settings):
            service = get_storage_service()
            assert isinstance(service, S3StorageService)

    def test_get_storage_service_default_local(self, test_settings):
        """Test factory defaults to LocalStorageService for unknown storage type"""
        test_settings.STORAGE_TYPE = "unknown"
        
        with patch('src.services.storage_service.settings', test_settings):
            service = get_storage_service()
            assert isinstance(service, LocalStorageService)


class TestStorageServiceIntegration:
    """Integration tests for storage services"""

    @pytest.mark.asyncio
    async def test_large_file_handling(self, local_storage_service: LocalStorageService):
        """Test handling of large files"""
        # Create 1MB test content
        large_content = b"x" * (1024 * 1024)
        filename = "large_file.pdf"
        
        storage_path = await local_storage_service.save_file(large_content, filename)
        retrieved_content = await local_storage_service.get_file(storage_path)
        
        assert len(retrieved_content) == len(large_content)
        assert retrieved_content == large_content

    @pytest.mark.asyncio
    async def test_binary_content_integrity(self, local_storage_service: LocalStorageService):
        """Test that binary content is preserved exactly"""
        # Create binary content with various byte values
        binary_content = bytes(range(256))
        filename = "binary_file.bin"
        
        storage_path = await local_storage_service.save_file(binary_content, filename)
        retrieved_content = await local_storage_service.get_file(storage_path)
        
        assert retrieved_content == binary_content

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, local_storage_service: LocalStorageService):
        """Test concurrent file operations"""
        import asyncio
        
        async def save_and_retrieve(content: bytes, filename: str):
            storage_path = await local_storage_service.save_file(content, filename)
            retrieved = await local_storage_service.get_file(storage_path)
            return retrieved == content
        
        # Run multiple operations concurrently
        tasks = [
            save_and_retrieve(f"content_{i}".encode(), f"file_{i}.txt")
            for i in range(10)
        ]
        
        results = await asyncio.gather(*tasks)
        assert all(results)

    @pytest.mark.asyncio
    async def test_unicode_filename_handling(self, local_storage_service: LocalStorageService):
        """Test handling of Unicode characters in filenames"""
        test_content = b"test content"
        unicode_filename = "тест_файл_🐍.pdf"
        
        storage_path = await local_storage_service.save_file(test_content, unicode_filename)
        retrieved_content = await local_storage_service.get_file(storage_path)
        
        assert retrieved_content == test_content
        assert Path(storage_path).suffix == ".pdf"