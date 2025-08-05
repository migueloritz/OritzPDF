import os
import uuid
import aiofiles
from typing import Optional, BinaryIO
from abc import ABC, abstractmethod
from pathlib import Path
import aioboto3
from src.config import settings
import logging

logger = logging.getLogger(__name__)


class StorageService(ABC):
    """Abstract base class for storage services"""
    
    @abstractmethod
    async def save_file(self, file_content: bytes, filename: str) -> str:
        """Save file and return storage path"""
        pass
    
    @abstractmethod
    async def get_file(self, storage_path: str) -> bytes:
        """Retrieve file content"""
        pass
    
    @abstractmethod
    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from storage"""
        pass
    
    @abstractmethod
    async def file_exists(self, storage_path: str) -> bool:
        """Check if file exists"""
        pass


class LocalStorageService(StorageService):
    """Local filesystem storage implementation"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path or settings.LOCAL_STORAGE_PATH)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    def _generate_path(self, filename: str) -> Path:
        """Generate unique storage path"""
        file_id = str(uuid.uuid4())
        extension = Path(filename).suffix
        return self.base_path / f"{file_id}{extension}"
    
    async def save_file(self, file_content: bytes, filename: str) -> str:
        """Save file to local filesystem"""
        file_path = self._generate_path(filename)
        
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(file_content)
            
            logger.info(f"File saved: {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            raise
    
    async def get_file(self, storage_path: str) -> bytes:
        """Read file from local filesystem"""
        try:
            async with aiofiles.open(storage_path, 'rb') as f:
                content = await f.read()
            return content
        except FileNotFoundError:
            logger.error(f"File not found: {storage_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise
    
    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from local filesystem"""
        try:
            Path(storage_path).unlink()
            logger.info(f"File deleted: {storage_path}")
            return True
        except FileNotFoundError:
            logger.warning(f"File not found for deletion: {storage_path}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            raise
    
    async def file_exists(self, storage_path: str) -> bool:
        """Check if file exists in local filesystem"""
        return Path(storage_path).exists()


class S3StorageService(StorageService):
    """AWS S3 or S3-compatible storage implementation"""
    
    def __init__(self, bucket_name: str = None,
                 endpoint_url: str = None,
                 access_key: str = None,
                 secret_key: str = None):
        self.bucket_name = bucket_name or settings.S3_BUCKET_NAME
        self.endpoint_url = endpoint_url or settings.S3_ENDPOINT_URL
        self.access_key = access_key or settings.S3_ACCESS_KEY
        self.secret_key = secret_key or settings.S3_SECRET_KEY
        
        if not all([self.bucket_name, self.access_key, self.secret_key]):
            raise ValueError("S3 storage requires bucket_name, access_key, and secret_key")
    
    def _generate_key(self, filename: str) -> str:
        """Generate unique S3 key"""
        file_id = str(uuid.uuid4())
        extension = Path(filename).suffix
        return f"documents/{file_id}{extension}"
    
    async def _get_client(self):
        """Get S3 client"""
        session = aioboto3.Session()
        return session.client(
            's3',
            endpoint_url=self.endpoint_url,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key
        )
    
    async def save_file(self, file_content: bytes, filename: str) -> str:
        """Save file to S3"""
        key = self._generate_key(filename)
        
        try:
            async with await self._get_client() as s3:
                await s3.put_object(
                    Bucket=self.bucket_name,
                    Key=key,
                    Body=file_content
                )
            
            logger.info(f"File saved to S3: {key}")
            return f"s3://{self.bucket_name}/{key}"
        except Exception as e:
            logger.error(f"Failed to save file to S3: {e}")
            raise
    
    async def get_file(self, storage_path: str) -> bytes:
        """Read file from S3"""
        # Extract bucket and key from s3:// URL
        if storage_path.startswith("s3://"):
            parts = storage_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1]
        else:
            bucket = self.bucket_name
            key = storage_path
        
        try:
            async with await self._get_client() as s3:
                response = await s3.get_object(Bucket=bucket, Key=key)
                content = await response['Body'].read()
            return content
        except Exception as e:
            logger.error(f"Failed to read file from S3: {e}")
            raise
    
    async def delete_file(self, storage_path: str) -> bool:
        """Delete file from S3"""
        # Extract bucket and key from s3:// URL
        if storage_path.startswith("s3://"):
            parts = storage_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1]
        else:
            bucket = self.bucket_name
            key = storage_path
        
        try:
            async with await self._get_client() as s3:
                await s3.delete_object(Bucket=bucket, Key=key)
            logger.info(f"File deleted from S3: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete file from S3: {e}")
            raise
    
    async def file_exists(self, storage_path: str) -> bool:
        """Check if file exists in S3"""
        # Extract bucket and key from s3:// URL
        if storage_path.startswith("s3://"):
            parts = storage_path[5:].split("/", 1)
            bucket = parts[0]
            key = parts[1]
        else:
            bucket = self.bucket_name
            key = storage_path
        
        try:
            async with await self._get_client() as s3:
                await s3.head_object(Bucket=bucket, Key=key)
            return True
        except:
            return False


def get_storage_service() -> StorageService:
    """Factory function to get appropriate storage service"""
    if settings.STORAGE_TYPE == "s3":
        return S3StorageService()
    else:
        return LocalStorageService()