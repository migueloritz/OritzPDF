import json
import redis.asyncio as redis
from typing import Optional, Any, Callable
import logging
from src.config import settings
from src.models.document import DocumentContent
from src.models.analysis import (
    SummarizationResponse, QuestionAnswerResponse,
    DataExtractionResponse, TranslationResponse
)

logger = logging.getLogger(__name__)


class CacheService:
    """Redis-based caching service"""
    
    # Cache TTL configurations (in seconds)
    CACHE_TTL = {
        "document_content": 3600,      # 1 hour
        "summaries": 86400,           # 24 hours
        "qa_results": 1800,           # 30 minutes
        "extracted_entities": 7200,   # 2 hours
        "translations": 604800        # 7 days
    }
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis = None
    
    async def connect(self):
        """Connect to Redis"""
        if not self._redis:
            try:
                self._redis = await redis.from_url(self.redis_url, decode_responses=True)
                # Test the connection
                await self._redis.ping()
                logger.info("Connected to Redis cache")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                self._redis = None
                raise ConnectionError(f"Unable to connect to Redis cache: {e}")
    
    async def disconnect(self):
        """Disconnect from Redis"""
        if self._redis:
            try:
                await self._redis.close()
                logger.info("Disconnected from Redis cache")
            except Exception as e:
                logger.warning(f"Error during Redis disconnect: {e}")
            finally:
                self._redis = None
    
    async def _ensure_connected(self):
        """Ensure Redis connection is established"""
        if not self._redis:
            await self.connect()
        
        # Test connection health
        try:
            await self._redis.ping()
        except Exception as e:
            logger.warning(f"Redis connection lost, attempting reconnect: {e}")
            self._redis = None
            await self.connect()
    
    def _get_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key"""
        return f"oritzpdf:{prefix}:{identifier}"
    
    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not key or not isinstance(key, str):
            logger.warning("Invalid cache key provided")
            return None
        
        try:
            await self._ensure_connected()
            value = await self._redis.get(key)
            if value:
                logger.debug(f"Cache hit: {key}")
            else:
                logger.debug(f"Cache miss: {key}")
            return value
        except Exception as e:
            logger.error(f"Cache get error for key '{key}': {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache with TTL"""
        if not key or not isinstance(key, str):
            logger.warning("Invalid cache key provided")
            return False
        
        if ttl is not None and (not isinstance(ttl, int) or ttl < 0):
            logger.warning(f"Invalid TTL value: {ttl}")
            ttl = None
        
        try:
            await self._ensure_connected()
            # Convert to JSON if not string
            if not isinstance(value, str):
                try:
                    value = json.dumps(value, default=str)  # Handle datetime and other types
                except (TypeError, ValueError) as json_error:
                    logger.error(f"Failed to serialize value for cache key '{key}': {json_error}")
                    return False
            
            if ttl:
                await self._redis.setex(key, ttl, value)
            else:
                await self._redis.set(key, value)
            
            logger.debug(f"Cache set: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache set error for key '{key}': {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache"""
        await self._ensure_connected()
        try:
            result = await self._redis.delete(key)
            logger.debug(f"Cache delete: {key}")
            return bool(result)
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        await self._ensure_connected()
        try:
            return bool(await self._redis.exists(key))
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    # Document-specific cache methods
    
    async def get_document_content(self, document_id: str) -> Optional[DocumentContent]:
        """Get document content from cache"""
        key = self._get_key("document", document_id)
        data = await self.get(key)
        
        if data:
            try:
                return DocumentContent(**json.loads(data))
            except Exception as e:
                logger.error(f"Failed to deserialize document content: {e}")
                return None
        return None
    
    async def set_document_content(self, document_id: str, content: DocumentContent) -> bool:
        """Cache document content"""
        key = self._get_key("document", document_id)
        return await self.set(
            key,
            content.model_dump_json(),
            self.CACHE_TTL["document_content"]
        )
    
    async def get_summary(self, document_id: str, page_range: Optional[str] = None) -> Optional[SummarizationResponse]:
        """Get document summary from cache"""
        suffix = f"_{page_range}" if page_range else ""
        key = self._get_key("summary", f"{document_id}{suffix}")
        data = await self.get(key)
        
        if data:
            try:
                return SummarizationResponse(**json.loads(data))
            except Exception as e:
                logger.error(f"Failed to deserialize summary: {e}")
                return None
        return None
    
    async def set_summary(self, document_id: str, summary: SummarizationResponse, 
                         page_range: Optional[str] = None) -> bool:
        """Cache document summary"""
        suffix = f"_{page_range}" if page_range else ""
        key = self._get_key("summary", f"{document_id}{suffix}")
        return await self.set(
            key,
            summary.model_dump_json(),
            self.CACHE_TTL["summaries"]
        )
    
    async def get_qa_result(self, document_id: str, question_hash: str) -> Optional[QuestionAnswerResponse]:
        """Get Q&A result from cache"""
        key = self._get_key("qa", f"{document_id}:{question_hash}")
        data = await self.get(key)
        
        if data:
            try:
                return QuestionAnswerResponse(**json.loads(data))
            except Exception as e:
                logger.error(f"Failed to deserialize Q&A result: {e}")
                return None
        return None
    
    async def set_qa_result(self, document_id: str, question_hash: str, 
                           result: QuestionAnswerResponse) -> bool:
        """Cache Q&A result"""
        key = self._get_key("qa", f"{document_id}:{question_hash}")
        return await self.set(
            key,
            result.model_dump_json(),
            self.CACHE_TTL["qa_results"]
        )
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete all cached data for a document"""
        await self._ensure_connected()
        
        # Find all keys related to this document
        pattern = f"oritzpdf:*:{document_id}*"
        try:
            cursor = 0
            deleted_count = 0
            
            while True:
                cursor, keys = await self._redis.scan(cursor, match=pattern, count=100)
                if keys:
                    deleted_count += await self._redis.delete(*keys)
                
                if cursor == 0:
                    break
            
            logger.info(f"Deleted {deleted_count} cache entries for document {document_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document cache: {e}")
            return False
    
    async def get_or_compute(self, key: str, compute_func: Callable, 
                           ttl: int = 3600) -> Any:
        """Get from cache or compute if missing"""
        # Try to get from cache
        cached_value = await self.get(key)
        if cached_value:
            try:
                return json.loads(cached_value)
            except:
                return cached_value
        
        # Compute the value
        computed_value = await compute_func()
        
        # Cache the result
        await self.set(key, computed_value, ttl)
        
        return computed_value


def get_cache_service() -> CacheService:
    """Factory function to get cache service"""
    return CacheService()