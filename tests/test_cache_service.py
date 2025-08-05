"""
Tests for cache service implementation with mocked Redis
"""
import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, Mock
import json
from datetime import datetime

from src.services.cache_service import CacheService, get_cache_service
from src.models.document import DocumentContent, ExtractedText, DocumentMetadata
from src.models.analysis import (
    SummarizationResponse, QuestionAnswerResponse,
    DataExtractionResponse, TranslationResponse
)


class TestCacheService:
    """Test cases for CacheService"""

    @pytest.mark.asyncio
    async def test_init_with_default_url(self):
        """Test CacheService initialization with default Redis URL"""
        service = CacheService()
        assert service.redis_url == "redis://localhost:6379"
        assert service._redis is None

    @pytest.mark.asyncio
    async def test_init_with_custom_url(self):
        """Test CacheService initialization with custom Redis URL"""
        custom_url = "redis://localhost:6380/2"
        service = CacheService(custom_url)
        assert service.redis_url == custom_url

    @pytest.mark.asyncio
    async def test_connect_success(self, mock_redis):
        """Test successful Redis connection"""
        service = CacheService()
        await service.connect()
        
        assert service._redis is not None
        mock_redis.get.assert_not_called()  # Connection doesn't call get

    @pytest.mark.asyncio
    async def test_disconnect_success(self, cache_service):
        """Test successful Redis disconnection"""
        # Service is already connected via fixture
        assert cache_service._redis is not None
        
        await cache_service.disconnect()
        cache_service._redis.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensure_connected_when_not_connected(self, mock_redis):
        """Test _ensure_connected creates connection when not connected"""
        service = CacheService()
        assert service._redis is None
        
        await service._ensure_connected()
        assert service._redis is not None

    @pytest.mark.asyncio
    async def test_ensure_connected_when_already_connected(self, cache_service):
        """Test _ensure_connected doesn't recreate existing connection"""
        original_redis = cache_service._redis
        
        await cache_service._ensure_connected()
        assert cache_service._redis is original_redis

    def test_get_key_format(self, cache_service):
        """Test cache key generation format"""
        key = cache_service._get_key("document", "123")
        assert key == "oritzpdf:document:123"
        
        key = cache_service._get_key("summary", "doc-456")
        assert key == "oritzpdf:summary:doc-456"

    @pytest.mark.asyncio
    async def test_get_cache_hit(self, cache_service, mock_redis):
        """Test successful cache get with hit"""
        test_key = "test:key"
        test_value = "test value"
        mock_redis.get.return_value = test_value
        
        result = await cache_service.get(test_key)
        
        assert result == test_value
        mock_redis.get.assert_called_once_with(test_key)

    @pytest.mark.asyncio
    async def test_get_cache_miss(self, cache_service, mock_redis):
        """Test cache get with miss"""
        test_key = "test:key"
        mock_redis.get.return_value = None
        
        result = await cache_service.get(test_key)
        
        assert result is None
        mock_redis.get.assert_called_once_with(test_key)

    @pytest.mark.asyncio
    async def test_get_redis_error(self, cache_service, mock_redis):
        """Test cache get handles Redis errors gracefully"""
        test_key = "test:key"
        mock_redis.get.side_effect = Exception("Redis connection error")
        
        result = await cache_service.get(test_key)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_set_string_value(self, cache_service, mock_redis):
        """Test cache set with string value"""
        test_key = "test:key"
        test_value = "test string value"
        
        result = await cache_service.set(test_key, test_value)
        
        assert result is True
        mock_redis.set.assert_called_once_with(test_key, test_value)

    @pytest.mark.asyncio
    async def test_set_object_value(self, cache_service, mock_redis):
        """Test cache set with object value (gets JSON serialized)"""
        test_key = "test:key"
        test_value = {"message": "test", "count": 42}
        
        result = await cache_service.set(test_key, test_value)
        
        assert result is True
        expected_json = json.dumps(test_value)
        mock_redis.set.assert_called_once_with(test_key, expected_json)

    @pytest.mark.asyncio
    async def test_set_with_ttl(self, cache_service, mock_redis):
        """Test cache set with TTL"""
        test_key = "test:key"
        test_value = "test value"
        ttl = 3600
        
        result = await cache_service.set(test_key, test_value, ttl)
        
        assert result is True
        mock_redis.setex.assert_called_once_with(test_key, ttl, test_value)
        mock_redis.set.assert_not_called()

    @pytest.mark.asyncio
    async def test_set_redis_error(self, cache_service, mock_redis):
        """Test cache set handles Redis errors gracefully"""
        test_key = "test:key"
        test_value = "test value"
        mock_redis.set.side_effect = Exception("Redis connection error")
        
        result = await cache_service.set(test_key, test_value)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_success(self, cache_service, mock_redis):
        """Test successful cache delete"""
        test_key = "test:key"
        mock_redis.delete.return_value = 1
        
        result = await cache_service.delete(test_key)
        
        assert result is True
        mock_redis.delete.assert_called_once_with(test_key)

    @pytest.mark.asyncio
    async def test_delete_key_not_found(self, cache_service, mock_redis):
        """Test cache delete when key doesn't exist"""
        test_key = "test:key"
        mock_redis.delete.return_value = 0
        
        result = await cache_service.delete(test_key)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_delete_redis_error(self, cache_service, mock_redis):
        """Test cache delete handles Redis errors gracefully"""
        test_key = "test:key"
        mock_redis.delete.side_effect = Exception("Redis connection error")
        
        result = await cache_service.delete(test_key)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_true(self, cache_service, mock_redis):
        """Test cache exists returns True for existing key"""
        test_key = "test:key"
        mock_redis.exists.return_value = 1
        
        result = await cache_service.exists(test_key)
        
        assert result is True
        mock_redis.exists.assert_called_once_with(test_key)

    @pytest.mark.asyncio
    async def test_exists_false(self, cache_service, mock_redis):
        """Test cache exists returns False for non-existing key"""
        test_key = "test:key"
        mock_redis.exists.return_value = 0
        
        result = await cache_service.exists(test_key)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_exists_redis_error(self, cache_service, mock_redis):
        """Test cache exists handles Redis errors gracefully"""
        test_key = "test:key"
        mock_redis.exists.side_effect = Exception("Redis connection error")
        
        result = await cache_service.exists(test_key)
        
        assert result is False


class TestCacheServiceDocumentMethods:
    """Test cases for document-specific cache methods"""

    @pytest.mark.asyncio
    async def test_get_document_content_hit(self, cache_service, mock_redis, sample_document_content):
        """Test successful document content retrieval from cache"""
        document_id = "test-doc-123"
        cached_json = sample_document_content.model_dump_json()
        mock_redis.get.return_value = cached_json
        
        result = await cache_service.get_document_content(document_id)
        
        assert result is not None
        assert result.document_id == document_id
        assert result.full_text == sample_document_content.full_text
        
        expected_key = "oritzpdf:document:test-doc-123"
        mock_redis.get.assert_called_once_with(expected_key)

    @pytest.mark.asyncio
    async def test_get_document_content_miss(self, cache_service, mock_redis):
        """Test document content cache miss"""
        document_id = "test-doc-123"
        mock_redis.get.return_value = None
        
        result = await cache_service.get_document_content(document_id)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_get_document_content_invalid_json(self, cache_service, mock_redis):
        """Test document content with invalid JSON in cache"""
        document_id = "test-doc-123"
        mock_redis.get.return_value = "invalid json"
        
        result = await cache_service.get_document_content(document_id)
        
        assert result is None

    @pytest.mark.asyncio
    async def test_set_document_content(self, cache_service, mock_redis, sample_document_content):
        """Test caching document content"""
        document_id = "test-doc-123"
        
        result = await cache_service.set_document_content(document_id, sample_document_content)
        
        assert result is True
        expected_key = "oritzpdf:document:test-doc-123"
        expected_ttl = cache_service.CACHE_TTL["document_content"]
        mock_redis.setex.assert_called_once_with(
            expected_key, expected_ttl, sample_document_content.model_dump_json()
        )

    @pytest.mark.asyncio
    async def test_get_summary_hit(self, cache_service, mock_redis, sample_summarization_response):
        """Test successful summary retrieval from cache"""
        document_id = "test-doc-123"
        cached_json = sample_summarization_response.model_dump_json()
        mock_redis.get.return_value = cached_json
        
        result = await cache_service.get_summary(document_id)
        
        assert result is not None
        assert result.document_id == document_id
        assert result.summary == sample_summarization_response.summary

    @pytest.mark.asyncio
    async def test_get_summary_with_page_range(self, cache_service, mock_redis, sample_summarization_response):
        """Test summary retrieval with page range"""
        document_id = "test-doc-123"
        page_range = "1-5"
        cached_json = sample_summarization_response.model_dump_json()
        mock_redis.get.return_value = cached_json
        
        result = await cache_service.get_summary(document_id, page_range)
        
        expected_key = "oritzpdf:summary:test-doc-123_1-5"
        mock_redis.get.assert_called_once_with(expected_key)

    @pytest.mark.asyncio
    async def test_set_summary(self, cache_service, mock_redis, sample_summarization_response):
        """Test caching summary"""
        document_id = "test-doc-123"
        
        result = await cache_service.set_summary(document_id, sample_summarization_response)
        
        assert result is True
        expected_key = "oritzpdf:summary:test-doc-123"
        expected_ttl = cache_service.CACHE_TTL["summaries"]
        mock_redis.setex.assert_called_once_with(
            expected_key, expected_ttl, sample_summarization_response.model_dump_json()
        )

    @pytest.mark.asyncio
    async def test_set_summary_with_page_range(self, cache_service, mock_redis, sample_summarization_response):
        """Test caching summary with page range"""
        document_id = "test-doc-123"
        page_range = "1-5"
        
        result = await cache_service.set_summary(document_id, sample_summarization_response, page_range)
        
        expected_key = "oritzpdf:summary:test-doc-123_1-5"
        mock_redis.setex.assert_called_once_with(
            expected_key, mock_redis.setex.call_args[0][1], sample_summarization_response.model_dump_json()
        )

    @pytest.mark.asyncio
    async def test_get_qa_result_hit(self, cache_service, mock_redis, sample_qa_response):
        """Test successful Q&A result retrieval from cache"""
        document_id = "test-doc-123"
        question_hash = "abc123"
        cached_json = sample_qa_response.model_dump_json()
        mock_redis.get.return_value = cached_json
        
        result = await cache_service.get_qa_result(document_id, question_hash)
        
        assert result is not None
        assert result.question == sample_qa_response.question
        
        expected_key = "oritzpdf:qa:test-doc-123:abc123"
        mock_redis.get.assert_called_once_with(expected_key)

    @pytest.mark.asyncio
    async def test_set_qa_result(self, cache_service, mock_redis, sample_qa_response):
        """Test caching Q&A result"""
        document_id = "test-doc-123"
        question_hash = "abc123"
        
        result = await cache_service.set_qa_result(document_id, question_hash, sample_qa_response)
        
        assert result is True
        expected_key = "oritzpdf:qa:test-doc-123:abc123"
        expected_ttl = cache_service.CACHE_TTL["qa_results"]
        mock_redis.setex.assert_called_once_with(
            expected_key, expected_ttl, sample_qa_response.model_dump_json()
        )

    @pytest.mark.asyncio
    async def test_delete_document_success(self, cache_service, mock_redis):
        """Test successful deletion of all document-related cache entries"""
        document_id = "test-doc-123"
        
        # Mock scan results - simulate finding keys
        mock_keys = [
            "oritzpdf:document:test-doc-123",
            "oritzpdf:summary:test-doc-123",
            "oritzpdf:qa:test-doc-123:hash1"
        ]
        mock_redis.scan.return_value = (0, mock_keys)  # cursor=0 means done
        mock_redis.delete.return_value = len(mock_keys)
        
        result = await cache_service.delete_document(document_id)
        
        assert result is True
        mock_redis.scan.assert_called_once_with(0, match=f"oritzpdf:*:{document_id}*", count=100)
        mock_redis.delete.assert_called_once_with(*mock_keys)

    @pytest.mark.asyncio
    async def test_delete_document_multiple_scans(self, cache_service, mock_redis):
        """Test document deletion with multiple scan iterations"""
        document_id = "test-doc-123"
        
        # Mock multiple scan iterations
        mock_redis.scan.side_effect = [
            (123, ["key1", "key2"]),  # First scan
            (0, ["key3"])             # Second scan (cursor=0 means done)
        ]
        mock_redis.delete.side_effect = [2, 1]  # Return counts for each delete
        
        result = await cache_service.delete_document(document_id)
        
        assert result is True
        assert mock_redis.scan.call_count == 2
        assert mock_redis.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_document_no_keys_found(self, cache_service, mock_redis):
        """Test document deletion when no keys are found"""
        document_id = "test-doc-123"
        mock_redis.scan.return_value = (0, [])  # No keys found
        
        result = await cache_service.delete_document(document_id)
        
        assert result is True
        mock_redis.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_document_redis_error(self, cache_service, mock_redis):
        """Test document deletion handles Redis errors gracefully"""
        document_id = "test-doc-123"
        mock_redis.scan.side_effect = Exception("Redis error")
        
        result = await cache_service.delete_document(document_id)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_hit(self, cache_service, mock_redis):
        """Test get_or_compute with cache hit"""
        test_key = "test:key"
        cached_value = '{"result": "cached"}'
        mock_redis.get.return_value = cached_value
        
        compute_func = AsyncMock()  # Should not be called
        
        result = await cache_service.get_or_compute(test_key, compute_func)
        
        assert result == {"result": "cached"}
        compute_func.assert_not_called()
        mock_redis.get.assert_called_once_with(test_key)

    @pytest.mark.asyncio
    async def test_get_or_compute_cache_miss(self, cache_service, mock_redis):
        """Test get_or_compute with cache miss"""
        test_key = "test:key"
        mock_redis.get.return_value = None
        
        computed_value = {"result": "computed"}
        compute_func = AsyncMock(return_value=computed_value)
        
        result = await cache_service.get_or_compute(test_key, compute_func, ttl=1800)
        
        assert result == computed_value
        compute_func.assert_called_once()
        mock_redis.setex.assert_called_once_with(test_key, 1800, json.dumps(computed_value))

    @pytest.mark.asyncio
    async def test_get_or_compute_string_value(self, cache_service, mock_redis):
        """Test get_or_compute with string cached value"""
        test_key = "test:key"
        cached_value = "simple string"
        mock_redis.get.return_value = cached_value
        
        compute_func = AsyncMock()
        
        result = await cache_service.get_or_compute(test_key, compute_func)
        
        assert result == "simple string"
        compute_func.assert_not_called()

    @pytest.mark.asyncio
    async def test_get_or_compute_invalid_json_fallback(self, cache_service, mock_redis):
        """Test get_or_compute handles invalid JSON gracefully"""
        test_key = "test:key"
        cached_value = "invalid json {"
        mock_redis.get.return_value = cached_value
        
        compute_func = AsyncMock()
        
        result = await cache_service.get_or_compute(test_key, compute_func)
        
        # Should return the raw string when JSON parsing fails
        assert result == "invalid json {"
        compute_func.assert_not_called()


class TestCacheServiceFactory:
    """Test cases for cache service factory function"""

    def test_get_cache_service(self):
        """Test factory function returns CacheService instance"""
        service = get_cache_service()
        assert isinstance(service, CacheService)


class TestCacheServiceTTLConfiguration:
    """Test cases for cache TTL configuration"""

    def test_cache_ttl_constants(self):
        """Test that cache TTL constants are properly defined"""
        service = CacheService()
        
        assert service.CACHE_TTL["document_content"] == 3600
        assert service.CACHE_TTL["summaries"] == 86400
        assert service.CACHE_TTL["qa_results"] == 1800
        assert service.CACHE_TTL["extracted_entities"] == 7200
        assert service.CACHE_TTL["translations"] == 604800

    @pytest.mark.asyncio
    async def test_different_ttl_values_applied(self, cache_service, mock_redis, sample_document_content, sample_summarization_response):
        """Test that different cache operations use appropriate TTL values"""
        document_id = "test-doc-123"
        
        # Test document content TTL
        await cache_service.set_document_content(document_id, sample_document_content)
        args, kwargs = mock_redis.setex.call_args
        assert args[1] == cache_service.CACHE_TTL["document_content"]
        
        # Reset mock
        mock_redis.reset_mock()
        
        # Test summary TTL
        await cache_service.set_summary(document_id, sample_summarization_response)
        args, kwargs = mock_redis.setex.call_args
        assert args[1] == cache_service.CACHE_TTL["summaries"]


class TestCacheServiceEdgeCases:
    """Test edge cases and error conditions"""

    @pytest.mark.asyncio
    async def test_very_large_object_caching(self, cache_service, mock_redis):
        """Test caching of very large objects"""
        # Create a large document content
        large_text = "x" * 100000  # 100KB text
        large_pages = [
            ExtractedText(page_number=i, text=f"Page {i} text " * 1000)
            for i in range(1, 101)  # 100 pages
        ]
        
        large_content = DocumentContent(
            document_id="large-doc",
            full_text=large_text,
            pages=large_pages,
            tables=[],
            images=[],
            metadata=DocumentMetadata()
        )
        
        result = await cache_service.set_document_content("large-doc", large_content)
        
        assert result is True
        # Verify that setex was called (meaning serialization worked)
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_unicode_in_cache_values(self, cache_service, mock_redis):
        """Test caching objects with Unicode characters"""
        unicode_content = DocumentContent(
            document_id="unicode-doc",
            full_text="测试文档 🐍 émojis and spëcial chars",
            pages=[
                ExtractedText(
                    page_number=1, 
                    text="Тест на русском языке 中文测试",
                    confidence=0.95
                )
            ],
            tables=[],
            images=[],
            metadata=DocumentMetadata(title="Título con acentos")
        )
        
        result = await cache_service.set_document_content("unicode-doc", unicode_content)
        
        assert result is True
        # Verify JSON serialization handles Unicode
        call_args = mock_redis.setex.call_args
        json_str = call_args[0][2]
        # Should not raise any encoding errors
        parsed = json.loads(json_str)
        assert "测试文档" in parsed["full_text"]

    @pytest.mark.asyncio
    async def test_none_values_handling(self, cache_service, mock_redis):
        """Test handling of None values in objects"""
        content_with_nones = DocumentContent(
            document_id="none-doc",
            full_text="Sample text",
            pages=[],
            tables=None,  # Explicitly None
            images=None,  # Explicitly None
            metadata=None  # Explicitly None
        )
        
        result = await cache_service.set_document_content("none-doc", content_with_nones)
        
        assert result is True
        mock_redis.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_document_id(self, cache_service, mock_redis):
        """Test behavior with empty document ID"""
        empty_id = ""
        
        # Should still work, just creates empty key suffix
        result = await cache_service.get_document_content(empty_id)
        
        expected_key = "oritzpdf:document:"
        mock_redis.get.assert_called_once_with(expected_key)