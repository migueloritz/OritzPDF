"""
Tests for configuration loading and settings
"""
import pytest
import os
from unittest.mock import patch, mock_open
from pathlib import Path
import tempfile

from src.config import Settings, get_settings


class TestSettings:
    """Test cases for Settings configuration"""

    def test_default_settings(self):
        """Test default configuration values"""
        settings = Settings()
        
        # Application Settings
        assert settings.APP_NAME == "OritzPDF"
        assert settings.APP_VERSION == "1.0.0"
        assert settings.DEBUG is False
        assert settings.LOG_LEVEL == "INFO"
        
        # API Configuration
        assert settings.API_HOST == "0.0.0.0"
        assert settings.API_PORT == 8000
        assert settings.API_PREFIX == "/api/v1"
        
        # Database
        assert settings.DATABASE_URL == "postgresql+asyncpg://user:password@localhost:5432/oritzdoc"
        
        # Redis Cache
        assert settings.REDIS_URL == "redis://localhost:6379"
        
        # File Storage
        assert settings.STORAGE_TYPE == "local"
        assert settings.LOCAL_STORAGE_PATH == "./uploads"
        
        # Security
        assert settings.JWT_SECRET_KEY == "your-secret-key-here"
        assert settings.JWT_ALGORITHM == "HS256"
        assert settings.JWT_EXPIRATION_HOURS == 24
        
        # File Processing Limits
        assert settings.MAX_FILE_SIZE_MB == 32
        assert settings.MAX_PAGES_PER_DOCUMENT == 100
        assert settings.SUPPORTED_FORMATS == "pdf,docx,txt,csv,html,xlsx"

    def test_environment_variable_override(self):
        """Test that environment variables override default settings"""
        with patch.dict(os.environ, {
            'APP_NAME': 'TestApp',
            'DEBUG': 'true',
            'API_PORT': '9000',
            'MAX_FILE_SIZE_MB': '64',
            'REDIS_URL': 'redis://test:6379',
            'STORAGE_TYPE': 's3'
        }):
            settings = Settings()
            
            assert settings.APP_NAME == "TestApp"
            assert settings.DEBUG is True
            assert settings.API_PORT == 9000
            assert settings.MAX_FILE_SIZE_MB == 64
            assert settings.REDIS_URL == "redis://test:6379"
            assert settings.STORAGE_TYPE == "s3"

    def test_boolean_environment_variables(self):
        """Test boolean environment variable parsing"""
        test_cases = [
            ('true', True),
            ('True', True),
            ('TRUE', True),
            ('1', True),
            ('yes', True),
            ('false', False),
            ('False', False),
            ('FALSE', False),
            ('0', False),
            ('no', False),
            ('', False)
        ]
        
        for env_value, expected in test_cases:
            with patch.dict(os.environ, {'DEBUG': env_value}):
                settings = Settings()
                assert settings.DEBUG is expected, f"Failed for env_value: {env_value}"

    def test_integer_environment_variables(self):
        """Test integer environment variable parsing"""
        with patch.dict(os.environ, {
            'API_PORT': '8080',
            'MAX_FILE_SIZE_MB': '128',
            'MAX_PAGES_PER_DOCUMENT': '500',
            'JWT_EXPIRATION_HOURS': '48'
        }):
            settings = Settings()
            
            assert settings.API_PORT == 8080
            assert settings.MAX_FILE_SIZE_MB == 128
            assert settings.MAX_PAGES_PER_DOCUMENT == 500
            assert settings.JWT_EXPIRATION_HOURS == 48

    def test_list_environment_variables(self):
        """Test list environment variable parsing"""
        with patch.dict(os.environ, {
            'CORS_ORIGINS': 'http://localhost:3000,http://localhost:8080,https://example.com'
        }):
            settings = Settings()
            
            expected_origins = [
                "http://localhost:3000",
                "http://localhost:8080", 
                "https://example.com"
            ]
            assert settings.CORS_ORIGINS == expected_origins

    def test_optional_environment_variables(self):
        """Test optional environment variables"""
        # Test with None values
        settings = Settings()
        
        assert settings.S3_BUCKET_NAME is None
        assert settings.S3_ACCESS_KEY is None
        assert settings.S3_SECRET_KEY is None
        assert settings.S3_ENDPOINT_URL is None
        assert settings.GOOGLE_CLOUD_VISION_KEY is None
        assert settings.AZURE_COMPUTER_VISION_KEY is None
        
        # Test with actual values
        with patch.dict(os.environ, {
            'S3_BUCKET_NAME': 'test-bucket',
            'S3_ACCESS_KEY': 'test-access-key',
            'S3_SECRET_KEY': 'test-secret-key',
            'GOOGLE_CLOUD_VISION_KEY': 'google-key',
            'AZURE_COMPUTER_VISION_KEY': 'azure-key'
        }):
            settings = Settings()
            
            assert settings.S3_BUCKET_NAME == "test-bucket"
            assert settings.S3_ACCESS_KEY == "test-access-key"
            assert settings.S3_SECRET_KEY == "test-secret-key"
            assert settings.GOOGLE_CLOUD_VISION_KEY == "google-key"
            assert settings.AZURE_COMPUTER_VISION_KEY == "azure-key"

    def test_dotenv_file_loading(self):
        """Test loading configuration from .env file"""
        # Create temporary .env file
        env_content = """
APP_NAME=EnvTestApp
DEBUG=true
API_PORT=7000
REDIS_URL=redis://env:6379
MAX_FILE_SIZE_MB=16
S3_BUCKET_NAME=env-bucket
        """.strip()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            # Mock the env_file path in Settings Config
            with patch.object(Settings.Config, 'env_file', env_file_path):
                settings = Settings()
                
                assert settings.APP_NAME == "EnvTestApp"
                assert settings.DEBUG is True
                assert settings.API_PORT == 7000
                assert settings.REDIS_URL == "redis://env:6379"
                assert settings.MAX_FILE_SIZE_MB == 16
                assert settings.S3_BUCKET_NAME == "env-bucket"
        finally:
            # Clean up
            os.unlink(env_file_path)

    def test_case_sensitive_settings(self):
        """Test that settings are case sensitive"""
        with patch.dict(os.environ, {
            'app_name': 'lowercase',  # Should not override APP_NAME
            'APP_NAME': 'UPPERCASE'   # Should override
        }):
            settings = Settings()
            
            assert settings.APP_NAME == "UPPERCASE"

    def test_supported_formats_list_property(self):
        """Test supported_formats_list property parsing"""
        settings = Settings()
        
        expected_formats = ["pdf", "docx", "txt", "csv", "html", "xlsx"]
        assert settings.supported_formats_list == expected_formats
        
        # Test with custom formats
        with patch.dict(os.environ, {
            'SUPPORTED_FORMATS': 'pdf,docx,png, jpg ,txt'  # Test whitespace handling
        }):
            settings = Settings()
            expected_custom = ["pdf", "docx", "png", "jpg", "txt"]
            assert settings.supported_formats_list == expected_custom

    def test_max_file_size_bytes_property(self):
        """Test max_file_size_bytes property calculation"""
        settings = Settings()
        
        expected_bytes = 32 * 1024 * 1024  # 32MB in bytes
        assert settings.max_file_size_bytes == expected_bytes
        
        # Test with custom size
        with patch.dict(os.environ, {'MAX_FILE_SIZE_MB': '64'}):
            settings = Settings()
            expected_custom = 64 * 1024 * 1024
            assert settings.max_file_size_bytes == expected_custom

    def test_invalid_environment_values(self):
        """Test handling of invalid environment variable values"""
        # Test invalid integer
        with patch.dict(os.environ, {'API_PORT': 'invalid'}):
            with pytest.raises(ValueError):
                Settings()
        
        # Test invalid boolean (should default to False for most invalid values)
        with patch.dict(os.environ, {'DEBUG': 'invalid'}):
            settings = Settings()
            assert settings.DEBUG is False

    def test_database_url_variations(self):
        """Test different database URL formats"""
        test_urls = [
            "postgresql://user:pass@localhost/db",
            "postgresql+asyncpg://user:pass@localhost:5432/db",
            "sqlite:///./test.db",
            "mysql://user:pass@localhost/db"
        ]
        
        for url in test_urls:
            with patch.dict(os.environ, {'DATABASE_URL': url}):
                settings = Settings()
                assert settings.DATABASE_URL == url

    def test_cors_origins_default(self):
        """Test default CORS origins"""
        settings = Settings()
        
        expected_default = ["http://localhost:3000", "http://localhost:8080"]
        assert settings.CORS_ORIGINS == expected_default

    def test_ocr_engine_options(self):
        """Test OCR engine configuration options"""
        ocr_engines = ["tesseract", "google", "azure"]
        
        for engine in ocr_engines:
            with patch.dict(os.environ, {'OCR_ENGINE': engine}):
                settings = Settings()
                assert settings.OCR_ENGINE == engine

    def test_translation_engine_options(self):
        """Test translation engine configuration options"""
        translation_engines = ["google", "deepl", "azure"]
        
        for engine in translation_engines:
            with patch.dict(os.environ, {'TRANSLATION_ENGINE': engine}):
                settings = Settings()
                assert settings.TRANSLATION_ENGINE == engine

    def test_nlp_model_configurations(self):
        """Test NLP model configuration settings"""
        with patch.dict(os.environ, {
            'SUMMARIZATION_MODEL': 'custom/summarization-model',
            'QA_MODEL': 'custom/qa-model',
            'NER_MODEL': 'custom_ner_model'
        }):
            settings = Settings()
            
            assert settings.SUMMARIZATION_MODEL == "custom/summarization-model"
            assert settings.QA_MODEL == "custom/qa-model"
            assert settings.NER_MODEL == "custom_ner_model"

    def test_rate_limiting_configuration(self):
        """Test rate limiting configuration"""
        with patch.dict(os.environ, {
            'RATE_LIMIT_REQUESTS': '200',
            'RATE_LIMIT_PERIOD': '7200'
        }):
            settings = Settings()
            
            assert settings.RATE_LIMIT_REQUESTS == 200
            assert settings.RATE_LIMIT_PERIOD == 7200

    def test_security_configuration(self):
        """Test security-related configuration"""
        with patch.dict(os.environ, {
            'JWT_SECRET_KEY': 'super-secret-key-for-testing',
            'JWT_ALGORITHM': 'HS512',
            'JWT_EXPIRATION_HOURS': '72'
        }):
            settings = Settings()
            
            assert settings.JWT_SECRET_KEY == "super-secret-key-for-testing"
            assert settings.JWT_ALGORITHM == "HS512"
            assert settings.JWT_EXPIRATION_HOURS == 72

    def test_storage_configuration_local(self):
        """Test local storage configuration"""
        with patch.dict(os.environ, {
            'STORAGE_TYPE': 'local',
            'LOCAL_STORAGE_PATH': '/custom/storage/path'
        }):
            settings = Settings()
            
            assert settings.STORAGE_TYPE == "local"
            assert settings.LOCAL_STORAGE_PATH == "/custom/storage/path"

    def test_storage_configuration_s3(self):
        """Test S3 storage configuration"""
        with patch.dict(os.environ, {
            'STORAGE_TYPE': 's3',
            'S3_BUCKET_NAME': 'my-pdf-bucket',
            'S3_ACCESS_KEY': 'AKIAIOSFODNN7EXAMPLE',
            'S3_SECRET_KEY': 'wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY',
            'S3_ENDPOINT_URL': 'https://s3.amazonaws.com'
        }):
            settings = Settings()
            
            assert settings.STORAGE_TYPE == "s3"
            assert settings.S3_BUCKET_NAME == "my-pdf-bucket"
            assert settings.S3_ACCESS_KEY == "AKIAIOSFODNN7EXAMPLE"
            assert settings.S3_SECRET_KEY == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
            assert settings.S3_ENDPOINT_URL == "https://s3.amazonaws.com"


class TestGetSettings:
    """Test cases for get_settings factory function"""

    def test_get_settings_returns_same_instance(self):
        """Test that get_settings returns the same instance (caching)"""
        settings1 = get_settings()
        settings2 = get_settings()
        
        assert settings1 is settings2  # Same instance due to lru_cache

    def test_get_settings_type(self):
        """Test that get_settings returns Settings instance"""
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_with_environment_changes(self):
        """Test get_settings behavior with environment changes"""
        # First call
        settings1 = get_settings()
        original_app_name = settings1.APP_NAME
        
        # Change environment
        with patch.dict(os.environ, {'APP_NAME': 'ChangedApp'}):
            # Due to lru_cache, this should still return the original instance
            settings2 = get_settings()
            assert settings2.APP_NAME == original_app_name  # Cached value
            assert settings1 is settings2

    def test_get_settings_cache_clear(self):
        """Test clearing get_settings cache"""
        settings1 = get_settings()
        
        # Clear the cache
        get_settings.cache_clear()
        
        # Should return new instance
        settings2 = get_settings()
        assert settings1 is not settings2  # Different instances after cache clear

    def test_get_settings_concurrent_access(self):
        """Test concurrent access to get_settings"""
        import threading
        import time
        
        results = []
        
        def get_settings_worker():
            time.sleep(0.01)  # Small delay to simulate concurrent access
            settings = get_settings()
            results.append(settings)
        
        # Clear cache first
        get_settings.cache_clear()
        
        # Create multiple threads
        threads = [threading.Thread(target=get_settings_worker) for _ in range(10)]
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All should return the same instance
        assert len(results) == 10
        first_instance = results[0]
        for result in results:
            assert result is first_instance


class TestConfigurationValidation:
    """Test cases for configuration validation"""

    def test_required_secret_key_warning(self):
        """Test that using default secret key should be avoided in production"""
        settings = Settings()
        
        # In a real implementation, you might want to add validation
        # that warns or raises an error if using the default secret key
        if not settings.DEBUG:
            assert settings.JWT_SECRET_KEY != "your-secret-key-here", \
                "Default secret key should not be used in production"

    def test_database_url_format(self):
        """Test database URL format validation"""
        valid_urls = [
            "postgresql://user:pass@localhost/db",
            "postgresql+asyncpg://user:pass@localhost:5432/db",
            "sqlite:///./test.db"
        ]
        
        for url in valid_urls:
            with patch.dict(os.environ, {'DATABASE_URL': url}):
                settings = Settings()
                assert settings.DATABASE_URL == url

    def test_redis_url_format(self):
        """Test Redis URL format validation"""
        valid_urls = [
            "redis://localhost:6379",
            "redis://localhost:6379/0",
            "redis://user:pass@localhost:6379/1",
            "rediss://localhost:6379"  # SSL
        ]
        
        for url in valid_urls:
            with patch.dict(os.environ, {'REDIS_URL': url}):
                settings = Settings()
                assert settings.REDIS_URL == url

    def test_file_size_limits(self):
        """Test file size limit validation"""
        # Test reasonable limits
        reasonable_limits = [1, 10, 32, 64, 128]
        
        for limit in reasonable_limits:
            with patch.dict(os.environ, {'MAX_FILE_SIZE_MB': str(limit)}):
                settings = Settings()
                assert settings.MAX_FILE_SIZE_MB == limit
                assert settings.max_file_size_bytes == limit * 1024 * 1024

    def test_supported_formats_validation(self):
        """Test supported formats validation"""
        # Test various format combinations
        format_combinations = [
            "pdf",
            "pdf,docx",
            "pdf,docx,txt,csv,html,xlsx",
            "pdf, docx, txt",  # With spaces
            "PDF,DOCX"  # Case doesn't matter for parsing
        ]
        
        for formats in format_combinations:
            with patch.dict(os.environ, {'SUPPORTED_FORMATS': formats}):
                settings = Settings()
                assert isinstance(settings.supported_formats_list, list)
                assert len(settings.supported_formats_list) > 0

    def test_port_range_validation(self):
        """Test API port range validation"""
        valid_ports = [80, 443, 8000, 8080, 9000]
        
        for port in valid_ports:
            with patch.dict(os.environ, {'API_PORT': str(port)}):
                settings = Settings()
                assert settings.API_PORT == port
                assert 1 <= settings.API_PORT <= 65535


class TestConfigurationEdgeCases:
    """Test edge cases and error conditions"""

    def test_empty_environment_variables(self):
        """Test handling of empty environment variables"""
        with patch.dict(os.environ, {
            'APP_NAME': '',
            'REDIS_URL': '',
            'SUPPORTED_FORMATS': ''
        }):
            settings = Settings()
            
            # Empty strings should be treated as None/default for some fields
            assert settings.APP_NAME == ""  # String field keeps empty value
            assert settings.REDIS_URL == ""  # String field keeps empty value

    def test_very_large_numbers(self):
        """Test handling of very large number values"""
        with patch.dict(os.environ, {
            'MAX_FILE_SIZE_MB': '999999',
            'MAX_PAGES_PER_DOCUMENT': '999999'
        }):
            settings = Settings()
            
            assert settings.MAX_FILE_SIZE_MB == 999999
            assert settings.MAX_PAGES_PER_DOCUMENT == 999999

    def test_special_characters_in_strings(self):
        """Test handling of special characters in string values"""
        special_values = {
            'APP_NAME': 'App-Name_With.Special@Characters!',
            'JWT_SECRET_KEY': 'key!@#$%^&*()_+-=[]{}|;:,.<>?',
            'DATABASE_URL': 'postgresql://user:p@ss!@localhost/db-name_test'
        }
        
        with patch.dict(os.environ, special_values):
            settings = Settings()
            
            for key, value in special_values.items():
                assert getattr(settings, key) == value

    def test_unicode_in_configuration(self):
        """Test handling of Unicode characters in configuration"""
        with patch.dict(os.environ, {
            'APP_NAME': 'OritzPDF-测试',
            'LOCAL_STORAGE_PATH': './uploads/文档'
        }):
            settings = Settings()
            
            assert settings.APP_NAME == "OritzPDF-测试"
            assert settings.LOCAL_STORAGE_PATH == "./uploads/文档"

    def test_configuration_serialization(self):
        """Test that configuration can be serialized (for logging/debugging)"""
        settings = Settings()
        
        # Should be able to convert to dict
        settings_dict = settings.model_dump()
        assert isinstance(settings_dict, dict)
        assert 'APP_NAME' in settings_dict
        
        # Should be able to convert to JSON
        settings_json = settings.model_dump_json()
        assert isinstance(settings_json, str)
        
        # Parse back
        import json
        parsed = json.loads(settings_json)
        assert parsed['APP_NAME'] == settings.APP_NAME