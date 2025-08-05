from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache


class Settings(BaseSettings):
    # Application Settings
    APP_NAME: str = "OritzPDF"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    
    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost:5432/oritzdoc"
    
    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379"
    
    # File Storage
    STORAGE_TYPE: str = "local"  # local, s3, minio
    LOCAL_STORAGE_PATH: str = "./uploads"
    S3_BUCKET_NAME: Optional[str] = None
    S3_ACCESS_KEY: Optional[str] = None
    S3_SECRET_KEY: Optional[str] = None
    S3_ENDPOINT_URL: Optional[str] = None
    
    # OCR Settings
    OCR_ENGINE: str = "tesseract"  # tesseract, google, azure
    TESSERACT_PATH: str = "/usr/bin/tesseract"
    GOOGLE_CLOUD_VISION_KEY: Optional[str] = None
    AZURE_COMPUTER_VISION_KEY: Optional[str] = None
    AZURE_COMPUTER_VISION_ENDPOINT: Optional[str] = None
    
    # Translation
    TRANSLATION_ENGINE: str = "google"  # google, deepl, azure
    GOOGLE_TRANSLATE_KEY: Optional[str] = None
    DEEPL_AUTH_KEY: Optional[str] = None
    AZURE_TRANSLATOR_KEY: Optional[str] = None
    AZURE_TRANSLATOR_ENDPOINT: Optional[str] = None
    
    # NLP Models
    SUMMARIZATION_MODEL: str = "facebook/bart-large-cnn"
    QA_MODEL: str = "deepset/roberta-base-squad2"
    NER_MODEL: str = "en_core_web_sm"
    
    # Security
    JWT_SECRET_KEY: str = "your-secret-key-here"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    
    def validate_jwt_secret(self) -> None:
        """Validate JWT secret key security"""
        if self.JWT_SECRET_KEY == "your-secret-key-here":
            import warnings
            warnings.warn(
                "Using default JWT secret key in production is insecure. "
                "Please set a strong, random JWT_SECRET_KEY in production.",
                SecurityWarning
            )
        
        if len(self.JWT_SECRET_KEY) < 32:
            import warnings
            warnings.warn(
                "JWT secret key should be at least 32 characters long for security.",
                SecurityWarning
            )
    
    @property 
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.DEBUG or self.LOG_LEVEL == "DEBUG"
    
    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8080"]
    
    # Rate Limiting
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_PERIOD: int = 3600  # seconds
    
    # File Processing Limits
    MAX_FILE_SIZE_MB: int = 32
    MAX_PAGES_PER_DOCUMENT: int = 100
    SUPPORTED_FORMATS: str = "pdf,docx,txt,csv,html,xlsx"
    
    @property
    def supported_formats_list(self) -> List[str]:
        return [fmt.strip() for fmt in self.SUPPORTED_FORMATS.split(',')]
    
    @property
    def max_file_size_bytes(self) -> int:
        return self.MAX_FILE_SIZE_MB * 1024 * 1024
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_jwt_secret()


class SecurityWarning(UserWarning):
    """Custom warning for security-related configuration issues"""
    pass


@lru_cache()
def get_settings():
    return Settings()


settings = get_settings()