# OritzPDF Project Context

## Project Overview
OritzPDF is an intelligent document assistant that analyzes, summarizes, and interacts with PDF and document files using natural language. It emulates core functions of Adobe Acrobat Pro while introducing conversational capabilities.

## Architecture
- **Backend**: FastAPI (Python)
- **PDF Processing**: PyMuPDF for performance, pypdf for open-source compatibility
- **OCR**: Hybrid approach with Tesseract (free) and cloud services (Google Vision, Azure)
- **NLP**: Hugging Face Transformers for summarization/QA, spaCy for entity extraction
- **Storage**: Local filesystem with S3/MinIO support
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Cache**: Redis for performance optimization

## Key Design Decisions
1. **Microservices Architecture**: Separate services for document processing, NLP, and comparison
2. **Async Processing**: Use FastAPI's async capabilities for high performance
3. **Hybrid OCR**: Combine Tesseract's layout detection with cloud OCR for accuracy
4. **Caching Strategy**: Cache processed documents and NLP results in Redis
5. **Error Handling**: Implement fallback processors for robustness

## Implementation Guidelines
- Follow PEP 8 style guide
- Use type hints for all function parameters and returns
- Implement comprehensive error handling with proper logging
- Write unit tests for all business logic
- Document all API endpoints with OpenAPI schemas
- Use dependency injection for services
- Implement proper input validation with Pydantic

## Testing Requirements
- Minimum 80% code coverage
- Unit tests for all services
- Integration tests for API endpoints
- Performance tests for document processing
- Run linting and type checking before commits

## Security Considerations
- Validate all file uploads (type, size, content)
- Implement rate limiting
- Use JWT for authentication
- Sanitize all user inputs
- Store sensitive configuration in environment variables
- Never log sensitive information

## Performance Targets
- Process 10-page PDF in < 5 seconds
- Support concurrent processing of 10 documents
- API response time < 200ms for cached queries
- Memory usage < 1GB per worker process