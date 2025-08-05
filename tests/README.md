# OritzPDF Test Suite

This directory contains comprehensive tests for the OritzPDF document assistant project. The test suite is designed to achieve at least 80% code coverage and ensures reliability of all components.

## Test Structure

```
tests/
├── __init__.py                 # Test package initialization
├── conftest.py                 # Pytest configuration and shared fixtures
├── test_storage_service.py     # Storage service tests (local & S3)
├── test_pdf_processor.py       # PDF processing tests
├── test_cache_service.py       # Redis cache service tests
├── test_document_service.py    # Document service integration tests
├── test_api_documents.py       # Documents API endpoint tests
├── test_api_analysis.py        # Analysis API endpoint tests
├── test_config.py              # Configuration loading tests
├── utils/
│   ├── __init__.py
│   └── test_helpers.py         # Test utilities and helpers
└── README.md                   # This file
```

## Test Categories

### Unit Tests
- **Storage Service**: Tests for local filesystem and S3 storage implementations
- **PDF Processor**: Tests for PDF text/metadata/table/image extraction
- **Cache Service**: Tests for Redis-based caching with mocked Redis
- **Configuration**: Tests for settings loading and validation

### Integration Tests
- **Document Service**: End-to-end document processing workflows
- **API Endpoints**: FastAPI route testing with mocked dependencies

### Test Features
- **Mocked External Dependencies**: Redis, S3, PDF libraries, NLP models
- **Async Testing**: Full async/await support with pytest-asyncio
- **Error Scenarios**: Comprehensive error handling testing
- **Performance Testing**: Large file and concurrent operation testing
- **Edge Cases**: Unicode, empty files, malformed data

## Running Tests

### Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Using the Test Runner
```bash
# Run all tests
python run_tests.py

# Run specific test types
python run_tests.py --type unit
python run_tests.py --type integration
python run_tests.py --type api

# Run with coverage
python run_tests.py --coverage

# Run specific test file
python run_tests.py --file test_storage_service.py

# Run specific test function
python run_tests.py --function test_upload_document_success

# Skip slow tests for faster feedback
python run_tests.py --fast

# Install dependencies and run tests
python run_tests.py --install-deps
```

### Test Selection with Markers
```bash
# Run only storage-related tests
python -m pytest -m storage

# Run only fast tests (skip slow ones)
python -m pytest -m "not slow"

# Run only PDF processing tests
python -m pytest -m pdf

# Run only API tests
python -m pytest -m api
```

## Test Configuration

### Pytest Configuration (`pytest.ini`)
- Minimum 80% code coverage requirement
- Async test support
- Custom markers for test categorization
- Coverage reporting (terminal, HTML, XML)
- Test timeout settings

### Environment Variables for Testing
```bash
# Override settings for testing
export REDIS_URL="redis://localhost:6379/1"
export STORAGE_TYPE="local"
export LOCAL_STORAGE_PATH="./test_uploads"
export DEBUG="true"
```

## Fixtures and Test Utilities

### Key Fixtures (`conftest.py`)
- `test_settings`: Test configuration settings
- `temp_dir`: Temporary directory for file operations
- `sample_pdf_content`: Minimal valid PDF bytes
- `mock_redis`: Mocked Redis client
- `mock_s3_client`: Mocked S3 client
- `cache_service`: Configured cache service with mocks
- `document_service`: Document service with mocked dependencies
- `sample_document*`: Various test data objects

### Test Helpers (`utils/test_helpers.py`)
- `TestDataFactory`: Factory for creating test data objects
- `MockFactory`: Factory for creating mock objects
- `PDFTestHelper`: PDF file creation utilities
- `FileTestHelper`: File and directory management
- `AsyncTestHelper`: Async testing utilities
- `ValidationHelper`: Assertion helpers
- `PerformanceTestHelper`: Performance testing utilities

## Writing New Tests

### Test Naming Convention
```python
class TestServiceName:
    """Test cases for ServiceName"""
    
    def test_method_name_success(self):
        """Test successful method execution"""
        pass
    
    def test_method_name_error_condition(self):
        """Test method with specific error condition"""
        pass
    
    @pytest.mark.slow
    def test_method_name_performance(self):
        """Test method performance with large data"""
        pass
```

### Using Fixtures
```python
def test_storage_operation(self, local_storage_service, sample_pdf_content):
    """Test using pre-configured fixtures"""
    # Use the fixtures directly
    storage_path = await local_storage_service.save_file(
        sample_pdf_content, "test.pdf"
    )
    assert storage_path is not None
```

### Async Testing
```python
@pytest.mark.asyncio
async def test_async_operation(self):
    """Test async operations"""
    result = await some_async_function()
    assert result is not None
```

### Mocking External Services
```python
@patch('src.services.storage_service.aioboto3.Session')
def test_s3_operation(self, mock_session):
    """Test with mocked S3"""
    # Configure mock
    mock_client = AsyncMock()
    mock_session.return_value.client.return_value.__aenter__.return_value = mock_client
    
    # Test implementation
    service = S3StorageService()
    # ... test code
```

## Test Data Management

### Creating Test Documents
```python
from tests.utils.test_helpers import TestDataFactory

# Create test document
document = TestDataFactory.create_document(
    filename="custom.pdf",
    status=DocumentStatus.COMPLETED
)

# Create test content
content = TestDataFactory.create_document_content(
    document_id="test-123",
    full_text="Custom test content",
    num_pages=3
)
```

### File Testing
```python
from tests.utils.test_helpers import TemporaryFile, create_minimal_pdf

# Use temporary files
with TemporaryFile(create_minimal_pdf(), ".pdf") as pdf_path:
    # Test with the temporary PDF file
    result = await process_pdf(pdf_path)
```

## Coverage Requirements

The test suite maintains a minimum of 80% code coverage across all modules:

- **Storage Services**: 90%+ coverage
- **PDF Processing**: 85%+ coverage  
- **Cache Service**: 90%+ coverage
- **Document Service**: 85%+ coverage
- **API Endpoints**: 90%+ coverage
- **Configuration**: 95%+ coverage

## Continuous Integration

### GitHub Actions Workflow
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      - name: Run tests
        run: |
          python run_tests.py --coverage
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Debugging Tests

### Running Individual Tests
```bash
# Run single test with verbose output
python -m pytest tests/test_storage_service.py::TestLocalStorageService::test_save_file_success -v

# Run with debugging
python -m pytest tests/test_storage_service.py -v -s --tb=long

# Run with pdb on failure
python -m pytest tests/test_storage_service.py --pdb
```

### Common Issues

1. **Async Test Failures**: Ensure `@pytest.mark.asyncio` decorator is used
2. **Mock Not Called**: Check that mocks are properly configured and awaited
3. **File Permission Errors**: Use temporary directories with proper cleanup
4. **Redis Connection**: Ensure Redis mocks are properly set up in fixtures

## Performance Testing

### Large File Testing
```python
def test_large_file_processing(self):
    """Test processing of large files"""
    large_content = b"x" * (10 * 1024 * 1024)  # 10MB
    # ... test implementation
```

### Concurrent Testing
```python
async def test_concurrent_operations(self):
    """Test concurrent document processing"""
    tasks = [process_document(f"doc-{i}") for i in range(10)]
    results = await asyncio.gather(*tasks)
    assert all(result.status == DocumentStatus.COMPLETED for result in results)
```

## Best Practices

1. **Use Fixtures**: Leverage shared fixtures for common test data
2. **Mock External Services**: Always mock Redis, S3, databases, APIs
3. **Test Error Conditions**: Include negative test cases
4. **Clean Up Resources**: Use context managers and fixtures for cleanup
5. **Descriptive Names**: Use clear, descriptive test and method names
6. **Test Isolation**: Ensure tests can run independently
7. **Performance Awareness**: Mark slow tests appropriately
8. **Documentation**: Include docstrings explaining test purpose

## Contributing

When adding new tests:

1. Follow the existing naming conventions
2. Add appropriate markers (`@pytest.mark.unit`, `@pytest.mark.slow`, etc.)
3. Use existing fixtures when possible
4. Mock all external dependencies
5. Include both success and failure scenarios
6. Update this README if adding new test categories

## Troubleshooting

### Common Test Failures

1. **Import Errors**: Ensure `PYTHONPATH` includes project root
2. **Fixture Not Found**: Check fixture is defined in `conftest.py`
3. **Async Issues**: Use `pytest-asyncio` and proper decorators
4. **Mock Issues**: Verify mocks are configured before use
5. **File Cleanup**: Use proper cleanup in fixtures and context managers

For more specific issues, check the test logs and ensure all dependencies are installed correctly.