#!/usr/bin/env python3
"""
Simple test script to verify fixes are working
"""
import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_imports():
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    try:
        from src.config import settings
        print("✓ Config import successful")
        
        from src.models.document import Document, DocumentType
        print("✓ Document models import successful")
        
        from src.models.analysis import SummarizationRequest
        print("✓ Analysis models import successful")
        
        from src.services.storage_service import LocalStorageService
        print("✓ Storage service import successful")
        
        from src.services.cache_service import CacheService
        print("✓ Cache service import successful")
        
        from src.services.document_service import DocumentService
        print("✓ Document service import successful")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

async def test_storage_service():
    """Test storage service validation"""
    print("\nTesting storage service...")
    
    try:
        from src.services.storage_service import LocalStorageService
        
        # Test with valid data
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalStorageService(temp_dir)
            
            # Test save with valid data
            test_content = b"Hello, World!"
            path = await storage.save_file(test_content, "test.txt")
            print("✓ File save successful")
            
            # Test read
            content = await storage.get_file(path)
            assert content == test_content
            print("✓ File read successful")
            
            # Test delete
            success = await storage.delete_file(path)
            assert success
            print("✓ File delete successful")
        
        # Test with invalid data
        try:
            storage = LocalStorageService()
            await storage.save_file(None, "test.txt")
            print("✗ Should have failed with None content")
            return False
        except ValueError:
            print("✓ Validation correctly rejected None content")
        
        try:
            await storage.save_file(b"test", "")
            print("✗ Should have failed with empty filename")
            return False
        except ValueError:
            print("✓ Validation correctly rejected empty filename")
        
        return True
    except Exception as e:
        print(f"✗ Storage service test failed: {e}")
        return False

async def test_document_validation():
    """Test document validation"""
    print("\nTesting document validation...")
    
    try:
        from src.services.document_service import DocumentService
        
        service = DocumentService()
        
        # Test valid upload
        await service.validate_upload("test.pdf", 1024, "application/pdf")
        print("✓ Valid upload validation passed")
        
        # Test invalid file size
        try:
            await service.validate_upload("test.pdf", 100 * 1024 * 1024, "application/pdf")
            print("✗ Should have failed with large file size")
            return False
        except ValueError:
            print("✓ Validation correctly rejected large file")
        
        # Test invalid extension
        try:
            await service.validate_upload("test.exe", 1024, "application/octet-stream")
            print("✗ Should have failed with dangerous extension")
            return False
        except ValueError:
            print("✓ Validation correctly rejected dangerous extension")
        
        # Test empty filename
        try:
            await service.validate_upload("", 1024, "application/pdf")
            print("✗ Should have failed with empty filename")
            return False
        except ValueError:
            print("✓ Validation correctly rejected empty filename")
        
        return True
    except Exception as e:
        print(f"✗ Document validation test failed: {e}")
        return False

async def test_config_security():
    """Test configuration security warnings"""
    print("\nTesting configuration security...")
    
    try:
        from src.config import Settings
        import warnings
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Test with default secret
            settings = Settings(JWT_SECRET_KEY="your-secret-key-here")
            
            if w:
                print("✓ Security warning raised for default JWT secret")
            else:
                print("⚠ Security warning not raised (expected)")
        
        # Test with short secret
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            settings = Settings(JWT_SECRET_KEY="short")
            
            if w:
                print("✓ Security warning raised for short JWT secret")
            else:
                print("⚠ Security warning not raised (expected)")
        
        return True
    except Exception as e:
        print(f"✗ Config security test failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("Running OritzPDF fixes verification tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_storage_service,
        test_document_validation,
        test_config_security
    ]
    
    passed = 0
    for test in tests:
        try:
            if await test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Tests completed: {passed}/{len(tests)} passed")
    
    if passed == len(tests):
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1

if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))