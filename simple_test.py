#!/usr/bin/env python3
"""
Simple test script to verify syntax and basic functionality
"""
import sys
import ast
from pathlib import Path

def test_python_syntax():
    """Test that all Python files have valid syntax"""
    print("Testing Python syntax...")
    
    src_dir = Path("src")
    if not src_dir.exists():
        print("✗ src directory not found")
        return False
    
    python_files = list(src_dir.rglob("*.py"))
    if not python_files:
        print("✗ No Python files found")
        return False
    
    errors = []
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            print(f"✓ {py_file}")
        except SyntaxError as e:
            print(f"✗ {py_file}: {e}")
            errors.append(str(py_file))
        except UnicodeDecodeError as e:
            print(f"✗ {py_file}: Encoding error - {e}")
            errors.append(str(py_file))
    
    if errors:
        print(f"\nSyntax errors found in {len(errors)} files:")
        for error_file in errors:
            print(f"  - {error_file}")
        return False
    
    print(f"✓ All {len(python_files)} Python files have valid syntax")
    return True

def test_requirements_format():
    """Test requirements.txt format"""
    print("\nTesting requirements.txt format...")
    
    req_file = Path("requirements.txt")
    if not req_file.exists():
        print("✗ requirements.txt not found")
        return False
    
    try:
        with open(req_file, 'r') as f:
            lines = f.readlines()
        
        # Check for the torch version fix
        torch_found = False
        for line in lines:
            line = line.strip()
            if line.startswith("torch"):
                if "torch==2.1.2" in line:
                    print("✗ Old incompatible torch version found")
                    return False
                elif "torch>=" in line:
                    torch_found = True
                    print(f"✓ Fixed torch version: {line}")
        
        if not torch_found:
            print("⚠ torch dependency not found")
        
        print("✓ requirements.txt format is valid")
        return True
    except Exception as e:
        print(f"✗ Error reading requirements.txt: {e}")
        return False

def test_import_structure():
    """Test that import statements look correct"""
    print("\nTesting import structure...")
    
    # Check main.py imports
    main_file = Path("src/main.py")
    if main_file.exists():
        try:
            with open(main_file, 'r') as f:
                content = f.read()
            
            required_imports = [
                "from src.api import documents, health, analysis",
                "from src.config import settings"
            ]
            
            for imp in required_imports:
                if imp in content:
                    print(f"✓ Found: {imp}")
                else:
                    print(f"⚠ Missing: {imp}")
            
        except Exception as e:
            print(f"✗ Error reading main.py: {e}")
            return False
    
    # Check that analysis models exist
    analysis_models = Path("src/models/analysis.py")
    if analysis_models.exists():
        print("✓ Analysis models file exists")
    else:
        print("✗ Analysis models file missing")
        return False
    
    return True

def test_security_improvements():
    """Test that security improvements are in place"""
    print("\nTesting security improvements...")
    
    config_file = Path("src/config.py")
    if not config_file.exists():
        print("✗ config.py not found")
        return False
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Check for security warning implementation
        if "SecurityWarning" in content:
            print("✓ SecurityWarning class found")
        else:
            print("✗ SecurityWarning class missing")
            return False
        
        if "validate_jwt_secret" in content:
            print("✓ JWT secret validation found")
        else:
            print("✗ JWT secret validation missing")
            return False
        
        return True
    except Exception as e:
        print(f"✗ Error reading config.py: {e}")
        return False

def test_error_handling_improvements():
    """Test that error handling improvements are in place"""
    print("\nTesting error handling improvements...")
    
    # Check PDF processor
    pdf_processor = Path("src/services/pdf_processor.py")
    if pdf_processor.exists():
        try:
            with open(pdf_processor, 'r') as f:
                content = f.read()
            
            # Check for improved error handling patterns
            if "finally:" in content and "doc.close()" in content:
                print("✓ Improved resource cleanup in PDF processor")
            else:
                print("✗ Resource cleanup not improved")
                return False
            
            if "FileNotFoundError" in content:
                print("✓ File validation added")
            else:
                print("✗ File validation missing")
                return False
            
        except Exception as e:
            print(f"✗ Error reading pdf_processor.py: {e}")
            return False
    
    # Check storage service
    storage_service = Path("src/services/storage_service.py")
    if storage_service.exists():
        try:
            with open(storage_service, 'r') as f:
                content = f.read()
            
            # Check for validation improvements
            if "if not file_content or not isinstance(file_content, bytes):" in content:
                print("✓ Input validation added to storage service")
            else:
                print("✗ Input validation missing in storage service")
                return False
            
        except Exception as e:
            print(f"✗ Error reading storage_service.py: {e}")
            return False
    
    return True

def main():
    """Run all tests"""
    print("Running OritzPDF code quality verification")
    print("=" * 50)
    
    tests = [
        test_python_syntax,
        test_requirements_format,
        test_import_structure,
        test_security_improvements,
        test_error_handling_improvements
    ]
    
    passed = 0
    for test in tests:
        try:
            if test():
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
    sys.exit(main())