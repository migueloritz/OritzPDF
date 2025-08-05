#!/usr/bin/env python3
"""
Test runner script for OritzPDF
"""
import sys
import subprocess
import argparse
import os
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result"""
    print(f"\n🔄 {description}")
    print(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Exit code: {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Run OritzPDF tests")
    parser.add_argument(
        "--type", 
        choices=["unit", "integration", "api", "all"],
        default="all",
        help="Type of tests to run"
    )
    parser.add_argument(
        "--coverage", 
        action="store_true",
        help="Run with coverage report"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests"
    )
    parser.add_argument(
        "--file",
        help="Run specific test file"
    )
    parser.add_argument(
        "--function",
        help="Run specific test function"
    )
    parser.add_argument(
        "--install-deps",
        action="store_true",
        help="Install test dependencies first"
    )
    
    args = parser.parse_args()
    
    # Set up environment
    os.environ["PYTHONPATH"] = str(Path.cwd())
    
    # Install dependencies if requested
    if args.install_deps:
        deps_cmd = [
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ]
        if not run_command(deps_cmd, "Installing dependencies"):
            return 1
        
        # Install test-specific dependencies
        test_deps = [
            "pytest-mock", "pytest-timeout", "httpx[testing]", 
            "factory_boy", "freezegun"
        ]
        test_deps_cmd = [sys.executable, "-m", "pip", "install"] + test_deps
        run_command(test_deps_cmd, "Installing additional test dependencies")
    
    # Build pytest command
    pytest_cmd = [sys.executable, "-m", "pytest"]
    
    # Add test selection
    if args.file:
        pytest_cmd.append(f"tests/{args.file}")
    elif args.function:
        pytest_cmd.extend(["-k", args.function])
    elif args.type == "unit":
        pytest_cmd.extend(["-m", "unit"])
    elif args.type == "integration":
        pytest_cmd.extend(["-m", "integration"])
    elif args.type == "api":
        pytest_cmd.extend(["-m", "api"])
    else:
        pytest_cmd.append("tests/")
    
    # Add options
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.fast:
        pytest_cmd.extend(["-m", "not slow"])
    
    if args.coverage:
        pytest_cmd.extend([
            "--cov=src",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    # Run tests
    success = run_command(pytest_cmd, f"Running {args.type} tests")
    
    if success:
        print("\n🎉 All tests passed!")
        if args.coverage:
            print("📊 Coverage report generated in htmlcov/")
        return 0
    else:
        print("\n💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())