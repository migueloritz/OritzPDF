#!/usr/bin/env python
"""
Setup script for OritzPDF
This script helps set up the development environment
"""
import subprocess
import sys
import os
from pathlib import Path

def create_venv():
    """Create virtual environment"""
    venv_path = Path("venv")
    if not venv_path.exists():
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
        print("✓ Virtual environment created")
    else:
        print("✓ Virtual environment already exists")
    return venv_path

def get_pip_command():
    """Get the pip command for the current platform"""
    if sys.platform == "win32":
        return str(Path("venv/Scripts/pip.exe"))
    else:
        return str(Path("venv/bin/pip"))

def get_python_command():
    """Get the python command for the current platform"""
    if sys.platform == "win32":
        return str(Path("venv/Scripts/python.exe"))
    else:
        return str(Path("venv/bin/python"))

def install_dependencies():
    """Install project dependencies"""
    pip_cmd = get_pip_command()
    
    print("\nUpgrading pip...")
    subprocess.run([pip_cmd, "install", "--upgrade", "pip"], check=True)
    
    print("\nInstalling project dependencies...")
    subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
    print("✓ Dependencies installed")

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("\nCreating .env file from template...")
        env_example = Path(".env.example")
        if env_example.exists():
            env_file.write_text(env_example.read_text())
            print("✓ .env file created (please update with your settings)")
        else:
            print("⚠ .env.example not found")
    else:
        print("✓ .env file already exists")

def create_upload_directory():
    """Create upload directory for local storage"""
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    print("✓ Upload directory created")

def download_spacy_model():
    """Download spaCy language model"""
    python_cmd = get_python_command()
    print("\nDownloading spaCy English model...")
    try:
        subprocess.run([python_cmd, "-m", "spacy", "download", "en_core_web_sm"], check=True)
        print("✓ spaCy model downloaded")
    except:
        print("⚠ Failed to download spaCy model (optional)")

def main():
    """Main setup function"""
    print("OritzPDF Setup Script")
    print("=" * 50)
    
    # Change to project directory
    os.chdir(Path(__file__).parent)
    
    # Create virtual environment
    create_venv()
    
    # Install dependencies
    install_dependencies()
    
    # Create .env file
    create_env_file()
    
    # Create directories
    create_upload_directory()
    
    # Download spaCy model
    download_spacy_model()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\nNext steps:")
    print("1. Activate virtual environment:")
    if sys.platform == "win32":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Update .env file with your configuration")
    print("3. Run tests: python run_tests.py")
    print("4. Start server: python run.py")
    print("\nVisit http://localhost:8000/docs for API documentation")

if __name__ == "__main__":
    main()