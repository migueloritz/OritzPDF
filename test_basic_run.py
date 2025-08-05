#!/usr/bin/env python
"""
Basic test to verify the application can start and respond to requests
"""
import asyncio
import httpx
import sys
from contextlib import asynccontextmanager
import uvicorn
from multiprocessing import Process
import time

async def test_basic_endpoints():
    """Test basic API endpoints"""
    print("Testing basic API endpoints...")
    
    # Wait for server to start
    await asyncio.sleep(2)
    
    async with httpx.AsyncClient(base_url="http://localhost:8000") as client:
        # Test root endpoint
        print("\n1. Testing root endpoint...")
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Welcome to OritzPDF"
        print("✓ Root endpoint working")
        
        # Test health endpoint
        print("\n2. Testing health endpoint...")
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✓ Health endpoint working")
        
        # Test detailed health endpoint
        print("\n3. Testing detailed health endpoint...")
        response = await client.get("/health/detailed")
        assert response.status_code == 200
        data = response.json()
        assert "system" in data
        print("✓ Detailed health endpoint working")
        
        # Test API documentation
        print("\n4. Testing API documentation...")
        response = await client.get("/docs")
        assert response.status_code == 200
        print("✓ API documentation available")
        
        print("\n✅ All basic tests passed!")

def run_server():
    """Run the FastAPI server"""
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, log_level="error")

async def main():
    """Main test function"""
    print("Starting OritzPDF basic functionality test...")
    print("=" * 50)
    
    # Start server in separate process
    server_process = Process(target=run_server)
    server_process.start()
    
    try:
        # Run tests
        await test_basic_endpoints()
        
        print("\n" + "=" * 50)
        print("Basic functionality test completed successfully!")
        print("\nThe OritzPDF application is ready for use.")
        print("\nNext steps:")
        print("1. Run the full test suite: python run_tests.py")
        print("2. Start the development server: python run.py")
        print("3. Visit http://localhost:8000/docs for API documentation")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    finally:
        # Stop server
        server_process.terminate()
        server_process.join()

if __name__ == "__main__":
    asyncio.run(main())