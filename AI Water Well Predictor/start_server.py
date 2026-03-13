#!/usr/bin/env python
"""Startup script with better error handling"""
import sys
import traceback

try:
    print("=" * 60)
    print("Starting AI Water Well Predictor Server...")
    print("=" * 60)
    
    # Import and run the app
    import app
    
    print("\n✓ Server started successfully!")
    print("✓ Access the application at: http://127.0.0.1:8000")
    print("✓ Login page: http://127.0.0.1:8000/login.html")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)
    
except KeyboardInterrupt:
    print("\n\nServer stopped by user")
    sys.exit(0)
except Exception as e:
    print("\n" + "=" * 60)
    print("ERROR: Failed to start server")
    print("=" * 60)
    print(f"Error Type: {type(e).__name__}")
    print(f"Error Message: {str(e)}")
    print("\nFull Traceback:")
    print("-" * 60)
    traceback.print_exc()
    print("=" * 60)
    sys.exit(1)

