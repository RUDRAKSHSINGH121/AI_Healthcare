#!/usr/bin/env python3
"""
Simple script to start the Flask application
"""

import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from app import app
    print("🚀 Starting AI Healthcare Diagnostics Server...")
    print("🌐 Server will be available at: http://localhost:5000")
    print("📱 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
    
except Exception as e:
    print(f"❌ Error starting server: {e}")
    print("🔧 Please check the error and try again")
    sys.exit(1)
