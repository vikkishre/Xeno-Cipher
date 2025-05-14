# serve.py
import os
import sys

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(__file__))

from waitress import serve
from app import app

if __name__ == "__main__":
    print("Starting XenoCipher server with Waitress...")
    print("Open your browser and navigate to: http://localhost:8000")
    print("----------------------------------------")
    serve(app, host='0.0.0.0', port=8000) 