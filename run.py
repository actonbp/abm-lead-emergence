"""
Script to run the Shiny application with proper path handling.
"""

import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = str(Path(__file__).parent / "src")
sys.path.insert(0, src_path)

# Import after path setup
from app.app import app

if __name__ == "__main__":
    print("Starting Leadership Emergence Simulation...")
    print(f"Python path includes: {sys.path}")
    app.run() 