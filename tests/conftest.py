"""
Test configuration and shared fixtures.
"""

import os
import sys
from pathlib import Path

# Add src directory to Python path
src_dir = Path(__file__).parent.parent / "src"
sys.path.append(str(src_dir)) 