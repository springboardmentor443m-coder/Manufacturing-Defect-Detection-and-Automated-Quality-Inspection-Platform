#!/usr/bin/env python3
"""
Camera simulation runner entrypoint for ml/scripts/
"""
import sys
from pathlib import Path

# Add backend/scripts to sys.path
backend_scripts = Path(__file__).resolve().parent.parent.parent / "backend" / "scripts"
sys.path.insert(0, str(backend_scripts))

from camera_sim import main

if __name__ == "__main__":
    main()
