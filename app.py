"""
LifeOps — Entry point for Hugging Face Spaces.

This file is used when deploying to HF Spaces. It launches the week-view Gradio app.
"""

import sys
from pathlib import Path

# Ensure repo root is on path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from app.week_view import create_week_demo

demo = create_week_demo()

if __name__ == "__main__":
    demo.launch()
