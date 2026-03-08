#!/usr/bin/env python3
"""
Test that HF_TOKEN is set and working correctly.
Run: python scripts/test_hf_token.py
"""

import os
import sys
from pathlib import Path

# Load .env
repo_root = Path(__file__).resolve().parent.parent
env_path = repo_root / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)
    print(f"Loaded .env from {env_path}")
else:
    print(f"No .env at {env_path}")

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
if not token:
    print("FAIL: HF_TOKEN (or HUGGING_FACE_HUB_TOKEN) is not set.")
    sys.exit(1)

print(f"Token found: {token[:10]}...{token[-4:] if len(token) > 14 else '***'}")

# Verify token with HuggingFace API
try:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    whoami = api.whoami()
    print(f"OK: Authenticated as {whoami.get('name', '?')} (type: {whoami.get('type', '?')})")
except Exception as e:
    print(f"FAIL: Token validation failed: {e}")
    sys.exit(1)

print("\nHF token is working correctly.")
