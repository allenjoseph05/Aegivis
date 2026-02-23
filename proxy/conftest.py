"""pytest configuration for proxy package tests."""
import sys
import os

# Add the repo root to sys.path so `from proxy.app.xxx import ...` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
