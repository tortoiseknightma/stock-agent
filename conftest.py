"""
Pytest configuration.
Ensures the project root is on sys.path so tests can import project modules
without being affected by the root __init__.py (which uses relative imports
and is intended for use as an installed package, not standalone pytest runs).
"""
import sys
from pathlib import Path

# Make project root importable
root = Path(__file__).parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
