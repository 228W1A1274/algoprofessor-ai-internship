"""
conftest.py
────────────
Adds the project root to sys.path so pytest can resolve all local imports
(agents/, tasks/, tools/, llm_client.py, crew.py) without requiring an
installed package.

This file is picked up automatically by pytest before any test is collected.
No changes needed here — just keep it in the project root.
"""
import sys
from pathlib import Path

# Insert project root at the front of sys.path
sys.path.insert(0, str(Path(__file__).parent))
