"""Make the flat project root importable for tests under any pytest invocation.

With a flat package (modules at repo root, tests in tests/), `python -m pytest`
already puts the root on sys.path, but bare `pytest tests/` would only add the
tests/ dir. This root conftest guarantees `from book import Book` etc. resolve
either way.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
