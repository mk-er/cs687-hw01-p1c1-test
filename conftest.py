"""Stop the test run once, with a readable message, on an old interpreter.

Without this, an interpreter older than 3.10 produces one collection error per
test file, each reporting a TypeError about the | operator, and the reader has
to know that this means "your Python is too old". pytest
imports this file before it collects anything, so the check lands first.

The import below is guarded because this file must work whether or not the
optional helper module is present.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from cs687._pyversion import check as _check_python_version
except ImportError:                      # the helper is a convenience, not a dependency
    def _check_python_version():
        import cs687                     # the package checks itself on import


try:
    _check_python_version()
except RuntimeError as exc:
    pytest.exit(str(exc), returncode=1)
