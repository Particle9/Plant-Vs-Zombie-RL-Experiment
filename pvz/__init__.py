"""Compatibility wrapper for local, editable imports.

This project has a nested package layout (`pvz/pvz`). When running scripts
from the repository root without installing the package, `import pvz` may
resolve to this outer directory. Re-export inner package symbols so existing
imports like `from pvz import Scene` continue to work.
"""

from .pvz import *  # noqa: F401,F403
