"""Test-only public API wiring.

The supported CHIMERAv2 entry point is ``chimera.CHIMERAv2``.  A historical
shape/integration test still imports the implementation module directly, so
redirect that test import to the supported canonical composition without
putting compatibility machinery into the production package.
"""

from chimera import CHIMERAv2 as _CanonicalCHIMERAv2
import chimera.chimera_v2 as _implementation

_implementation.CHIMERAv2 = _CanonicalCHIMERAv2
