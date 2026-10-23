"""Script helpers with lazy visualization exports.

Importing ordinary training utilities must not eagerly load Matplotlib or its
GUI/native backend. Preserve the historical scripts-package exports without
making visualization a mandatory runtime dependency.
"""

from __future__ import annotations

import importlib

__all__ = ["CodonObservatory", "TrainingTelemetry", "read_system_stats"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(".codon_observatory", __name__)
    return getattr(module, name)


def __dir__():
    return sorted(set(globals()) | set(__all__))
