"""Regression tests for isolating optional visualization dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_plain_training_import_does_not_load_matplotlib():
    """A non-visual training run should not initialize GUI/native plotting code."""
    repo_root = Path(__file__).resolve().parents[1]
    probe = r"""
import sys
import scripts.train_codon_optimizer
assert "matplotlib" not in sys.modules, sorted(
    name for name in sys.modules if name.startswith("matplotlib")
)
assert "matplotlib.pyplot" not in sys.modules
print("TRAINING_IMPORT_ISOLATED_FROM_MATPLOTLIB", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-X", "faulthandler", "-c", probe],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        "Importing the ordinary training entry point loaded plotting libraries "
        "or failed in its isolated process.\n"
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "TRAINING_IMPORT_ISOLATED_FROM_MATPLOTLIB" in result.stdout
