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


def test_training_telemetry_is_imported_lazily_for_visualization():
    """The visualizer needs its telemetry type without breaking plain imports."""
    import ast

    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "scripts" / "train_codon_optimizer.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    train_epoch = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_epoch"
    )
    lazy_imports = [
        node for node in ast.walk(train_epoch)
        if isinstance(node, ast.ImportFrom)
        and node.module == "scripts.codon_observatory"
    ]
    imported_names = {
        alias.name for node in lazy_imports for alias in node.names
    }
    assert {"TrainingTelemetry", "read_system_stats"} <= imported_names

    # Keep the optional observatory out of the ordinary module import path.
    top_level_imports = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "scripts.codon_observatory"
        for node in top_level_imports
    )
