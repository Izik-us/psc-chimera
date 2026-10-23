"""Regression tests for the terminal-native CodonOptimizer observatory."""

from __future__ import annotations

from io import StringIO
import subprocess
import sys
from pathlib import Path

import numpy as np

from scripts.codon_observatory import CodonObservatory, TrainingTelemetry


def _telemetry(step: int = 3) -> TrainingTelemetry:
    return TrainingTelemetry(
        epoch=1,
        total_epochs=2,
        global_step=step,
        total_steps=20,
        loss=1.25,
        ce=1.10,
        cai=0.87,
        gc=0.56,
        motif=0.08,
        upa=0.02,
        expression_loss=0.14,
        gradient_norm=0.72,
        learning_rate=1e-4,
        cpu_percent=42.0,
        memory_gb=8.5,
        batch_time=1.2,
        samples_per_second=0.83,
        protein_length=8,
        codon_length=8,
        predicted_expression=0.48,
        target_expression=0.50,
        architecture="d_model=256, heads=8, decoder_layers=10, dim_ff=1024",
        sample_protein="MTEYKLVV",
        selected_position=2,
        selected_amino_acid="E",
        candidate_codons=("GAA", "GAG"),
        candidate_probabilities=(0.35, 0.65),
        predicted_codons=("ATG", "ACC", "GAG", "TAT", "AAG", "CTG", "GTG", "GTT"),
        total_parameters=12_345,
        trainable_parameters=6_789,
        model_config="d_model=256, heads=8, decoder_layers=10, dim_ff=1024",
        cross_attention=np.asarray(
            [
                [0.10, 0.20, 0.70, 0.00],
                [0.05, 0.10, 0.80, 0.05],
                [0.05, 0.15, 0.75, 0.05],
                [0.20, 0.20, 0.50, 0.10],
            ],
            dtype=float,
        ),
    )


def test_attention_capture_skips_long_sequences():
    observatory = CodonObservatory(update_every=3, max_attention_length=512)
    try:
        assert observatory.should_capture_attention(0, 512)
        assert observatory.should_capture_attention(3, 128)
        assert not observatory.should_capture_attention(1, 128)
        assert not observatory.should_capture_attention(3, 513)
    finally:
        observatory.close()


def test_cli_observatory_renders_telemetry_and_attention():
    stream = StringIO()
    observatory = CodonObservatory(update_every=1, stream=stream)
    observatory.update(_telemetry())
    output = stream.getvalue()
    observatory.close()

    assert "\x1b[2J\x1b[H" in output  # dashboard redraws in place, not as appended snapshots
    assert "PSC CODON OPTIMIZER • CLI TRAINING OBSERVATORY" in output
    assert "OBJECTIVES" in output
    assert "CODON DECODING" in output
    assert "DECODER → PROTEIN ATTENTION" in output
    assert "matrix shape" in output
    assert "selected codon top protein positions" in output
    assert "GAG=0.650" in output
    assert "predicted DNA codons" in output
    assert "MODEL CONFIG" in output
    assert "d_model=256" in output
    assert "MODEL PARAMETERS" in output
    assert "total                : 12,345" in output
    assert "trainable            : 6,789" in output
    assert "UPA" in output
    assert "DNA length 24 nt" in output


def test_observatory_does_not_import_matplotlib():
    """The terminal observatory must not require plotting/native GUI code."""
    repo_root = Path(__file__).resolve().parents[1]
    probe = r"""
import sys
from io import StringIO
import numpy as np
from scripts.codon_observatory import CodonObservatory, TrainingTelemetry
assert "matplotlib" not in sys.modules
stream = StringIO()
observatory = CodonObservatory(update_every=1, stream=stream)
telemetry = TrainingTelemetry(
    epoch=1, total_epochs=1, global_step=1, total_steps=1,
    loss=1.0, ce=0.9, cai=0.8, gc=0.5, motif=0.1, upa=0.0,
    expression_loss=0.2, gradient_norm=0.5, learning_rate=1e-4,
    cpu_percent=None, memory_gb=None, batch_time=1.0,
    samples_per_second=1.0, protein_length=4, codon_length=4,
    predicted_expression=0.4, target_expression=0.5,
    architecture="test", sample_protein="MTEY", selected_position=1,
    selected_amino_acid="T", candidate_codons=("ACT", "ACC"),
    candidate_probabilities=(0.4, 0.6), predicted_codons=("ATG", "ACC", "GAG", "TAT"),
    cross_attention=np.eye(4, dtype=float),
)
observatory.update(telemetry)
observatory.close()
assert "PSC CODON OPTIMIZER • CLI TRAINING OBSERVATORY" in stream.getvalue()
assert "matplotlib" not in sys.modules
print("CLI_OBSERVATORY_ISOLATED_FROM_MATPLOTLIB", flush=True)
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
        "CLI observatory imported plotting/native GUI code or crashed.\n"
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "CLI_OBSERVATORY_ISOLATED_FROM_MATPLOTLIB" in result.stdout
