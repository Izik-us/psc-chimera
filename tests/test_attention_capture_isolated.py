"""Subprocess regression for the optional decoder attention-capture path."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_CAPTURE_SMOKE = r"""
import torch
from chimera.codon_optimizer import ALL_CODONS, CODON_TO_IDX, CodonOptimizer

torch.set_num_threads(1)
length = 24
model = CodonOptimizer(
    d_model=32,
    n_heads=4,
    n_dec_layers=2,
    dim_ff=64,
    max_protein_length=length,
    max_codons=length,
)
model.train()
memory = torch.randn(1, length, 32, requires_grad=True)
target = torch.full((1, length), CODON_TO_IDX["ATG"], dtype=torch.long)
mask = torch.zeros((1, length), dtype=torch.bool)
sequence = ["M" * length]

model.set_attention_capture(True, final_layer_only=True)
assert not model.decoder.layers[0].capture_cross_attention
assert model.decoder.layers[-1].capture_cross_attention
logits = model.decode_teacher_forced(
    memory=memory,
    target_codons=target,
    aa_sequence=sequence,
    target_padding_mask=mask,
    memory_padding_mask=mask,
)
attention = model.get_cross_attention()
assert logits.shape == (1, length, len(ALL_CODONS))
assert attention is not None
assert attention.shape == (1, model.n_heads, length, length)
assert torch.isfinite(attention).all()
(logits.square().mean()).backward()
assert memory.grad is not None and torch.isfinite(memory.grad).all()
model.set_attention_capture(False)
assert model.get_cross_attention() is None
print("ATTENTION_CAPTURE_FORWARD_BACKWARD_OK", flush=True)
"""


def test_attention_capture_forward_backward_in_isolated_process():
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", _CAPTURE_SMOKE],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "Attention-capture subprocess exceeded 120 seconds."
        ) from exc

    assert result.returncode == 0, (
        "Attention-capture native path failed in an isolated process.\n"
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "ATTENTION_CAPTURE_FORWARD_BACKWARD_OK" in result.stdout
