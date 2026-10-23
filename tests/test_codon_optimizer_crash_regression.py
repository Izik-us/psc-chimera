"""Isolated native-crash regression for long codon training sequences.

The forward/backward workload runs in a child process so a native access
violation or segmentation fault is reported as a normal pytest failure rather
than killing the complete test runner.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


_LONG_SEQUENCE_SMOKE = r"""
import torch
from chimera.codon_optimizer import (
    ALL_CODONS,
    CODON_TO_IDX,
    CodonOptimizer,
)

torch.set_num_threads(1)
length = 2426
atg = CODON_TO_IDX["ATG"]
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
target_codons = torch.full((1, length), atg, dtype=torch.long)
target_mask = torch.zeros((1, length), dtype=torch.bool)
memory_mask = torch.zeros((1, length), dtype=torch.bool)
aa_sequence = ["M" * length]

decoder_lengths = []
for layer in model.decoder.layers:
    layer.register_forward_pre_hook(
        lambda module, args: decoder_lengths.append(int(args[0].shape[1]))
    )

logits = model.decode_teacher_forced(
    memory=memory,
    target_codons=target_codons,
    aa_sequence=aa_sequence,
    target_padding_mask=target_mask,
    memory_padding_mask=memory_mask,
)
assert logits.shape == (1, length, len(ALL_CODONS))
assert decoder_lengths and max(decoder_lengths) <= 512

global_lengths = []
model.expression_predictor.global_transformer.register_forward_pre_hook(
    lambda module, args: global_lengths.append(int(args[0].shape[1]))
)
probabilities = torch.softmax(logits, dim=-1)
expression = model.expression_predictor(
    probabilities,
    padding_mask=target_mask,
)
assert expression.shape[0] == 1
assert global_lengths and max(global_lengths) <= 512

loss = logits[:, :, atg].mean() + expression.mean()
assert torch.isfinite(loss)
loss.backward()
assert memory.grad is not None and torch.isfinite(memory.grad).all()
assert any(
    parameter.grad is not None and torch.isfinite(parameter.grad).all()
    for parameter in model.decoder.parameters()
)
print("LONG_SEQUENCE_NATIVE_CRASH_REGRESSION_OK", flush=True)
"""


def test_long_sequence_forward_backward_survives_native_crash_boundary():
    repo_root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            [sys.executable, "-X", "faulthandler", "-c", _LONG_SEQUENCE_SMOKE],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            "Long-sequence native-crash smoke test exceeded 300 seconds. "
            f"stdout={exc.stdout!r}; stderr={exc.stderr!r}"
        ) from exc

    assert result.returncode == 0, (
        "Long-sequence child process failed. This captures Windows access "
        "violations (for example 0xC0000005), POSIX signals, Python errors, "
        "and native segmentation faults without crashing pytest itself.\n"
        f"returncode={result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "LONG_SEQUENCE_NATIVE_CRASH_REGRESSION_OK" in result.stdout
