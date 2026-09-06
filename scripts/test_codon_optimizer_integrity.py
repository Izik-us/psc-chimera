"""
PSC Codon Optimizer Integrity Test Suite
=========================================

End-to-end validation of the repaired codon optimizer.

Tests:

1.  Dataset loading
2.  Variable-length batching
3.  Protein padding
4.  Codon padding
5.  ESM encoding
6.  Autoregressive decoder
7.  Synonymous masking
8.  Expression predictor
9.  All loss terms
10. Backward pass
11. Optimizer step
12. Checkpoint save + reload

This suite intentionally uses a REAL dataset and a REAL ESM checkpoint.

Run from repository root:

    python scripts/test_codon_optimizer_integrity.py ^
        --dataset data/merged/train.jsonl ^
        --esm-model-path C:\\Users\\ng692\\Downloads\\esm2_t30_150M_UR50D.pt

For Git Bash:

    python scripts/test_codon_optimizer_integrity.py \
        --dataset data/merged/train.jsonl \
        --esm-model-path /c/Users/ng692/Downloads/esm2_t30_150M_UR50D.pt
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Repository root
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from chimera.codon_optimizer import (  # noqa: E402
    AA_PAD_TOKEN,
    AA_VOCAB,
    ALL_CODONS,
    CODON_TABLE,
    CODON_TO_AA,
    PAD_TOKEN,
    CodonOptimizer,
    codon_optimizer_loss,
)

from data.codon_dataset import CodonJSONLDataset  # noqa: E402
from scripts.train_codon_optimizer import collate_records  # noqa: E402


# ---------------------------------------------------------------------------
# Test utilities
# ---------------------------------------------------------------------------

class IntegrityFailure(RuntimeError):
    pass


def check(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrityFailure(message)


def check_finite(tensor: torch.Tensor, name: str) -> None:
    check(
        torch.isfinite(tensor).all().item(),
        f"{name} contains NaN or Inf",
    )


def section(number: int, title: str) -> None:
    print(f"\n[{number}/12] {title}")


# ---------------------------------------------------------------------------
# 1. Dataset loading
# ---------------------------------------------------------------------------

def test_dataset_loading(
    dataset_path: Path,
) -> CodonJSONLDataset:

    check(
        dataset_path.is_file(),
        f"Dataset does not exist: {dataset_path}",
    )

    dataset = CodonJSONLDataset(dataset_path)

    check(
        len(dataset) > 1,
        "Dataset must contain at least two records.",
    )

    first = dataset[0]

    required_keys = {
        "protein_tokens",
        "codon_tokens",
        "aa_sequence",
        "expression",
    }

    check(
        required_keys.issubset(first.keys()),
        "Dataset record is missing required fields.",
    )

    protein_tokens = first["protein_tokens"]
    codon_tokens = first["codon_tokens"]
    aa_sequence = first["aa_sequence"][0]
    expression = first["expression"]

    check(
        protein_tokens.ndim == 1,
        "protein_tokens must be 1-D per example.",
    )

    check(
        codon_tokens.ndim == 1,
        "codon_tokens must be 1-D per example.",
    )

    check(
        protein_tokens.numel() == len(aa_sequence),
        "Protein token length != amino-acid sequence length.",
    )

    check(
        codon_tokens.numel() == len(aa_sequence),
        "Codon token length != amino-acid sequence length.",
    )

    check(
        0.0 <= float(expression) <= 1.0,
        f"Expression outside [0,1]: {float(expression)}",
    )

    print(f"       records={len(dataset)}")
    print(f"       first_sequence_length={len(aa_sequence)}")
    print("       PASS")

    return dataset


# ---------------------------------------------------------------------------
# 2. Variable-length batching
# ---------------------------------------------------------------------------

def get_variable_length_batch(
    dataset: CodonJSONLDataset,
    batch_size: int,
) -> dict:

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_records,
    )

    for batch in loader:

        protein_lengths = (
            ~batch["protein_padding_mask"]
        ).sum(dim=1)

        unique_lengths = torch.unique(protein_lengths)

        if unique_lengths.numel() >= 2:
            print(
                f"       lengths={protein_lengths.tolist()}"
            )
            print("       PASS")
            return batch

    raise IntegrityFailure(
        "Could not find a batch with variable sequence lengths. "
        "Increase --batch-size or verify the dataset contains "
        "different protein lengths."
    )


# ---------------------------------------------------------------------------
# 3 + 4. Padding validation
# ---------------------------------------------------------------------------

def validate_padding(
    tokens: torch.Tensor,
    mask: torch.Tensor,
    pad_token: int,
    name: str,
) -> None:

    check(
        tokens.ndim == 2,
        f"{name}: tokens must have shape [B,L].",
    )

    check(
        mask.ndim == 2,
        f"{name}: mask must have shape [B,L].",
    )

    check(
        mask.dtype == torch.bool,
        f"{name}: mask must be bool.",
    )

    check(
        tokens.shape == mask.shape,
        f"{name}: token/mask shape mismatch.",
    )

    lengths = (~mask).sum(dim=1)

    expected_mask = (
        torch.arange(
            tokens.shape[1],
            device=tokens.device,
        )
        .unsqueeze(0)
        >= lengths.unsqueeze(1)
    )

    check(
        torch.equal(mask, expected_mask),
        f"{name}: mask is not strict right-padding.",
    )

    padded_values = tokens[mask]

    if padded_values.numel() > 0:
        check(
            torch.all(
                padded_values == pad_token
            ).item(),
            f"{name}: padded positions contain non-PAD tokens.",
        )

    valid_values = tokens[~mask]

    if valid_values.numel() > 0:
        check(
            not torch.any(
                valid_values == pad_token
            ).item(),
            f"{name}: PAD token appears inside a biological sequence.",
        )


def test_padding(batch: dict) -> None:

    protein_tokens = batch["protein_tokens"]
    protein_mask = batch["protein_padding_mask"]

    codon_tokens = batch["codon_tokens"]
    codon_mask = batch["codon_padding_mask"]

    # Protein padding
    validate_padding(
        protein_tokens,
        protein_mask,
        AA_PAD_TOKEN,
        "protein padding",
    )

    # Codon padding
    validate_padding(
        codon_tokens,
        codon_mask,
        PAD_TOKEN,
        "codon padding",
    )

    protein_lengths = (~protein_mask).sum(dim=1)
    codon_lengths = (~codon_mask).sum(dim=1)

    check(
        torch.equal(
            protein_lengths,
            codon_lengths,
        ),
        "Protein and codon biological lengths differ.",
    )

    valid_protein = protein_tokens[~protein_mask]

    check(
        torch.all(
            (valid_protein >= 0)
            & (valid_protein < len(AA_VOCAB))
        ).item(),
        "Invalid protein token detected.",
    )

    print(
        f"       protein_pad={AA_PAD_TOKEN}"
    )
    print(
        f"       codon_pad={PAD_TOKEN}"
    )
    print("       PASS")


# ---------------------------------------------------------------------------
# 5. ESM encoding
# ---------------------------------------------------------------------------

def test_esm_encoding(
    model: CodonOptimizer,
    batch: dict,
    device: torch.device,
) -> None:

    model.eval()

    protein_tokens = batch["protein_tokens"].to(device)
    protein_mask = batch["protein_padding_mask"].to(device)

    with torch.no_grad():

        memory, memory_mask = model.encode_protein(
            protein_tokens,
            batch["aa_sequence"],
            protein_padding_mask=protein_mask,
        )

    check(
        memory.ndim == 3,
        "ESM memory must have shape [B,L,D].",
    )

    check(
        memory.shape[:2] == protein_tokens.shape,
        (
            "ESM memory spatial dimensions do not match "
            "protein token dimensions."
        ),
    )

    check(
        torch.equal(
            memory_mask,
            protein_mask,
        ),
        "ESM returned a padding mask different from the input mask.",
    )

    check_finite(
        memory,
        "ESM memory",
    )

    padded_memory = memory[memory_mask]

    if padded_memory.numel() > 0:

        check(
            torch.allclose(
                padded_memory,
                torch.zeros_like(padded_memory),
                atol=1e-6,
            ),
            "Padded ESM memory is not zero.",
        )

    print(
        f"       memory_shape={tuple(memory.shape)}"
    )
    print("       PASS")


# ---------------------------------------------------------------------------
# 6. Decoder
# ---------------------------------------------------------------------------

def test_decoder(
    model: CodonOptimizer,
    batch: dict,
    device: torch.device,
) -> dict:

    model.train()

    protein_tokens = batch["protein_tokens"].to(device)
    codon_tokens = batch["codon_tokens"].to(device)

    protein_mask = batch["protein_padding_mask"].to(device)
    codon_mask = batch["codon_padding_mask"].to(device)

    output = model(
        protein_tokens,
        codon_tokens,
        batch["aa_sequence"],
        protein_padding_mask=protein_mask,
        codon_padding_mask=codon_mask,
    )

    check(
        "logits" in output,
        "Model output does not contain logits.",
    )

    check(
        "expression" in output,
        "Model output does not contain expression prediction.",
    )

    logits = output["logits"]

    expected_shape = (
        codon_tokens.shape[0],
        codon_tokens.shape[1],
        len(ALL_CODONS),
    )

    check(
        logits.shape == expected_shape,
        (
            f"Decoder logits shape mismatch: "
            f"got {tuple(logits.shape)}, "
            f"expected {expected_shape}"
        ),
    )

    check_finite(
        logits,
        "decoder logits",
    )

    print(
        f"       logits_shape={tuple(logits.shape)}"
    )
    print("       PASS")

    return output


# ---------------------------------------------------------------------------
# 7. Synonymous masking
# ---------------------------------------------------------------------------

def test_synonymous_masking(
    model: CodonOptimizer,
    output: dict,
    batch: dict,
    device: torch.device,
) -> None:

    logits = output["logits"]
    codon_mask = batch["codon_padding_mask"].to(device)

    for b in range(logits.shape[0]):

        length = int(
            (~codon_mask[b]).sum().item()
        )

        aa_sequence = batch["aa_sequence"][b]

        check(
            length == len(aa_sequence),
            (
                f"Batch {b}: target length does not match "
                f"amino-acid sequence."
            ),
        )

        for pos in range(length):

            aa = aa_sequence[pos]

            allowed_codons = {
                codon
                for codon in CODON_TABLE[aa]
            }

            allowed_indices = {
                model.CODON_TO_IDX[codon]
                if hasattr(model, "CODON_TO_IDX")
                else None
                for codon in allowed_codons
            }

            # Current implementation exposes CODON_TO_IDX at module level,
            # so use the model output indices through the global mapping.
            from chimera.codon_optimizer import CODON_TO_IDX

            allowed_indices = {
                CODON_TO_IDX[codon]
                for codon in allowed_codons
            }

            for codon_idx in range(64):

                if codon_idx not in allowed_indices:

                    value = logits[
                        b,
                        pos,
                        codon_idx,
                    ]

                    check(
                        value.detach().item() < -1e20,
                        (
                            "Synonymous mask failed: "
                            f"batch={b}, "
                            f"position={pos}, "
                            f"codon_index={codon_idx}, "
                            f"AA={aa}, "
                            f"logit={float(value)}"
                        ),
                    )

            # Every allowed codon must remain numerically viable.
            allowed_logits = logits[
                b,
                pos,
                list(allowed_indices),
            ]

            check_finite(
                allowed_logits,
                (
                    f"allowed synonymous logits "
                    f"batch={b}, position={pos}"
                ),
            )

    print(
        "       all valid positions restricted to synonymous codons"
    )
    print("       PASS")


# ---------------------------------------------------------------------------
# 8. Expression predictor
# ---------------------------------------------------------------------------

def test_expression_predictor(
    output: dict,
    batch: dict,
) -> None:

    expression = output["expression"]

    check(
        expression.ndim >= 1,
        "Expression predictor output has invalid rank.",
    )

    check(
        expression.shape[0]
        == batch["codon_tokens"].shape[0],
        "Expression predictor batch dimension mismatch.",
    )

    check_finite(
        expression,
        "expression prediction",
    )

    print(
        f"       expression_shape={tuple(expression.shape)}"
    )
    print("       PASS")


# ---------------------------------------------------------------------------
# 9. All loss terms
# ---------------------------------------------------------------------------

def test_all_losses(
    model: CodonOptimizer,
    output: dict,
    batch: dict,
    device: torch.device,
):
    codon_tokens = batch["codon_tokens"].to(device)
    codon_mask = batch["codon_padding_mask"].to(device)
    protein_tokens = batch["protein_tokens"].to(device)
    target_expression = batch["expression"].to(device)

    loss, metrics = codon_optimizer_loss(
        output["logits"],
        codon_tokens,
        output["expression"],
        target_expression=target_expression,
        protein_tokens=protein_tokens,
        codon_padding_mask=codon_mask,
    )

    check(
        loss.ndim == 0,
        "Total loss must be scalar.",
    )

    check_finite(
        loss,
        "total loss",
    )

    required_terms = {
        "total",
        "ce",
        "cai",
        "gc",
        "upa",
        "motif",
        "expression",
    }

    missing = required_terms - set(metrics.keys())

    check(
        not missing,
        f"Missing loss metrics: {sorted(missing)}",
    )

    for name, value in metrics.items():

        check(
            math.isfinite(float(value)),
            f"Loss term '{name}' is NaN/Inf.",
        )

    print(
        "       "
        + " ".join(
            f"{name}={float(value):.6f}"
            for name, value in sorted(metrics.items())
        )
    )

    print("       PASS")

    return loss, metrics


# ---------------------------------------------------------------------------
# 10. Backward
# ---------------------------------------------------------------------------

def test_backward(
    model: CodonOptimizer,
    loss: torch.Tensor,
) -> None:

    loss.backward()

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    check(
        trainable_parameters,
        "Model has no trainable parameters.",
    )

    parameters_with_grad = [
        parameter
        for parameter in trainable_parameters
        if parameter.grad is not None
    ]

    check(
        parameters_with_grad,
        "Backward pass produced no gradients.",
    )

    for parameter in parameters_with_grad:

        check_finite(
            parameter.grad,
            "gradient",
        )

    print(
        f"       parameters_with_grad="
        f"{len(parameters_with_grad)}/"
        f"{len(trainable_parameters)}"
    )

    print("       PASS")


# ---------------------------------------------------------------------------
# 11. Optimizer step
# ---------------------------------------------------------------------------

def test_optimizer_step(
    model: CodonOptimizer,
    optimizer: torch.optim.Optimizer,
) -> None:

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0,
    )

    optimizer.step()

    changed_parameters = []

    for name, parameter in model.named_parameters():

        if not parameter.requires_grad:
            continue

        if not torch.equal(
            before[name],
            parameter.detach(),
        ):
            changed_parameters.append(name)

    check(
        changed_parameters,
        "Optimizer step did not modify any trainable parameter.",
    )

    print(
        f"       changed_parameters={len(changed_parameters)}"
    )

    print("       PASS")


# ---------------------------------------------------------------------------
# 12. Checkpoint save + reload
# ---------------------------------------------------------------------------

def test_checkpoint_round_trip(
    model: CodonOptimizer,
    optimizer: torch.optim.Optimizer,
    model_config: dict,
    device: torch.device,
) -> None:

    with tempfile.TemporaryDirectory(
        prefix="psc_codon_integrity_"
    ) as tmpdir:

        checkpoint_path = (
            Path(tmpdir)
            / "codon_optimizer_integrity.pt"
        )

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": model_config,
            "seed": 7,
        }

        # ---------------------------------------------------------------
        # Save
        # ---------------------------------------------------------------

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        check(
            checkpoint_path.is_file(),
            "Checkpoint file was not created.",
        )

        check(
            checkpoint_path.stat().st_size > 0,
            "Checkpoint file is empty.",
        )

        print(
            f"       saved={checkpoint_path.stat().st_size:,} bytes"
        )

        # ---------------------------------------------------------------
        # Reload
        # ---------------------------------------------------------------

        reloaded = CodonOptimizer.from_checkpoint(
            str(checkpoint_path),
            map_location=device,
        )

        reloaded.to(device)

        original_state = model.state_dict()
        restored_state = reloaded.state_dict()

        check(
            original_state.keys()
            == restored_state.keys(),
            "Reloaded state_dict keys differ.",
        )

        mismatches = []

        for key in original_state:

            original_tensor = (
                original_state[key]
                .detach()
                .cpu()
            )

            restored_tensor = (
                restored_state[key]
                .detach()
                .cpu()
            )

            if not torch.equal(
                original_tensor,
                restored_tensor,
            ):
                mismatches.append(key)

        check(
            not mismatches,
            (
                "Checkpoint tensor mismatch after reload: "
                f"{mismatches[:10]}"
            ),
        )

        # Also verify the optimizer state can be restored.
        reloaded_optimizer = torch.optim.AdamW(
            (
                parameter
                for parameter in reloaded.parameters()
                if parameter.requires_grad
            ),
            lr=1e-4,
        )

        reloaded_optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        print(
            "       model_state_dict=identical"
        )

        print(
            "       optimizer_state_dict=loadable"
        )

        print("       PASS")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:

    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/merged/train.jsonl"),
    )

    parser.add_argument(
        "--esm-model-path",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    args = parser.parse_args()

    if args.batch_size < 2:
        raise ValueError(
            "--batch-size must be at least 2."
        )

    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # Model configuration must match training script.
    # ------------------------------------------------------------------

    model_config = {
        "d_model": 64,
        "n_heads": 4,
        "n_dec_layers": 2,
        "dim_ff": 256,
        "esm_model_path": str(
            args.esm_model_path
        ),
    }

    print("=" * 72)
    print("PSC CODON OPTIMIZER INTEGRITY TEST")
    print("=" * 72)

    print(f"dataset : {args.dataset}")
    print(f"ESM     : {args.esm_model_path}")
    print(f"device  : {device}")
    print(f"batch   : {args.batch_size}")

    # ------------------------------------------------------------------
    # 1. Dataset
    # ------------------------------------------------------------------

    section(1, "Dataset loading")

    dataset = test_dataset_loading(
        args.dataset
    )

    # ------------------------------------------------------------------
    # 2. Variable-length batching
    # ------------------------------------------------------------------

    section(2, "Variable-length batching")

    batch = get_variable_length_batch(
        dataset,
        args.batch_size,
    )

    # ------------------------------------------------------------------
    # 3. Protein padding
    # ------------------------------------------------------------------

    section(3, "Protein padding")

    validate_padding(
        batch["protein_tokens"],
        batch["protein_padding_mask"],
        AA_PAD_TOKEN,
        "protein padding",
    )

    print(
        f"       AA_PAD_TOKEN={AA_PAD_TOKEN}"
    )

    print("       PASS")

    # ------------------------------------------------------------------
    # 4. Codon padding
    # ------------------------------------------------------------------

    section(4, "Codon padding")

    validate_padding(
        batch["codon_tokens"],
        batch["codon_padding_mask"],
        PAD_TOKEN,
        "codon padding",
    )

    print(
        f"       PAD_TOKEN={PAD_TOKEN}"
    )

    print("       PASS")

    # ------------------------------------------------------------------
    # Instantiate model
    # ------------------------------------------------------------------

    print("\n[MODEL] constructing CodonOptimizer...")

    model = CodonOptimizer(
        **model_config
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print(
        f"[MODEL] total_parameters={total_parameters:,}"
    )

    print(
        f"[MODEL] trainable_parameters={trainable_parameters:,}"
    )

    # ------------------------------------------------------------------
    # 5. ESM
    # ------------------------------------------------------------------

    section(5, "ESM encoding")

    test_esm_encoding(
        model,
        batch,
        device,
    )

    # ------------------------------------------------------------------
    # 6. Decoder
    # ------------------------------------------------------------------

    section(6, "Autoregressive decoder")

    output = test_decoder(
        model,
        batch,
        device,
    )

    # ------------------------------------------------------------------
    # 7. Synonymous masking
    # ------------------------------------------------------------------

    section(7, "Synonymous masking")

    test_synonymous_masking(
        model,
        output,
        batch,
        device,
    )

    # ------------------------------------------------------------------
    # 8. Expression predictor
    # ------------------------------------------------------------------

    section(8, "Expression predictor")

    test_expression_predictor(
        output,
        batch,
    )

    # ------------------------------------------------------------------
    # 9. Losses
    # ------------------------------------------------------------------

    section(9, "All loss terms")

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=1e-4,
    )

    optimizer.zero_grad(
        set_to_none=True
    )

    loss, metrics = test_all_losses(
        model,
        output,
        batch,
        device,
    )

    # ------------------------------------------------------------------
    # 10. Backward
    # ------------------------------------------------------------------

    section(10, "Backward pass")

    test_backward(
        model,
        loss,
    )

    # ------------------------------------------------------------------
    # 11. Optimizer
    # ------------------------------------------------------------------

    section(11, "Optimizer step")

    test_optimizer_step(
        model,
        optimizer,
    )

    # ------------------------------------------------------------------
    # 12. Checkpoint
    # ------------------------------------------------------------------

    section(12, "Checkpoint save + reload")

    test_checkpoint_round_trip(
        model,
        optimizer,
        model_config,
        device,
    )

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print("\n" + "=" * 72)
    print("INTEGRITY TEST: PASS")
    print("=" * 72)

    print(
        "\nThe repaired codon-optimization pipeline successfully passed:"
    )

    print("  ✓ dataset loading")
    print("  ✓ variable-length batching")
    print("  ✓ protein padding")
    print("  ✓ codon padding")
    print("  ✓ ESM encoding")
    print("  ✓ autoregressive decoder")
    print("  ✓ synonymous masking")
    print("  ✓ expression predictor")
    print("  ✓ all loss terms")
    print("  ✓ backward pass")
    print("  ✓ optimizer step")
    print("  ✓ checkpoint save/reload")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())