#!/usr/bin/env python3
"""Source-derived CHIMERA compute and activation-memory estimator.

This is an analytical estimator, not a substitute for a measured CUDA benchmark.
It reports model parameters from the instantiated canonical architecture and
estimates the dominant EvoFormer MSA/pair states, OPM and triangle-multiplication
workloads from the requested batch/MSA/length dimensions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chimera.checkpoint_io import canonical_model_from_configuration


def gibibytes(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def estimate(batch: int, msa_depth: int, length: int, d_msa: int, d_pair: int,
             c_hidden_opm: int, c_hidden_triangle: int, n_evo_blocks: int,
             dtype_bytes: int) -> dict:
    msa_values = batch * msa_depth * length * d_msa
    pair_values = batch * length * length * d_pair
    msa_bytes = msa_values * dtype_bytes
    pair_bytes = pair_values * dtype_bytes
    # MACs: OPM projection and the outgoing+incoming triangle multiplications.
    opm_macs = batch * msa_depth * length * length * c_hidden_opm * c_hidden_opm
    triangle_macs = 2 * batch * length ** 3 * c_hidden_triangle
    per_block_macs = opm_macs + triangle_macs
    return {
        "dimensions": {
            "batch": batch, "msa_depth": msa_depth, "length": length,
            "d_msa": d_msa, "d_pair": d_pair,
            "c_hidden_opm": c_hidden_opm,
            "c_hidden_triangle": c_hidden_triangle,
            "evoformer_blocks": n_evo_blocks,
        },
        "representation_tensors": {
            "msa": {"elements": msa_values, "bytes": msa_bytes, "GiB": gibibytes(msa_bytes)},
            "pair": {"elements": pair_values, "bytes": pair_bytes, "GiB": gibibytes(pair_bytes)},
            "combined_one_state_GiB": gibibytes(msa_bytes + pair_bytes),
            "uncheckpointed_block_boundary_states_upper_bound_GiB": gibibytes(
                n_evo_blocks * (msa_bytes + pair_bytes)
            ),
        },
        "dominant_operations": {
            "opm_MACs_per_example_per_block": opm_macs // batch,
            "two_triangle_multiplications_MACs_per_example_per_block": triangle_macs // batch,
            "combined_MACs_per_example_per_block": per_block_macs // batch,
            "combined_MACs_per_example_all_blocks": (per_block_macs * n_evo_blocks) // batch,
            "combined_FLOPs_per_example_all_blocks": (
                2 * per_block_macs * n_evo_blocks
            ) // batch,
            "note": "MAC/FLOP estimates cover OPM and two triangle multiplications only; attention, transitions, projections, backward pass and optimizer work are excluded.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--msa-depth", type=int, default=128)
    parser.add_argument("--length", type=int, default=600)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.msa_depth, args.length) < 1:
        parser.error("batch size, MSA depth and length must be positive")

    config = json.loads(args.architecture_json.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        parser.error("architecture JSON must contain an object")
    model = canonical_model_from_configuration(config)
    dtype_bytes = {"fp32": 4, "bf16": 2, "fp16": 2}[args.dtype]
    evo = model.evoformer
    velocity = model.flow_model.flow_model.velocity_field
    ipa = velocity.ipa_blocks[0].ipa
    point_diff_elements = (
        args.batch_size * ipa.n_head * args.length * args.length * ipa.n_qk_pts * 3
    )
    point_diff_bytes = point_diff_elements * dtype_bytes
    attention_elements = args.batch_size * ipa.n_head * args.length * args.length
    attention_bytes = attention_elements * dtype_bytes
    result = {
        "status": "analytical_estimate_not_measured",
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "parameter_storage": {
            "dtype": args.dtype,
            "GiB": gibibytes(
                sum(parameter.numel() for parameter in model.parameters()) * dtype_bytes
            ),
        },
        "evoformer": estimate(
            args.batch_size, args.msa_depth, args.length,
            evo.d_msa, evo.d_pair,
            int(evo.blocks[0].outer_product_mean.left.out_features),
            int(evo.blocks[0].triangle_multiplication_outgoing.left.out_features),
            len(evo.blocks), dtype_bytes,
        ),
        "flow_ipa": {
            "blocks": len(velocity.ipa_blocks),
            "heads": ipa.n_head,
            "query_key_points": ipa.n_qk_pts,
            "value_points": ipa.n_v_pts,
            "point_difference_tensor": {
                "shape": [
                    args.batch_size, ipa.n_head, args.length, args.length,
                    ipa.n_qk_pts, 3,
                ],
                "elements": point_diff_elements,
                "GiB": gibibytes(point_diff_bytes),
            },
            "attention_logits_tensor": {
                "shape": [args.batch_size, ipa.n_head, args.length, args.length],
                "elements": attention_elements,
                "GiB": gibibytes(attention_bytes),
            },
            "note": "Single-intermediate lower-bound estimate; multiple IPA intermediates, autograd saves, pair conditioning and backward activations add memory.",
        },
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        },
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
