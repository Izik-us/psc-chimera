"""Full-size canonical CHIMERA parameter audit and staged-training smoke test.

This deliberately instantiates CanonicalCHIMERAv2 with production defaults,
then runs one real CanonicalTrainer sequence-regime step on a tiny synthetic
batch. It verifies full-size model construction, finite loss/gradients, the
trainer's component gradient contract, and an optimizer update. It is a
runtime/training-contract smoke test, not biological model training.
"""
from __future__ import annotations

import torch

from chimera.architecture import CanonicalCHIMERAv2
from chimera.training import (
    CanonicalTrainer,
    CanonicalTrainingBatch,
    TrainingRegime,
    configure_trainable_components,
)


EXPECTED_PARAMETERS = 749_197_845  # historical planning estimate, not an assertion
AA_VOCAB_SIZE = 20
BATCH_SIZE = 1
MSA_DEPTH = 2
SEQUENCE_LENGTH = 5


def main() -> None:
    torch.manual_seed(20261009)
    torch.set_num_threads(2)
    print(f"torch_version={torch.__version__}")
    print("device=cpu")
    print("configuration=CanonicalCHIMERAv2() [all canonical defaults]")

    model = CanonicalCHIMERAv2()
    model.eval()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_initial = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"total_parameters={total_parameters:,}")
    print(f"planned_reference_parameters={EXPECTED_PARAMETERS:,}")
    print(f"initial_trainable_parameters={trainable_initial:,}")
    print(f"fp32_parameter_storage_GiB={total_parameters * 4 / (1024**3):.3f}")
    print("top_level_parameter_breakdown:")
    for name, module in model.named_children():
        count = sum(parameter.numel() for parameter in module.parameters())
        if count:
            print(f"  {name}={count:,}")

    delta = total_parameters - EXPECTED_PARAMETERS
    relative_delta = 100.0 * delta / EXPECTED_PARAMETERS
    print(f"parameter_delta_vs_planned_reference={delta:+,} ({relative_delta:+.3f}%)")
    print("note=the reference count is a planning estimate; the instantiated source count is authoritative")

    # Use the actual staged trainer and its sequence loss/gradient contract.
    # SGD avoids allocating Adam moment buffers for hundreds of millions of
    # trainable parameters while still proving an optimizer update occurred.
    selected = configure_trainable_components(model, TrainingRegime.SEQUENCE)
    selected_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    trainable_count = sum(parameter.numel() for parameter in selected_parameters)
    optimizer = torch.optim.SGD(selected_parameters, lr=1e-5)
    trainer = CanonicalTrainer(
        model,
        TrainingRegime.SEQUENCE,
        optimizer=optimizer,
        dataset_manifest={
            "dataset_version": "synthetic-training-smoke-v1",
            "synthetic_only": True,
            "purpose": "training-contract verification, not biological supervision",
        },
        random_seed=20261009,
    )

    msa_tokens = torch.randint(0, AA_VOCAB_SIZE, (BATCH_SIZE, MSA_DEPTH, SEQUENCE_LENGTH))
    pair_features = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, SEQUENCE_LENGTH, model.d_evo_pair)
    target_sequence = torch.randint(0, AA_VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH))
    target_R = torch.eye(3).reshape(1, 1, 3, 3).expand(
        BATCH_SIZE, SEQUENCE_LENGTH, 3, 3
    ).clone()
    target_t = torch.zeros(BATCH_SIZE, SEQUENCE_LENGTH, 3)
    target_t[0, :, 0] = torch.arange(SEQUENCE_LENGTH, dtype=torch.float32) * 3.8
    batch = CanonicalTrainingBatch(
        msa_tokens=msa_tokens,
        pair_features=pair_features,
        target_R=target_R,
        target_t=target_t,
        target_sequence=target_sequence,
    )

    before = next(parameter for parameter in model.sequence_policy.parameters()).detach().clone()
    result = trainer.train_step(batch)
    loss = result["loss"]
    if not torch.isfinite(torch.tensor(loss)):
        raise AssertionError(f"training loss is not finite: {loss}")

    component_report = result["gradient_flow"]["components"]
    required = (
        "evoformer",
        "node_connector",
        "base_mpnn",
        "multi_scale_designer",
        "_seq_to_repr",
        "sequence_policy",
    )
    for name in required:
        row = component_report.get(name, {})
        if row.get("gradient_parameters", 0) == 0:
            raise AssertionError(f"trainer reported no gradients for required component {name}")
        if not row.get("all_gradients_finite", False):
            raise AssertionError(f"trainer reported non-finite gradients for {name}")

    after = next(parameter for parameter in model.sequence_policy.parameters()).detach()
    changed = not torch.equal(before, after)
    if not changed:
        raise AssertionError("optimizer step did not update sequence-policy parameters")

    print(f"training_regime={TrainingRegime.SEQUENCE.value}")
    print(f"selected_components={','.join(selected)}")
    print(f"sequence_length={SEQUENCE_LENGTH}")
    print(f"msa_shape={(BATCH_SIZE, MSA_DEPTH, SEQUENCE_LENGTH)}")
    print(f"regime_trainable_parameters={trainable_count:,}")
    print(f"loss={loss:.8f}")
    print(f"global_step={result['global_step']}")
    print("required_gradient_contract=PASS")
    print(f"optimizer_update_confirmed={changed}")
    print("result=PASS")
    print(
        "training_scope=full-size architecture instantiated; one synthetic CanonicalTrainer "
        "sequence-regime optimizer step confirmed; no biological training claim"
    )

    # Exercise the full canonical generation path at tiny length after the
    # training step. This verifies inference plumbing on the full-size model;
    # it is not evidence that the random model generates useful designs.
    model.eval()
    with torch.inference_mode():
        generated = model(
            msa_tokens,
            pair_features,
            target_R,
            target_t,
            n_flow_steps=2,
            n_mpnn_seqs=1,
            validate_geometry=False,
        )
    if generated["sequence_tokens"].shape != (BATCH_SIZE, 1, SEQUENCE_LENGTH):
        raise AssertionError(
            f"unexpected sequence output shape: {tuple(generated['sequence_tokens'].shape)}"
        )
    if generated["backbone_coords"].shape != (BATCH_SIZE, SEQUENCE_LENGTH, 4, 3):
        raise AssertionError(
            f"unexpected backbone output shape: {tuple(generated['backbone_coords'].shape)}"
        )
    if not torch.isfinite(generated["backbone_coords"]).all():
        raise AssertionError("full-size inference produced non-finite backbone coordinates")
    print(f"inference_sequence_shape={tuple(generated['sequence_tokens'].shape)}")
    print(f"inference_backbone_shape={tuple(generated['backbone_coords'].shape)}")
    print("full_size_inference_smoke=PASS")
    print("result=PASS")


if __name__ == "__main__":
    main()
