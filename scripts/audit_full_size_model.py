"""Full-size canonical CHIMERA parameter audit and staged-training smoke test.

This deliberately instantiates CanonicalCHIMERAv2 with production defaults.
It verifies the complete parameter count, then performs one teacher-forced
sequence-policy optimizer update while retaining the full-size model instance.
The synthetic batch is only a runtime/training-contract smoke test, not model
training on biological data or evidence of scientific readiness.
"""
from __future__ import annotations

import math
import torch

from chimera.architecture import CanonicalCHIMERAv2


EXPECTED_PARAMETERS = 749_197_845
AA_VOCAB_SIZE = 20
BATCH_SIZE = 1
SEQUENCE_LENGTH = 4


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
    print(f"target_parameters={EXPECTED_PARAMETERS:,}")
    print(f"initial_trainable_parameters={trainable_initial:,}")
    print(f"fp32_parameter_storage_GiB={total_parameters * 4 / (1024**3):.3f}")
    print("top_level_parameter_breakdown:")
    for name, module in model.named_children():
        count = sum(parameter.numel() for parameter in module.parameters())
        if count:
            print(f"  {name}={count:,}")

    if total_parameters != EXPECTED_PARAMETERS:
        raise AssertionError(
            f"canonical default parameter count drifted: expected "
            f"{EXPECTED_PARAMETERS:,}, observed {total_parameters:,}"
        )

    # The production trainer is staged. Exercise its real teacher-forced
    # sequence-policy loss on the full-size model instance with a tiny synthetic
    # batch, so the smoke test is bounded and does not pretend to train biology.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    for parameter in model.sequence_policy.parameters():
        parameter.requires_grad_(True)

    optimizer = torch.optim.AdamW(model.sequence_policy.parameters(), lr=1e-4)
    context = torch.randn(BATCH_SIZE, SEQUENCE_LENGTH, model.d_mpnn)
    targets = torch.randint(0, AA_VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH))
    before = next(model.sequence_policy.parameters()).detach().clone()

    optimizer.zero_grad(set_to_none=True)
    loss = model.sequence_policy_loss(context, targets)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise AssertionError(f"training loss is not a finite scalar: {loss}")
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.sequence_policy.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        raise AssertionError("sequence-policy training produced no gradients")
    if not all(torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("sequence-policy training produced non-finite gradients")
    grad_norm = torch.sqrt(sum(gradient.detach().float().square().sum() for gradient in gradients))
    optimizer.step()
    after = next(model.sequence_policy.parameters()).detach()
    changed = not torch.equal(before, after)
    if not changed:
        raise AssertionError("optimizer step did not update sequence-policy parameters")

    trainable_after = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"training_regime=teacher_forced_sequence_policy")
    print(f"training_batch_shape={(BATCH_SIZE, SEQUENCE_LENGTH)}")
    print(f"sequence_policy_trainable_parameters={trainable_after:,}")
    print(f"loss={float(loss.detach()):.8f}")
    print(f"gradient_l2_norm={float(grad_norm):.8f}")
    print(f"optimizer_update_confirmed={changed}")
    print("result=PASS")
    print(
        "scope=full-size architecture instantiated; one synthetic teacher-forced "
        "sequence-policy optimizer step confirmed; no biological training claim"
    )


if __name__ == "__main__":
    main()
