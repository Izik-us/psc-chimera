"""Regression tests for Stage 1 implementation sign-off boundaries."""

import pytest
import torch

from chimera import CHIMERAv2
from chimera.components import MSARepresentationBackbone


def _tiny_model(*, evoformer_n_blocks: int) -> CHIMERAv2:
    return CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_flow_steps=2,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
        evoformer_n_blocks=evoformer_n_blocks,
    )


def test_evoformer_checkpoint_depth_mismatch_fails_closed(tmp_path):
    """A checkpoint with fewer blocks must not silently initialize a deeper model."""
    checkpoint = tmp_path / "evoformer_2_blocks.pt"
    source = MSARepresentationBackbone(
        d_single=32,
        d_pair=16,
        n_blocks=2,
    )
    torch.save({"state_dict": source.state_dict()}, checkpoint)

    with pytest.raises(RuntimeError, match="checkpoint is incompatible"):
        CHIMERAv2.from_pretrained(
            evoformer_ckpt=str(checkpoint),
            d_evo_single=32,
            d_evo_pair=16,
            d_se3=32,
            d_pair_out=32,
            d_mpnn=16,
            n_flow_blocks=1,
            n_flow_steps=2,
            n_domains=2,
            n_modules=2,
            n_mc_dropout=2,
            evoformer_n_blocks=3,
        )


def test_matching_evoformer_checkpoint_depth_loads_but_remains_unverified(tmp_path):
    """Shape compatibility is not training or scientific validation evidence."""
    checkpoint = tmp_path / "evoformer_2_blocks.pt"
    source = MSARepresentationBackbone(
        d_single=32,
        d_pair=16,
        n_blocks=2,
    )
    torch.save({"state_dict": source.state_dict()}, checkpoint)

    model = CHIMERAv2.from_pretrained(
        evoformer_ckpt=str(checkpoint),
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_flow_steps=2,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
        evoformer_n_blocks=2,
    )

    assert model.evoformer.configuration["n_blocks"] == 2
    assert model.component_status["evoformer"]["initialization"] == "loaded_unverified"
    assert model.component_status["evoformer"]["training_status"] == "not_started"
    assert model.inference_readiness()["state"] == "UNINITIALIZED"
