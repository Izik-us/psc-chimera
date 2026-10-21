import torch
import pytest

from chimera.architecture import CHIMERAv2
from chimera.se3_flow import FlowMatchingBackbone


@pytest.mark.parametrize(
    ("old_heads", "old_head_dim"),
    [(12, 5), (8, 8)],
    ids=("truncated-12-head-to-8-head", "same-head-count-old-output-layout"),
)
def test_legacy_ipa_output_projection_migrates_by_track(old_heads, old_head_dim):
    torch.manual_seed(23)
    backbone = FlowMatchingBackbone(d_single=64, d_pair=16, n_blocks=1)
    target_state = backbone.state_dict()
    prefix = "flow_model.velocity_field.ipa_blocks.0.ipa."
    historical = {
        key: value.clone()
        for key, value in target_state.items()
        if not key.startswith("sb_model.drift_model.")
    }

    historical[prefix + "q_s.weight"] = torch.randn(old_heads * old_head_dim, 64)
    historical[prefix + "k_s.weight"] = torch.randn(old_heads * old_head_dim, 64)
    historical[prefix + "v_s.weight"] = torch.randn(old_heads * old_head_dim, 64)
    historical[prefix + "q_p.weight"] = torch.randn(old_heads * 4 * 3, 64)
    historical[prefix + "k_p.weight"] = torch.randn(old_heads * 4 * 3, 64)
    historical[prefix + "v_p.weight"] = torch.randn(old_heads * 8 * 3, 64)
    historical[prefix + "pair_bias.weight"] = torch.randn(old_heads, 16)
    historical[prefix + "gamma"] = torch.randn(old_heads)
    historical[prefix + "substrate_gate.weight"] = torch.randn(old_heads, 64)
    historical[prefix + "substrate_gate.bias"] = torch.randn(old_heads)

    legacy_width = old_heads * (old_head_dim + 3 + 16)
    source_out = torch.arange(64 * legacy_width, dtype=torch.float32).reshape(64, legacy_width)
    historical[prefix + "out.weight"] = source_out.clone()

    with pytest.warns(UserWarning, match="Migrated legacy IPA output/projection layout"):
        migrated = CHIMERAv2._migrate_flow_state_dict(historical, target_state)

    # The migrated checkpoint must satisfy strict loading, including the
    # flow_model / Schrödinger-bridge alias keys.
    incompatible = backbone.load_state_dict(migrated, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

    new_heads = 8
    new_head_dim = 8
    new_v_points = 8
    pair_dim = 16
    copy_heads = min(old_heads, new_heads)
    old_scalar_width = old_heads * old_head_dim
    old_point_width = old_heads * 3
    old_pair_start = old_scalar_width + old_point_width

    new_scalar_width = new_heads * new_head_dim
    new_point_width = new_heads * new_v_points * 3
    new_norm_width = new_heads * new_v_points
    new_pair_start = new_scalar_width + new_point_width + new_norm_width

    migrated_out = migrated[prefix + "out.weight"]
    scalar_source = source_out[:, :old_scalar_width].reshape(64, old_heads, old_head_dim)
    scalar_target = migrated_out[:, :new_scalar_width].reshape(64, new_heads, new_head_dim)
    for head in range(copy_heads):
        width = min(old_head_dim, new_head_dim)
        assert torch.equal(scalar_target[:, head, :width], scalar_source[:, head, :width])

    # Old displacement features seed only the first new value-point slot.
    point_source = source_out[:, old_scalar_width:old_scalar_width + old_point_width]
    point_source = point_source.reshape(64, old_heads, 3)
    point_target = migrated_out[:, new_scalar_width:new_scalar_width + new_point_width]
    point_target = point_target.reshape(64, new_heads, new_v_points, 3)
    for head in range(copy_heads):
        assert torch.equal(point_target[:, head, 0, :], point_source[:, head, :])

    # Added point slots and point norms retain the target initialization.
    original_out = target_state[prefix + "out.weight"]
    original_points = original_out[:, new_scalar_width:new_scalar_width + new_point_width]
    original_points = original_points.reshape(64, new_heads, new_v_points, 3)
    for head in range(new_heads):
        assert torch.equal(point_target[:, head, 1:, :], original_points[:, head, 1:, :])
    migrated_norms = migrated_out[:, new_scalar_width + new_point_width:new_pair_start]
    original_norms = original_out[:, new_scalar_width + new_point_width:new_pair_start]
    assert torch.equal(migrated_norms, original_norms)

    # Pair track is relocated after point coordinates and norms, then preserved.
    old_pair = source_out[:, old_pair_start:].reshape(64, old_heads, pair_dim)
    new_pair = migrated_out[:, new_pair_start:].reshape(64, new_heads, pair_dim)
    for head in range(copy_heads):
        assert torch.equal(new_pair[:, head, :], old_pair[:, head, :])
