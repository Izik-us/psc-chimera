from pathlib import Path

import pytest
import torch

from chimera import CHIMERAv2
from chimera.chimera_v2 import CHIMERAv2 as LegacyCHIMERAv2
from chimera.bayesian import BayesianUncertaintyEstimator


def test_public_chimera_is_independent_canonical_composition():
    import chimera
    import chimera.architecture as architecture
    import chimera.components as components
    import chimera.domain_schema as domain_schema
    import chimera.chimera_v2 as legacy_module
    from chimera.flow_matching import FlowMatchingBackbone, InvariantPointAttention
    from chimera.multi_objective import MultiScaleNRPSDesigner

    assert CHIMERAv2 is architecture.CanonicalCHIMERAv2
    assert CHIMERAv2 is not LegacyCHIMERAv2
    assert LegacyCHIMERAv2 not in architecture.CanonicalCHIMERAv2.__bases__
    assert not (Path(__file__).resolve().parents[1] / "chimera" / "canonical_components.py").exists()
    assert legacy_module.NRPSConstraints is domain_schema.NRPSConstraints
    assert legacy_module.EvoFormerBackbone is components.EvoFormerBackbone
    assert legacy_module.TriangularPairUpdateConnector is components.TriangularPairUpdateConnector
    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=16,
        d_mpnn=16,
        n_flow_blocks=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )
    assert getattr(model, "_is_canonical_composition", False) is True
    assert isinstance(model.flow_model, FlowMatchingBackbone)
    assert isinstance(model.multi_scale_designer, MultiScaleNRPSDesigner)
    assert all(
        isinstance(block.ipa, InvariantPointAttention)
        for block in model.flow_model.flow_model.velocity_field.ipa_blocks
    )
    assert not any(isinstance(module, LegacyCHIMERAv2) for module in model.modules())


def test_canonical_and_legacy_objective_imports_are_isolated():
    from chimera.bayesian import BayesianUncertaintyEstimator as CanonicalBayesian
    from chimera.dpo import DPOTrainer as CanonicalDPO
    from chimera.multi_objective import (
        BayesianUncertaintyEstimator as LegacyBayesian,
        DPOTrainer as LegacyDPO,
        LegacyBayesianUncertaintyEstimator,
        LegacyDPOTrainer,
    )

    assert CanonicalBayesian.__module__ == "chimera.bayesian"
    assert CanonicalDPO.__module__ == "chimera.dpo"
    assert LegacyBayesian is LegacyBayesianUncertaintyEstimator
    assert LegacyDPO is LegacyDPOTrainer
    assert CanonicalBayesian is not LegacyBayesian
    assert CanonicalDPO is not LegacyDPO


def test_canonical_synthetic_forward_and_backward():
    from chimera.lie import so3_exp

    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_flow_steps=2,
        n_mpnn_seqs=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )
    model.prepare_for_training()
    batch, msa_depth, length = 1, 3, 8
    msa = torch.randint(0, 23, (batch, msa_depth, length))
    pair = torch.zeros(batch, length, length, 16)
    source_R = so3_exp(torch.randn(batch, length, 3) * 0.1)
    source_t = torch.randn(batch, length, 3)

    output = model(msa, pair, source_R, source_t, n_flow_steps=2, n_mpnn_seqs=1)
    assert output["sequence_tokens"].shape == (batch, 1, length)
    assert output["sequences"].shape == (batch, 1, length, 20)
    assert output["pareto_objectives"].expression_efficiency.shape == (batch, 1)
    for value in output.values():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all()

    targets = output["sequence_tokens"][:, 0]
    loss = model.sequence_policy_loss(output["sequence_context"][:, 0], targets)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.sequence_policy.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    before = [parameter.detach().clone() for parameter in model.sequence_policy.parameters()]
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    assert any(
        parameter.grad is not None
        for parameter in model.sequence_policy.parameters()
        if parameter.requires_grad
    )
    optimizer.step()
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.sequence_policy.parameters())
    )


def test_canonical_proteus_update_changes_trainable_policy():
    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )
    model.prepare_for_training()
    msa = torch.randint(0, 23, (1, 2, 4))
    pair = torch.zeros(1, 4, 4, 16)
    before = [parameter.detach().clone() for parameter in model.sequence_policy.parameters()]
    metrics = model.update_from_proteus(
        ["ACDE"], ["AAAA"], msa, pair, n_dpo_steps=1, learning_rate=1e-3
    )
    assert "reward_margin" in metrics
    assert any(
        not torch.equal(old, new.detach())
        for old, new in zip(before, model.sequence_policy.parameters())
    )


def test_expected_improvement_requires_historical_best():
    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )
    with pytest.raises(ValueError, match="historical best"):
        model.compute_expected_improvement(torch.tensor([0.8]), torch.tensor([0.1]))


def test_legacy_best_observed_is_explicit_history():
    legacy = LegacyCHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )
    assert legacy.best_observed is None
    legacy.set_best_observed(0.72)
    assert legacy.best_observed == pytest.approx(0.72)
    with pytest.raises(ValueError, match="historical utility"):
        legacy.set_best_observed(1.2)


def test_canonical_forward_rejects_empty_domain_spans():
    from chimera.chimera_v2 import NRPSConstraints
    from chimera.lie import so3_exp

    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )
    batch, length = 1, 8
    constraints = NRPSConstraints(
        fixed_mask=torch.zeros(batch, length, dtype=torch.bool),
        stachelhaus_positions=torch.tensor([1]),
        domain_boundaries=torch.tensor([[[0, 0], [4, 8]]]),
        module_boundaries=torch.tensor([[[0, 4], [4, 8]]]),
        icosahedral_face=torch.tensor([0]),
        ppt_serine_position=2,
        hotspot_coords=None,
        hotspot_indices=None,
        target_substrate="PHE",
    )
    with pytest.raises(ValueError, match="domain spans must be non-empty"):
        model(
            torch.randint(0, 23, (batch, 2, length)),
            torch.zeros(batch, length, length, 16),
            so3_exp(torch.zeros(batch, length, 3)),
            torch.zeros(batch, length, 3),
            constraints=constraints,
        )


def test_canonical_fails_closed_for_unaligned_structural_retrieval():
    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )
    model.structural_retriever.build_index(
        torch.zeros(1, 16).numpy(),
        [{"pocket_coords": torch.zeros(10, 3).tolist()}],
    )
    with pytest.raises(RuntimeError, match="not aligned"):
        model(
            torch.randint(0, 23, (1, 2, 8)),
            torch.zeros(1, 8, 8, 16),
            torch.eye(3).expand(1, 8, 3, 3),
            torch.zeros(1, 8, 3),
            substrate_id=torch.tensor([0]),
            use_rag=True,
        )


def test_canonical_replacement_preserves_freeze_contract():
    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=16,
        n_flow_blocks=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )
    assert all(not p.requires_grad for p in model.flow_model.parameters())
    assert all(not p.requires_grad for p in model.multi_scale_designer.parameters())
    assert all(not p.requires_grad for p in model.pareto_head.parameters())
    assert any(p.requires_grad for p in model._seq_to_repr.parameters())


def test_canonical_component_state_dict_shapes_match_legacy_layout():
    config = dict(
        d_evo_single=64,
        d_evo_pair=32,
        d_se3=64,
        d_pair_out=64,
        d_mpnn=32,
        n_flow_blocks=1,
        n_domains=5,
        n_modules=2,
        n_mc_dropout=2,
    )
    legacy = LegacyCHIMERAv2(**config)
    canonical = CHIMERAv2(**config)

    for name in ("flow_model", "multi_scale_designer", "pareto_head", "substrate_conditioner"):
        old_state = getattr(legacy, name).state_dict()
        new_state = getattr(canonical, name).state_dict()
        if name == "flow_model":
            migrated = CHIMERAv2._migrate_flow_state_dict(old_state, new_state)
            incompatible = canonical.flow_model.load_state_dict(migrated, strict=True)
            assert not incompatible.missing_keys
            assert not incompatible.unexpected_keys
            continue
        assert set(old_state).issubset(new_state), name
        for key in old_state:
            if name == "multi_scale_designer" and key == "edge_proj.weight":
                assert old_state[key].shape == (new_state[key].shape[0], 16)
                assert new_state[key].shape == (old_state[key].shape[0], 28)
                continue
            assert old_state[key].shape == new_state[key].shape, (name, key)
    flow_state = canonical.flow_model.state_dict()
    for key, value in flow_state.items():
        if key.startswith("flow_model.velocity_field."):
            alias = key.replace("flow_model.velocity_field.", "sb_model.drift_model.", 1)
            if alias in flow_state:
                assert torch.equal(value, flow_state[alias])


def test_flow_checkpoint_migration_fills_bridge_alias(tmp_path):
    from chimera.flow_matching import FlowMatchingBackbone

    model = FlowMatchingBackbone(d_single=32, d_pair=16, n_blocks=1)
    target = model.state_dict()
    historical = {
        key: value.clone()
        for key, value in target.items()
        if not key.startswith("sb_model.drift_model.")
    }
    migrated = CHIMERAv2._migrate_flow_state_dict(historical, target)
    incompatible = model.load_state_dict(migrated, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys

    canonical = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=16,
        d_mpnn=16,
        n_flow_blocks=1,
        n_domains=2,
        n_modules=2,
        n_mc_dropout=2,
    )
    connector_path = tmp_path / "legacy_flow_connector.pt"
    torch.save({"flow_model": historical}, connector_path)
    canonical.load_connectors(str(connector_path))


def test_flow_migration_maps_truncated_12_head_ipa(tmp_path):
    from chimera.flow_matching import FlowMatchingBackbone

    backbone = FlowMatchingBackbone(d_single=64, d_pair=16, n_blocks=1)
    target = backbone.state_dict()
    historical = {
        key: value.clone()
        for key, value in target.items()
        if not key.startswith("sb_model.drift_model.")
    }
    prefix = "flow_model.velocity_field.ipa_blocks.0.ipa."
    historical[prefix + "q_s.weight"] = torch.randn(60, 64)
    historical[prefix + "k_s.weight"] = torch.randn(60, 64)
    historical[prefix + "v_s.weight"] = torch.randn(60, 64)
    historical[prefix + "q_p.weight"] = torch.randn(144, 64)
    historical[prefix + "k_p.weight"] = torch.randn(144, 64)
    historical[prefix + "v_p.weight"] = torch.randn(288, 64)
    historical[prefix + "pair_bias.weight"] = torch.randn(12, 16)
    historical[prefix + "gamma"] = torch.randn(12)
    historical[prefix + "substrate_gate.weight"] = torch.randn(12, 64)
    historical[prefix + "substrate_gate.bias"] = torch.randn(12)
    historical[prefix + "out.weight"] = torch.randn(64, 12 * (5 + 3 + 16))
    historical["frozen_bridge.0.weight"] = torch.randn(512, 64)
    historical["frozen_bridge.0.bias"] = torch.randn(512)

    with pytest.warns(UserWarning) as recorded:
        migrated = CHIMERAv2._migrate_flow_state_dict(historical, target)
    assert len(recorded) >= 2
    assert not any(key.startswith("frozen_bridge.") for key in migrated)
    incompatible = backbone.load_state_dict(migrated, strict=True)
    assert not incompatible.missing_keys
    assert not incompatible.unexpected_keys


def test_legacy_designer_checkpoint_migrates_edge_projection(tmp_path):
    config = dict(
        d_evo_single=64,
        d_evo_pair=32,
        d_se3=64,
        d_pair_out=64,
        d_mpnn=32,
        n_flow_blocks=1,
        n_domains=5,
        n_modules=2,
        n_mc_dropout=2,
    )
    legacy = LegacyCHIMERAv2(**config)
    canonical = CHIMERAv2(**config)
    old_projection = legacy.multi_scale_designer.edge_proj.weight.detach().clone()
    path = tmp_path / "legacy_connectors.pt"
    torch.save({"multi_scale_designer": legacy.multi_scale_designer.state_dict()}, path)

    with pytest.warns(UserWarning, match="Migrated legacy 16-D edge projection"):
        canonical.load_connectors(str(path))

    assert torch.equal(canonical.multi_scale_designer.edge_proj.weight[:, :16], old_projection)


def test_bayesian_chimera_normalization_uses_fixed_scales():
    output = {
        "evol_plausibility": torch.tensor([[0.0, 2.0]]),
        "structural_stability": torch.tensor([[50.0, 100.0]]),
        "expression_efficiency": torch.tensor([[0.2, 0.8]]),
        "substrate_selectivity": torch.tensor([[0.1, 0.9]]),
        "assembly_compat": torch.tensor([[0.3, 0.7]]),
    }
    normalized = BayesianUncertaintyEstimator._normalize_chimera_objectives(output)
    assert normalized.shape == (1, 2, 5)
    assert torch.all((normalized >= 0) & (normalized <= 1))
    assert torch.allclose(
        normalized[0, :, 0],
        torch.sigmoid(output["evol_plausibility"]).squeeze(0),
    )


def test_weight_downloaders_are_atomic():
    root = Path(__file__).resolve().parents[1]
    sh = (root / "scripts" / "download_weights.sh").read_text(encoding="utf-8")
    ps1 = (root / "scripts" / "download_weights.ps1").read_text(encoding="utf-8")
    assert "${out}.part" in sh
    assert "$Output.part" in ps1
    assert 'mv -f "$part" "$out"' in sh
    assert "Move-Item -Force $part $Output" in ps1
