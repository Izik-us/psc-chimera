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
    assert output["objective_proxy_scores"].shape == (batch, 1, 5)
    assert output["objective_proxy_outputs"]["assembly_compatibility"].source == "icosahedral_interface_geometry_proxy"
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


def test_experimental_design_ranks_explicit_proxy_values_not_random_objective_heads():
    model = CHIMERAv2(
        d_evo_single=32, d_evo_pair=16, d_se3=32, d_pair_out=32,
        d_mpnn=16, n_flow_blocks=1, n_flow_steps=2, n_mpnn_seqs=1,
        n_domains=2, n_modules=2, n_mc_dropout=2,
    )
    length = 8
    rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(1, length, -1, -1).clone()
    translations = torch.arange(length).reshape(1, length, 1).float() * torch.tensor([3.8, 0.0, 0.0])

    result = model.design(
        nrps_msa=torch.randint(0, 20, (1, 2, length)),
        source_backbone=(rotations, translations),
        initial_pair_features=torch.zeros(1, length, length, 16),
        n_designs=2,
        n_pareto_samples=1,
        flow_steps=2,
        experimental=True,
    )

    assert result["ranking_source"] == "deterministic_proxy_evaluator"
    assert result["all_objectives"].shape == (2, 5)
    assert result["objective_names"][1] == "backbone_sanity_validity"
    assert result["objective_names"][4] == "icosahedral_interface_geometry_proxy"


def test_canonical_proteus_update_rejects_untrained_policy():
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
    model.prepare_for_training("sequence")
    msa = torch.randint(0, 23, (1, 2, 4))
    pair = torch.zeros(1, 4, 4, 16)
    with pytest.raises(RuntimeError, match="supervised, validated sequence policy"):
        model.update_from_proteus(
            ["ACDE"], ["AAAA"], msa, pair, n_dpo_steps=1, learning_rate=1e-3
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


def test_canonical_reports_unavailable_for_unaligned_structural_retrieval():
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
    output = model(
        torch.randint(0, 23, (1, 2, 8)),
        torch.zeros(1, 8, 8, 16),
        torch.eye(3).expand(1, 8, 3, 3),
        torch.zeros(1, 8, 3),
        substrate_id=torch.tensor([0]),
        use_rag=True,
    )
    assert output["rag_status"] == "RAG_UNAVAILABLE: query/index embedding spaces are not aligned"


def test_canonical_random_components_are_not_frozen_as_pretrained():
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
    assert all(p.requires_grad for p in model.flow_model.parameters())
    assert all(p.requires_grad for p in model.multi_scale_designer.parameters())
    assert all(p.requires_grad for p in model.pareto_head.parameters())
    assert model.inference_readiness()["state"] == "UNINITIALIZED"
    with pytest.raises(RuntimeError, match="experimental mode"):
        model.require_inference_ready()

    model.prepare_for_training("sequence")
    assert all(p.requires_grad for p in model.sequence_policy.parameters())
    assert all(not p.requires_grad for p in model.flow_model.parameters())


def test_compatible_component_checkpoint_is_loaded_but_not_assumed_trained(tmp_path):
    from chimera.components import MSARepresentationBackbone

    checkpoint = tmp_path / "local_msa.pt"
    backbone = MSARepresentationBackbone(d_single=32, d_pair=16)
    torch.save({"state_dict": backbone.state_dict()}, checkpoint)
    model = CHIMERAv2.from_pretrained(
        evoformer_ckpt=str(checkpoint),
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

    state = model.component_status["evoformer"]
    assert state["initialization"] == "loaded_unverified"
    assert state["checkpoint_sha256"]
    assert all(parameter.requires_grad for parameter in model.evoformer.parameters())
    assert model.inference_readiness()["state"] == "UNINITIALIZED"
    assert model.objective_provenance()["evolutionary_plausibility"]["source"] == "random_neural_surrogate"


def test_incompatible_local_checkpoint_fails_closed(tmp_path):
    checkpoint = tmp_path / "wrong_schema.pt"
    torch.save({"state_dict": {"not_a_model_parameter": torch.zeros(1)}}, checkpoint)
    with pytest.raises(RuntimeError, match="checkpoint is incompatible"):
        CHIMERAv2.from_pretrained(
            evoformer_ckpt=str(checkpoint),
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


def test_trainer_rejects_optimizer_parameters_outside_selected_regime():
    from chimera.training import CanonicalTrainer, TrainingRegime

    model = CHIMERAv2(
        d_evo_single=32, d_evo_pair=16, d_se3=32, d_pair_out=32,
        d_mpnn=16, n_flow_blocks=1, n_domains=2, n_modules=2, n_mc_dropout=2,
    )
    optimizer = torch.optim.AdamW(model.flow_model.parameters(), lr=1e-4)
    with pytest.raises(ValueError, match="exactly match the selected regime"):
        CanonicalTrainer(model, TrainingRegime.SEQUENCE, optimizer=optimizer)


def test_canonical_flow_regime_updates_expected_gradient_paths():
    from chimera.lie import so3_exp
    from chimera.training import CanonicalTrainer, CanonicalTrainingBatch, TrainingRegime

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
    trainer = CanonicalTrainer(model, TrainingRegime.FLOW, learning_rate=1e-4)
    batch_size, msa_depth, length = 1, 2, 4
    source_R = torch.eye(3).reshape(1, 1, 3, 3).expand(batch_size, length, -1, -1).clone()
    target_R = so3_exp(torch.randn(batch_size, length, 3) * 0.1)
    batch = CanonicalTrainingBatch(
        msa_tokens=torch.randint(0, 20, (batch_size, msa_depth, length)),
        pair_features=torch.zeros(batch_size, length, length, 16),
        source_R=source_R,
        source_t=torch.zeros(batch_size, length, 3),
        target_R=target_R,
        target_t=torch.randn(batch_size, length, 3),
    )

    result = trainer.train_step(batch)

    assert torch.isfinite(torch.tensor(result["loss"]))
    for component in ("evoformer", "pair_connector", "flow_model", "evol_cross_attn"):
        component_gradients = result["gradient_flow"]["components"][component]
        assert component_gradients["gradient_parameters"] > 0
        assert component_gradients["gradient_norm"] > 0
        assert component_gradients["all_gradients_finite"]
    assert not any(parameter.requires_grad for parameter in model.sequence_policy.parameters())
    CanonicalTrainer(model, TrainingRegime.SEQUENCE)
    assert all(not parameter.requires_grad for parameter in model.flow_model.parameters())
    assert all(parameter.grad is None for parameter in model.flow_model.parameters())


def test_representation_regime_has_explicit_masked_token_gradient_path():
    from chimera.training import CanonicalTrainer, CanonicalTrainingBatch, TrainingRegime

    model = CHIMERAv2(
        d_evo_single=32, d_evo_pair=16, d_se3=32, d_pair_out=32,
        d_mpnn=16, n_flow_blocks=1, n_domains=2, n_modules=2, n_mc_dropout=2,
    )
    trainer = CanonicalTrainer(
        model,
        TrainingRegime.REPRESENTATION,
        dataset_manifest="msa-train-v1",
    )
    batch = CanonicalTrainingBatch(
        msa_tokens=torch.tensor([[[1, 2, 3, 4], [1, 5, 6, 7]]]),
        pair_features=torch.zeros(1, 4, 4, 16),
    )

    result = trainer.train_step(batch)

    gradients = result["gradient_flow"]["components"]["evoformer"]
    assert torch.isfinite(torch.tensor(result["loss"]))
    assert gradients["gradient_parameters"] > 0
    assert gradients["gradient_norm"] > 0
    assert model.evoformer.msa_column_encoder.layers[0].self_attn.in_proj_weight.grad is not None
    assert not model.evoformer.pair_init.weight.requires_grad
    assert model.component_status["evoformer"]["training_status"] == "in_progress"
    trainer.validate(batch, validation_manifest="msa-heldout-v1")
    assert model.component_status["evoformer"]["training_status"] == "validated"


def test_constraint_regime_trains_constraint_encoder_with_supervised_bridge_loss():
    from chimera.chimera_v2 import NRPSConstraints
    from chimera.lie import so3_exp
    from chimera.training import CanonicalTrainer, CanonicalTrainingBatch, TrainingRegime

    model = CHIMERAv2(
        d_evo_single=32, d_evo_pair=16, d_se3=32, d_pair_out=32,
        d_mpnn=16, n_flow_blocks=1, n_domains=2, n_modules=2, n_mc_dropout=2,
    )
    constraints = NRPSConstraints(
        fixed_mask=torch.zeros(1, 4, dtype=torch.bool),
        stachelhaus_positions=torch.tensor([1]),
        domain_boundaries=torch.tensor([[[0, 2], [2, 4]]]),
        module_boundaries=torch.tensor([[[0, 2], [2, 4]]]),
        icosahedral_face=torch.tensor([0]),
        ppt_serine_position=1,
        hotspot_coords=None,
        hotspot_indices=None,
        target_substrate="PHE",
    )
    rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(1, 4, -1, -1).clone()
    batch = CanonicalTrainingBatch(
        msa_tokens=torch.randint(0, 20, (1, 2, 4)),
        pair_features=torch.zeros(1, 4, 4, 16),
        source_R=rotations,
        source_t=torch.zeros(1, 4, 3),
        target_R=so3_exp(torch.randn(1, 4, 3) * 0.1),
        target_t=torch.randn(1, 4, 3),
        constraints=constraints,
    )
    trainer = CanonicalTrainer(model, TrainingRegime.CONSTRAINT)

    result = trainer.train_step(batch)

    constraint_gradients = result["gradient_flow"]["components"]["constraint_encoder"]
    assert constraint_gradients["gradient_parameters"] > 0
    assert constraint_gradients["gradient_norm"] > 0
    assert model.component_status["constraint_encoder"]["training_status"] == "in_progress"


def test_objective_regime_requires_labels_and_tracks_proxy_source():
    from chimera.training import CanonicalTrainer, CanonicalTrainingBatch, TrainingRegime

    model = CHIMERAv2(
        d_evo_single=32, d_evo_pair=16, d_se3=32, d_pair_out=32,
        d_mpnn=16, n_flow_blocks=1, n_domains=2, n_modules=2, n_mc_dropout=2,
    )
    trainer = CanonicalTrainer(
        model,
        TrainingRegime.OBJECTIVE,
        dataset_manifest="objective-train-v1",
    )
    batch = CanonicalTrainingBatch(
        msa_tokens=torch.randint(0, 20, (1, 2, 4)),
        pair_features=torch.zeros(1, 4, 4, 16),
        target_sequence=torch.randint(0, 20, (1, 4)),
        objective_labels={"expression_efficiency": torch.tensor([0.5])},
        objective_label_sources={"expression_efficiency": "measured-expression-v1"},
    )
    moved_batch = batch.to("cpu")
    assert moved_batch.objective_labels["expression_efficiency"].device.type == "cpu"

    result = trainer.train_step(batch)
    for component in ("_seq_to_repr", "pareto_head"):
        component_gradients = result["gradient_flow"]["components"][component]
        assert component_gradients["gradient_parameters"] > 0
        assert component_gradients["gradient_norm"] > 0
    assert model.objective_status["expression_efficiency"]["training_status"] == "in_progress"
    assert model.objective_status["evolutionary_plausibility"]["training_steps"] == 0

    trainer.validate(batch, validation_manifest="expression-heldout-v1")
    calibration = model.record_objective_calibration(
        "expression_efficiency",
        predicted_mean=torch.zeros(32),
        epistemic_std=torch.ones(32),
        targets=torch.tensor([0.0] * 22 + [2.0] * 10),
        calibration_manifest="expression-calibration-v1",
    )
    provenance = model.objective_provenance()["expression_efficiency"]
    assert provenance["training_status"] == "validated"
    assert provenance["label_sources"] == ["measured-expression-v1"]
    assert provenance["biological_measurement"] is False
    assert provenance["calibrated"] is True
    assert calibration["examples"] == 32
    assert calibration["method"] == "heldout_one_sigma_coverage"
    assert calibration["one_sigma_coverage"] == pytest.approx(22 / 32)


def test_canonical_sequence_regime_and_checkpoint_resume(tmp_path):
    from chimera.training import CanonicalTrainer, CanonicalTrainingBatch, TrainingRegime

    def make_model():
        return CHIMERAv2(
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

    model = make_model()
    trainer = CanonicalTrainer(
        model,
        TrainingRegime.SEQUENCE,
        learning_rate=1e-4,
        dataset_manifest="train-v1",
    )
    length = 4
    source_R = torch.eye(3).reshape(1, 1, 3, 3).expand(1, length, -1, -1).clone()
    source_t = torch.arange(length).reshape(1, length, 1).float() * torch.tensor([3.8, 0.0, 0.0])
    batch = CanonicalTrainingBatch(
        msa_tokens=torch.randint(0, 20, (1, 2, length)),
        pair_features=torch.zeros(1, length, length, 16),
        source_R=source_R,
        source_t=source_t,
        target_sequence=torch.randint(0, 20, (1, length)),
    )
    result = trainer.train_step(batch)

    for component in ("evoformer", "node_connector", "base_mpnn", "multi_scale_designer", "_seq_to_repr", "sequence_policy"):
        assert result["gradient_flow"]["components"][component]["gradient_parameters"] > 0
    assert all(not p.requires_grad for p in model.flow_model.parameters())
    validation_loss = trainer.validate_sequence_policy(batch, validation_manifest="heldout-v1")
    assert torch.isfinite(torch.tensor(validation_loss))

    path = tmp_path / "sequence-stage.pt"
    trainer.save_checkpoint(path)
    checkpoint_policy = [parameter.detach().clone() for parameter in model.sequence_policy.parameters()]
    restored_model = make_model()
    restored = CanonicalTrainer(
        restored_model,
        TrainingRegime.SEQUENCE,
        learning_rate=1e-4,
        dataset_manifest="train-v1",
    )
    restored.resume(path)
    assert restored.global_step == 1
    assert restored.epoch == 0
    assert restored_model.component_status["sequence_policy"]["training_status"] == "validated"
    assert all(
        torch.equal(saved, current)
        for saved, current in zip(checkpoint_policy, restored_model.sequence_policy.parameters())
    )

    from chimera.dpo import DPOBatch
    preference = CanonicalTrainer(model, TrainingRegime.PREFERENCE, learning_rate=1e-4)
    rejected = (batch.target_sequence + 1) % 20
    preference_result = preference.train_preference_step(DPOBatch(
        context=torch.zeros(1, length, 16),
        chosen=batch.target_sequence,
        rejected=rejected,
        chosen_mask=torch.ones_like(rejected, dtype=torch.bool),
        rejected_mask=torch.ones_like(rejected, dtype=torch.bool),
    ))
    preference_gradients = preference_result["gradient_flow"]["components"]["sequence_policy"]
    assert preference_gradients["gradient_parameters"] > 0
    assert preference_gradients["gradient_norm"] > 0
    preference_path = tmp_path / "preference-stage.pt"
    reference_weights = {
        name: value.detach().clone()
        for name, value in model._reference_policy.state_dict().items()
    }
    preference.save_checkpoint(preference_path)
    resumed_preference = CanonicalTrainer.from_checkpoint(make_model(), preference_path)
    assert resumed_preference.global_step == 1
    assert resumed_preference.model._reference_policy is not None
    assert all(not parameter.requires_grad for parameter in resumed_preference.model._reference_policy.parameters())
    assert all(
        torch.equal(reference_weights[name], value)
        for name, value in resumed_preference.model._reference_policy.state_dict().items()
    )


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
