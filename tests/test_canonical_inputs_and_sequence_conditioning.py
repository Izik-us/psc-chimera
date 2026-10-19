import hashlib

import torch

from chimera.components import NodeProjectionConnector
from data.canonical_inputs import (
    PREPROCESSING_ID,
    dataset_sample_to_prepared_inputs,
    dataset_sample_to_training_batch,
    pair_features_from_geometry,
)
from data.geometry import derive_geometry


def _sample(sequence="ACDEFGHI"):
    length = len(sequence)
    coordinates = torch.zeros(length, 5, 3)
    positions = torch.arange(length, dtype=torch.float32) * 3.8
    coordinates[:, 1, 0] = positions
    coordinates[:, 0, :] = coordinates[:, 1, :] + torch.tensor([-1.2, 0.7, 0.0])
    coordinates[:, 2, :] = coordinates[:, 1, :] + torch.tensor([1.5, 0.0, 0.2])
    coordinates[:, 3, :] = coordinates[:, 1, :] + torch.tensor([2.1, 0.8, 0.2])
    coordinates[:, 4, :] = coordinates[:, 1, :] + torch.tensor([0.0, 1.5, 0.0])
    atom_mask = torch.ones(length, 5, dtype=torch.bool)
    residue_mask = torch.ones(length, dtype=torch.bool)
    geometry = derive_geometry(coordinates, atom_mask, residue_mask, k_neighbors=4)
    token_map = {letter: index for index, letter in enumerate("ACDEFGHIKLMNPQRSTVWY")}
    query = torch.tensor([token_map.get(letter, 20) for letter in sequence], dtype=torch.long)
    homolog = query.clone()
    if length > 4:
        homolog[4:] = 22
    return {
        "sample_id": "test:chain:entity",
        "structure_id": "TEST",
        "sequence": sequence,
        "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
        "msa_tokens": torch.stack((query, homolog)),
        "geometry": geometry,
        "residue_mask": residue_mask,
        "residue_mappings": [{"sequence_index": index} for index in range(length)],
        "msa_metadata": {"source": "synthetic-test-alignment"},
        "provenance": {"source_database": "synthetic-test-structure"},
    }


def test_geometry_pair_features_are_shape_stable_and_mask_invalid_pairs():
    sample = _sample()
    geometry = sample["geometry"]
    features = pair_features_from_geometry(geometry, channels=128)
    assert features.shape == (8, 8, 128)
    assert torch.isfinite(features).all()
    assert torch.count_nonzero(features[~geometry.pair_mask]) == 0


def test_dataset_sample_adapter_builds_provenance_bound_canonical_inference_inputs():
    prepared = dataset_sample_to_prepared_inputs(_sample(), pair_channels=128)
    assert prepared.msa_tokens.shape == (1, 2, 8)
    assert prepared.pair_features.shape == (1, 8, 8, 128)
    assert prepared.source_rotations.shape == (1, 8, 3, 3)
    assert prepared.source_translations.shape == (1, 8, 3)
    assert prepared.msa_padding_mask.shape == (1, 2, 8)
    assert prepared.msa_padding_mask[0, 1, 4:].all()
    assert prepared.residue_mask.all()
    assert prepared.manifest.preprocessing_id == PREPROCESSING_ID
    assert prepared.identity


def test_dataset_sample_adapter_preserves_training_contract_without_inventing_source_frames():
    batch = dataset_sample_to_training_batch(_sample(), pair_channels=128)
    assert batch.msa_tokens.shape == (1, 2, 8)
    assert batch.pair_features.shape == (1, 8, 8, 128)
    assert batch.target_R.shape == (1, 8, 3, 3)
    assert batch.target_t.shape == (1, 8, 3)
    assert batch.target_sequence.shape == (1, 8)
    assert batch.source_R is None
    assert batch.source_t is None
    assert not batch.sequence_padding_mask.any()


def test_unsupported_residues_are_excluded_from_sequence_supervision():
    batch = dataset_sample_to_training_batch(_sample("ACDEXGHI"), pair_channels=128)
    assert batch.target_sequence[0, 4].item() == 20
    assert batch.sequence_padding_mask[0, 4].item()


def test_node_projection_fuses_pair_context_and_backpropagates_through_it():
    torch.manual_seed(29)
    connector = NodeProjectionConnector(evo_dim=8, mpnn_dim=16, pair_dim=12)
    single = torch.randn(1, 5, 8)
    pair_a = torch.zeros(1, 5, 5, 12)
    pair_b = torch.randn(1, 5, 5, 12)
    mask = torch.tensor([[True, True, True, True, True]])

    output_a = connector(single, pair_features=pair_a, residue_mask=mask)
    output_b = connector(single, pair_features=pair_b, residue_mask=mask)
    assert output_a.shape == output_b.shape == (1, 5, 16)
    assert not torch.allclose(output_a, output_b)

    output_b.square().mean().backward()
    assert connector.pair_proj[0].weight.grad is not None
    assert torch.isfinite(connector.pair_proj[0].weight.grad).all()
    assert connector.pair_proj[0].weight.grad.abs().sum() > 0


def test_node_projection_masks_invalid_partner_pairs_and_invalid_query_residues():
    torch.manual_seed(31)
    connector = NodeProjectionConnector(evo_dim=8, mpnn_dim=16, pair_dim=12).eval()
    single = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, True, False, False]])
    pair_a = torch.randn(1, 4, 4, 12)
    pair_b = pair_a.clone()
    pair_b[:, :, 2:, :] = torch.randn_like(pair_b[:, :, 2:, :]) * 100
    pair_b[:, 2:, :, :] = torch.randn_like(pair_b[:, 2:, :, :]) * 100

    output_a = connector(single, pair_features=pair_a, residue_mask=mask)
    output_b = connector(single, pair_features=pair_b, residue_mask=mask)
    assert torch.allclose(output_a, output_b, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(output_a[:, 2:]) == 0


def test_canonical_sequence_loss_uses_pair_connector_and_assigns_gradients():
    from chimera import CHIMERAv2
    from chimera.training import CanonicalTrainer, CanonicalTrainingBatch, TrainingRegime

    torch.manual_seed(37)
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
        n_mc_dropout=1,
        evoformer_n_blocks=1,
        evoformer_attention_chunk_size=None,
        evoformer_opm_chunk_size=None,
    )
    trainer = CanonicalTrainer(model, TrainingRegime.SEQUENCE)
    assert "pair_connector" in trainer.trainable_components
    assert "substrate_conditioner" in trainer.trainable_components
    assert "constraint_encoder" in trainer.trainable_components
    assert not any(parameter.requires_grad for parameter in model.evoformer.parameters())

    length = 8
    msa = torch.randint(0, 20, (1, 2, length))
    pair = torch.randn(1, length, length, 16)
    rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(1, length, 3, 3).clone()
    translations = torch.zeros(1, length, 3)
    translations[0, :, 0] = torch.arange(length, dtype=torch.float32) * 3.8
    batch = CanonicalTrainingBatch(
        msa_tokens=msa,
        pair_features=pair,
        target_R=rotations,
        target_t=translations,
        target_sequence=torch.randint(0, 20, (1, length)),
        substrate_id=torch.tensor([3]),
        msa_padding_mask=msa.eq(22),
        residue_mask=torch.ones(1, length, dtype=torch.bool),
        residue_index=torch.arange(length).unsqueeze(0),
    )
    loss = trainer._sequence_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()

    pair_gradients = [
        parameter.grad
        for parameter in model.pair_connector.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None and torch.isfinite(gradient).all()
               and gradient.abs().sum() > 0 for gradient in pair_gradients)
    pair_projection_grad = model.node_connector.pair_proj[0].weight.grad
    assert pair_projection_grad is not None
    assert torch.isfinite(pair_projection_grad).all()
    assert pair_projection_grad.abs().sum() > 0


def test_explicit_source_frames_cannot_reuse_target_geometry_pair_features():
    import pytest

    sample = _sample()
    length = len(sample["sequence"])
    source_R = torch.eye(3).reshape(1, 1, 3, 3).expand(1, length, 3, 3).clone()
    source_t = torch.zeros(1, length, 3)
    source_t[0, :, 0] = torch.arange(length, dtype=torch.float32) * 3.8
    with pytest.raises(ValueError, match="must not be reused as source conditioning"):
        dataset_sample_to_training_batch(
            sample,
            pair_channels=128,
            source_frames=(source_R, source_t),
        )


def test_explicit_source_frames_accept_explicitly_paired_source_features():
    sample = _sample()
    length = len(sample["sequence"])
    source_R = torch.eye(3).reshape(1, 1, 3, 3).expand(1, length, 3, 3).clone()
    source_t = sample["geometry"].ca_coordinates.unsqueeze(0).clone()
    source_pair = torch.full((1, length, length, 128), 0.125)

    batch = dataset_sample_to_training_batch(
        sample,
        pair_channels=128,
        source_frames=(source_R, source_t),
        source_pair_features=source_pair,
    )
    assert torch.equal(batch.source_R, source_R)
    assert torch.equal(batch.source_t, source_t)
    assert torch.equal(batch.pair_features, source_pair)
    assert not torch.equal(batch.pair_features, dataset_sample_to_prepared_inputs(sample).pair_features)
