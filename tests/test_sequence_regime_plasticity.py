import torch


def _model():
    from chimera import CHIMERAv2

    return CHIMERAv2(
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
        evoformer_n_blocks=1,
        evoformer_attention_chunk_size=None,
        evoformer_opm_chunk_size=None,
    )


def _batch(length=8):
    from chimera.domain_schema import NRPSConstraints
    from chimera.training import CanonicalTrainingBatch

    msa = torch.randint(0, 20, (1, 2, length))
    pair = torch.randn(1, length, length, 16)
    rotations = torch.eye(3).reshape(1, 1, 3, 3).expand(1, length, 3, 3).clone()
    translations = torch.zeros(1, length, 3)
    translations[0, :, 0] = torch.arange(length, dtype=torch.float32) * 3.8
    return CanonicalTrainingBatch(
        msa_tokens=msa,
        pair_features=pair,
        target_R=rotations,
        target_t=translations,
        target_sequence=torch.randint(0, 20, (1, length)),
        substrate_id=torch.tensor([3]),
        constraints=NRPSConstraints(
            fixed_mask=torch.zeros(1, length, dtype=torch.bool),
            stachelhaus_positions=torch.tensor([2]),
            domain_boundaries=torch.tensor([[[0, 4], [4, 8]]]),
            module_boundaries=torch.tensor([[[0, 4], [4, 8]]]),
            icosahedral_face=torch.tensor([1]),
            ppt_serine_position=3,
            hotspot_coords=None,
            hotspot_indices=None,
            target_substrate="ASP",
        ),
        msa_padding_mask=msa.eq(22),
        residue_mask=torch.ones(1, length, dtype=torch.bool),
        residue_index=torch.arange(length).unsqueeze(0),
    )


def test_sequence_backbone_plasticity_is_explicit_and_receives_gradient():
    from chimera.training import CanonicalTrainer, TrainingRegime

    torch.manual_seed(41)
    model = _model()
    trainer = CanonicalTrainer(
        model,
        TrainingRegime.SEQUENCE,
        allow_sequence_backbone_finetuning=True,
    )
    assert "evoformer" in trainer.trainable_components
    assert any(parameter.requires_grad for parameter in model.evoformer.parameters())

    result = trainer.train_step(_batch())
    assert torch.isfinite(torch.tensor(result["loss"]))
    evo_gradients = [
        parameter.grad
        for parameter in model.evoformer.parameters()
        if parameter.requires_grad
    ]
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and gradient.abs().sum() > 0
        for gradient in evo_gradients
    )


def test_sequence_default_keeps_evoformer_behaviorally_isolated():
    from chimera.training import CanonicalTrainer, TrainingRegime, _module_state_fingerprint

    torch.manual_seed(43)
    model = _model()
    before = _module_state_fingerprint(model.evoformer)
    trainer = CanonicalTrainer(model, TrainingRegime.SEQUENCE)
    trainer.train_step(_batch())
    after = _module_state_fingerprint(model.evoformer)
    assert before == after
    assert not any(parameter.requires_grad for parameter in model.evoformer.parameters())
