import torch

from chimera import CanonicalCHIMERAv2
from chimera.components import MSARepresentationBackbone
from chimera.evoformer_stack import (
    EvoFormerBlock,
    MSAColumnAttention,
    MSARowAttentionWithPairBias,
    OuterProductMean,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


def _small_stack(**kwargs):
    n_blocks = kwargs.pop("n_blocks", 2)
    return MSARepresentationBackbone(
        d_single=16,
        d_pair=8,
        n_blocks=n_blocks,
        n_heads_msa=4,
        n_heads_pair=2,
        c_hidden_opm=4,
        c_hidden_triangle=8,
        opm_chunk_size=2,
        attention_chunk_size=2,
        **kwargs,
    )


def test_default_depth_and_requested_depth_are_real_module_counts():
    shallow = _small_stack(n_blocks=3)
    full_depth = MSARepresentationBackbone(
        d_single=8,
        d_pair=4,
        n_blocks=48,
        n_heads_msa=2,
        n_heads_pair=2,
        c_hidden_opm=2,
        c_hidden_triangle=2,
    )
    assert len(shallow.blocks) == 3
    assert len(full_depth.blocks) == full_depth.n_blocks == 48
    assert len(MSARepresentationBackbone(
        d_single=8, d_pair=4, n_heads_msa=2, n_heads_pair=2,
        c_hidden_opm=2, c_hidden_triangle=2,
    ).blocks) == 48


def test_canonical_composition_passes_configured_depth_into_the_stack():
    model = CanonicalCHIMERAv2(
        d_evo_single=16,
        d_evo_pair=8,
        d_se3=16,
        d_pair_out=16,
        d_mpnn=16,
        evoformer_n_blocks=3,
        n_flow_blocks=1,
        n_domains=1,
        n_modules=1,
        n_mpnn_seqs=1,
        n_mc_dropout=2,
    )
    assert len(model.evoformer.blocks) == 3
    assert model.model_configuration()["evoformer"]["n_blocks"] == 3


def test_stack_emits_query_single_and_coupled_msa_pair_shapes():
    torch.manual_seed(1)
    model = _small_stack().eval()
    tokens = torch.randint(0, 20, (2, 3, 5))
    pairs = torch.randn(2, 5, 5, 8)

    output = model.forward_with_representations(tokens, pairs)

    assert output.msa_repr.shape == (2, 3, 5, 16)
    assert output.pair_repr.shape == (2, 5, 5, 8)
    assert output.single_repr.shape == (2, 5, 16)
    assert torch.allclose(
        output.single_repr,
        model.single_projection(output.msa_repr[:, 0]),
        atol=1e-6,
    )
    assert not torch.allclose(
        output.single_repr,
        model.single_projection(output.msa_repr.mean(dim=1)),
    )


def test_block_contains_the_nine_distinct_evoformer_updates():
    block = _small_stack().blocks[0]
    assert isinstance(block, EvoFormerBlock)
    assert all(
        item is not None
        for item in (
        block.msa_row_attention,
        block.msa_column_attention,
        block.msa_transition,
        block.outer_product_mean,
        block.triangle_multiplication_outgoing,
        block.triangle_multiplication_incoming,
        block.triangle_attention_starting,
        block.triangle_attention_ending,
        block.pair_transition,
        )
    )


def test_individual_evoformer_block_preserves_both_stream_shapes():
    block = _small_stack().blocks[0].eval()
    msa = torch.randn(1, 2, 4, 16)
    pair = torch.randn(1, 4, 4, 8)
    msa_mask = torch.ones(1, 2, 4, dtype=torch.bool)
    pair_mask = torch.ones(1, 4, 4, dtype=torch.bool)

    next_msa, next_pair, _ = block(msa, pair, msa_mask, pair_mask)

    assert next_msa.shape == msa.shape
    assert next_pair.shape == pair.shape


def test_row_attention_is_changed_by_pair_derived_bias():
    torch.manual_seed(2)
    operation = MSARowAttentionWithPairBias(
        c_m=8, c_z=4, n_heads=2, dropout=0.0, depth=1
    ).eval()
    msa = torch.randn(1, 2, 3, 8)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    pair_a = torch.randn(1, 3, 3, 4)
    pair_b = pair_a.clone()
    pair_b[:, 0, 2] += torch.tensor([2.0, -1.0, 0.5, 3.0])

    result_a = operation(msa, pair_a, mask)
    result_b = operation(msa, pair_b, mask)

    assert not torch.allclose(result_a, result_b)


def test_column_attention_propagates_across_rows_at_a_fixed_residue():
    torch.manual_seed(3)
    operation = MSAColumnAttention(c_m=8, n_heads=2, dropout=0.0, depth=1).eval()
    msa_a = torch.randn(1, 2, 2, 8)
    msa_b = msa_a.clone()
    msa_b[:, 1, 0, 0] += 2.0
    mask = torch.ones(1, 2, 2, dtype=torch.bool)

    result_a = operation(msa_a, mask)
    result_b = operation(msa_b, mask)

    assert not torch.allclose(result_a[:, 0, 0], result_b[:, 0, 0])


def test_outer_product_mean_updates_pair_from_each_valid_msa_row():
    torch.manual_seed(4)
    operation = OuterProductMean(
        c_m=8, c_z=4, c_hidden=3, depth=1, chunk_size=2
    ).eval()
    msa_a = torch.randn(1, 2, 3, 8)
    msa_b = msa_a.clone()
    msa_b[:, 1, 0, 0] += 1.5
    mask = torch.ones(1, 2, 3, dtype=torch.bool)

    update_a = operation(msa_a, mask)
    update_b = operation(msa_b, mask)

    assert update_a.shape == (1, 3, 3, 4)
    assert not torch.allclose(update_a, update_b)


def test_masked_outer_product_mean_ignores_invalid_msa_rows():
    torch.manual_seed(41)
    operation = OuterProductMean(
        c_m=8, c_z=4, c_hidden=3, depth=1, chunk_size=2
    ).eval()
    msa_a = torch.randn(1, 2, 3, 8)
    msa_b = msa_a.clone()
    msa_b[:, 1, 2, 0] += 50.0
    mask = torch.tensor([[[True, True, False], [False, False, False]]])

    update_a = operation(msa_a, mask)
    update_b = operation(msa_b, mask)

    assert torch.allclose(update_a, update_b)
    assert torch.count_nonzero(update_a[:, 2]) == 0
    assert torch.count_nonzero(update_a[:, :, 2]) == 0


def test_triangle_multiplication_contracts_the_correct_intermediate_edges():
    torch.manual_seed(5)
    pair = torch.randn(1, 3, 3, 4)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    outgoing = TriangleMultiplicationOutgoing(4, 5, 1, 0.0).eval()
    incoming = TriangleMultiplicationIncoming(4, 5, 1, 0.0).eval()

    changed_out = pair.clone()
    changed_out[:, 0, 2, 0] += 2.0
    changed_in = pair.clone()
    changed_in[:, 2, 0, 0] += 2.0

    assert not torch.allclose(
        outgoing(pair, mask)[:, 0, 1], outgoing(changed_out, mask)[:, 0, 1]
    )
    assert not torch.allclose(
        incoming(pair, mask)[:, 0, 1], incoming(changed_in, mask)[:, 0, 1]
    )


def test_triangle_attention_supports_both_pair_orientations():
    torch.manual_seed(6)
    pair = torch.randn(1, 3, 3, 8, requires_grad=True)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    starting = TriangleAttentionStartingNode(8, 2, 1, 0.0).eval()
    ending = TriangleAttentionEndingNode(8, 2, 1, 0.0).eval()

    start_result = starting(pair, mask)
    end_result = ending(pair, mask)
    start_grad = torch.autograd.grad(start_result[0, 0, 1].sum(), pair, retain_graph=True)[0]
    end_grad = torch.autograd.grad(end_result[0, 0, 1].sum(), pair)[0]

    assert start_result.shape == end_result.shape == pair.shape
    assert torch.isfinite(start_result).all() and torch.isfinite(end_result).all()
    assert start_grad[0, 0, 2].abs().sum() > 0
    assert end_grad[0, 2, 1].abs().sum() > 0


def test_padding_is_invariant_and_invalid_pair_updates_are_zero():
    torch.manual_seed(7)
    model = _small_stack().eval()
    tokens_a = torch.tensor([[[1, 2, 3, 22], [4, 5, 6, 22]]])
    tokens_b = tokens_a.clone()
    tokens_b[:, :, 3] = 17
    padding = tokens_a.eq(22)
    pair_features = torch.randn(1, 4, 4, 8)

    result_a = model.forward_with_representations(
        tokens_a, pair_features, msa_padding_mask=padding
    )
    result_b = model.forward_with_representations(
        tokens_b, pair_features, msa_padding_mask=padding
    )

    assert torch.allclose(result_a.msa_repr[:, :, :3], result_b.msa_repr[:, :, :3])
    assert torch.allclose(result_a.pair_repr[:, :3, :3], result_b.pair_repr[:, :3, :3])
    assert torch.count_nonzero(result_a.msa_repr[:, :, 3]) == 0
    assert torch.count_nonzero(result_a.single_repr[:, 3]) == 0
    assert torch.count_nonzero(result_a.pair_repr[:, 3]) == 0
    assert torch.count_nonzero(result_a.pair_repr[:, :, 3]) == 0


def test_fully_padded_msa_produces_finite_zero_representations():
    model = _small_stack().eval()
    tokens = torch.full((1, 2, 3), 22, dtype=torch.long)
    pair = torch.randn(1, 3, 3, 8)
    output = model.forward_with_representations(tokens, pair)

    assert torch.isfinite(output.msa_repr).all()
    assert torch.isfinite(output.pair_repr).all()
    assert torch.isfinite(output.single_repr).all()
    assert torch.count_nonzero(output.msa_repr) == 0
    assert torch.count_nonzero(output.pair_repr) == 0
    assert torch.count_nonzero(output.single_repr) == 0


def test_explicit_residue_mask_blocks_msa_and_pair_information():
    torch.manual_seed(71)
    model = _small_stack().eval()
    tokens_a = torch.randint(0, 20, (1, 2, 4))
    tokens_b = tokens_a.clone()
    tokens_b[:, :, 2] = (tokens_b[:, :, 2] + 7) % 20
    residue_mask = torch.tensor([[True, True, False, True]])
    pair = torch.randn(1, 4, 4, 8)

    output_a = model.forward_with_representations(
        tokens_a, pair, residue_mask=residue_mask
    )
    output_b = model.forward_with_representations(
        tokens_b, pair, residue_mask=residue_mask
    )

    assert torch.allclose(output_a.single_repr[:, [0, 1, 3]], output_b.single_repr[:, [0, 1, 3]])
    assert torch.allclose(output_a.pair_repr[:, :2, :2], output_b.pair_repr[:, :2, :2])
    assert torch.count_nonzero(output_a.pair_repr[:, 2]) == 0
    assert torch.count_nonzero(output_a.pair_repr[:, :, 2]) == 0


def test_relative_positions_and_caller_pair_features_both_initialize_pair_state():
    torch.manual_seed(8)
    model = _small_stack().eval()
    tokens = torch.randint(0, 20, (1, 2, 4))
    pair = torch.zeros(1, 4, 4, 8)
    pair_changed = pair.clone()
    pair_changed[:, 0, 1, 0] = 1.0
    indices = torch.tensor([[0, 1, 4, 5]])

    _, baseline = model(tokens, pair)
    _, caller_conditioned = model(tokens, pair_changed)
    _, gapped_indices = model(tokens, pair, residue_index=indices)

    assert not torch.allclose(baseline, caller_conditioned)
    assert not torch.allclose(baseline, gapped_indices)


def test_all_major_evoformer_operations_receive_finite_gradients():
    torch.manual_seed(9)
    model = _small_stack()
    tokens = torch.randint(0, 20, (1, 3, 5))
    pair = torch.randn(1, 5, 5, 8)
    loss, loss_parts = model.representation_loss(
        tokens,
        pair,
        mask_probability=0.4,
        generator=torch.Generator().manual_seed(23),
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert loss_parts["masked_msa_loss"] > 0
    assert loss_parts["pair_coevolution_loss"] >= 0
    gradients = model.gradient_diagnostics()
    assert all(value > 0 for value in gradients.values()), gradients


def test_gradient_checkpointing_preserves_gradients_for_each_block():
    model = _small_stack(gradient_checkpointing=True).train()
    tokens = torch.randint(0, 20, (1, 2, 4))
    pair = torch.randn(1, 4, 4, 8)

    single, pair_output = model(tokens, pair)
    (single.square().mean() + pair_output.square().mean()).backward()

    assert all(block.msa_transition.output.weight.grad is not None for block in model.blocks)


def test_diagnostics_are_opt_in_and_report_per_block_updates():
    model = _small_stack().eval()
    tokens = torch.randint(0, 20, (1, 2, 4))
    pair = torch.zeros(1, 4, 4, 8)

    without = model.forward_with_representations(tokens, pair)
    with_diagnostics = model.forward_with_representations(
        tokens, pair, collect_diagnostics=True
    )

    assert without.diagnostics == {}
    assert "msa_activation_norm" in with_diagnostics.diagnostics
    assert len(with_diagnostics.diagnostics["per_block_update_norms"]) == 2
