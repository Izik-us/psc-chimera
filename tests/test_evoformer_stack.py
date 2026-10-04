import torch

from chimera import CanonicalCHIMERAv2
from chimera.components import MSARepresentationBackbone
from chimera.evoformer_stack import (
    EvoFormerBlock,
    MSAColumnAttention,
    MSARowAttentionWithPairBias,
    OuterProductMean,
    _SharedDropout,
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


def _small_stack(**kwargs):
    n_blocks = kwargs.pop("n_blocks", 2)
    configuration = {
        "n_heads_msa": 4,
        "n_heads_pair": 2,
        "c_hidden_opm": 4,
        "c_hidden_triangle": 8,
        "opm_chunk_size": 2,
        "attention_chunk_size": 2,
    }
    configuration.update(kwargs)
    return MSARepresentationBackbone(
        d_single=16,
        d_pair=8,
        n_blocks=n_blocks,
        **configuration,
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


def test_evoformer_block_invokes_updates_in_reference_order():
    block = _small_stack(n_blocks=1).blocks[0].eval()
    observed = []
    names = (
        ("msa_row_attention", block.msa_row_attention),
        ("msa_column_attention", block.msa_column_attention),
        ("msa_transition", block.msa_transition),
        ("outer_product_mean", block.outer_product_mean),
        ("triangle_multiplication_outgoing", block.triangle_multiplication_outgoing),
        ("triangle_multiplication_incoming", block.triangle_multiplication_incoming),
        ("triangle_attention_starting", block.triangle_attention_starting),
        ("triangle_attention_ending", block.triangle_attention_ending),
        ("pair_transition", block.pair_transition),
    )
    handles = [module.register_forward_hook(lambda _m, _i, _o, name=name: observed.append(name))
               for name, module in names]
    try:
        block(
            torch.randn(1, 2, 3, 16),
            torch.randn(1, 3, 3, 8),
            torch.ones(1, 2, 3, dtype=torch.bool),
            torch.ones(1, 3, 3, dtype=torch.bool),
        )
    finally:
        for handle in handles:
            handle.remove()

    assert observed == [name for name, _ in names]


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


def test_row_pair_bias_is_indexed_by_query_residue_and_key_residue():
    operation = MSARowAttentionWithPairBias(
        c_m=8, c_z=4, n_heads=2, dropout=0.0, depth=1
    ).eval()
    with torch.no_grad():
        operation.pair_bias.weight.zero_()
        operation.pair_bias.weight[:, 0] = 1.0
    pair = torch.zeros(1, 3, 3, 4)
    pair[0, 1, 2] = torch.tensor([1.0, -1.0, 0.5, -0.5])

    observed = operation._project_pair_bias(pair)
    expected = operation.pair_bias(operation.norm_z(pair)).permute(0, 3, 1, 2)[:, None]

    assert observed.shape == (1, 1, 2, 3, 3)
    assert torch.equal(observed, expected)
    assert observed[0, 0, :, 1, 2].abs().sum() > 0
    assert torch.count_nonzero(observed[0, 0, :, 2, 1]) == 0


def test_msa_row_attention_matches_independent_pair_biased_equation():
    torch.manual_seed(341)
    operation = MSARowAttentionWithPairBias(
        c_m=8, c_z=4, n_heads=2, dropout=0.0, depth=1
    ).eval()
    msa = torch.randn(1, 2, 3, 8)
    pair = torch.randn(1, 3, 3, 4)
    mask = torch.tensor([[[True, True, False], [True, True, True]]])
    normalized = operation.norm_m(msa)
    normalized_pair = operation.norm_z(pair)
    pair_bias = operation.pair_bias(normalized_pair).permute(0, 3, 1, 2)
    expected_rows = []
    for row in range(msa.shape[1]):
        q = operation.q(normalized[:, row]).reshape(1, 3, 2, 4).transpose(1, 2)
        k = operation.k(normalized[:, row]).reshape(1, 3, 2, 4).transpose(1, 2)
        v = operation.v(normalized[:, row]).reshape(1, 3, 2, 4).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / (4**0.5)
        logits = logits + pair_bias
        weights = torch.softmax(
            logits.masked_fill(~mask[:, row, None, None, :], torch.finfo(logits.dtype).min),
            dim=-1,
        )
        attended = (weights @ v).transpose(1, 2).reshape(1, 3, 8)
        attended = attended * torch.sigmoid(operation.gate(normalized[:, row]))
        attended = operation.output(attended) * mask[:, row, :, None]
        expected_rows.append(attended)
    expected = torch.stack(expected_rows, dim=1)

    assert torch.allclose(operation(msa, pair, mask), expected, atol=1e-6, rtol=1e-6)


def test_column_attention_ignores_perturbations_in_masked_sequences():
    torch.manual_seed(35)
    operation = MSAColumnAttention(c_m=8, n_heads=2, dropout=0.0, depth=1).eval()
    msa_a = torch.randn(1, 3, 2, 8)
    msa_b = msa_a.clone()
    msa_b[:, 2] += 100.0
    mask = torch.tensor([[[True, True], [True, True], [False, False]]])

    result_a = operation(msa_a, mask)
    result_b = operation(msa_b, mask)

    assert torch.allclose(result_a[:, :2], result_b[:, :2], atol=1e-6)
    assert torch.count_nonzero(result_a[:, 2]) == 0


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


def test_opm_duplicate_rows_preserve_the_mean_and_invalid_rows_do_not_contribute():
    torch.manual_seed(36)
    operation = OuterProductMean(
        c_m=8, c_z=4, c_hidden=3, depth=1, chunk_size=None
    ).eval()
    row = torch.randn(1, 1, 3, 8)
    duplicated = row.expand(1, 2, 3, 8).clone()
    one_row_mask = torch.ones(1, 1, 3, dtype=torch.bool)
    two_row_mask = torch.ones(1, 2, 3, dtype=torch.bool)

    one_row = operation(row, one_row_mask)
    duplicate_rows = operation(duplicated, two_row_mask)
    invalid_row = duplicated.clone()
    invalid_row[:, 1] += 100.0
    invalid_mask = torch.tensor([[[True, True, True], [False, False, False]]])
    masked = operation(invalid_row, invalid_mask)

    assert torch.allclose(one_row, duplicate_rows, atol=1e-3, rtol=1e-3)
    assert torch.allclose(one_row, masked, atol=1e-6)


def test_opm_masked_residue_and_chunked_execution_match_full_equation():
    torch.manual_seed(37)
    chunked = OuterProductMean(8, 4, 3, 1, chunk_size=1).eval()
    unchunked = OuterProductMean(8, 4, 3, 1, chunk_size=None).eval()
    unchunked.load_state_dict(chunked.state_dict())
    msa = torch.randn(1, 3, 4, 8)
    mask = torch.ones(1, 3, 4, dtype=torch.bool)
    mask[:, :, 2] = False

    expected = unchunked(msa, mask)
    observed = chunked(msa, mask)

    assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(observed[:, 2]) == 0
    assert torch.count_nonzero(observed[:, :, 2]) == 0


def test_opm_matches_independent_valid_sequence_mean_equation():
    torch.manual_seed(371)
    operation = OuterProductMean(4, 3, 2, 1, chunk_size=None).eval()
    with torch.no_grad():
        operation.output.bias.fill_(0.5)
    msa = torch.randn(1, 2, 2, 4)
    mask = torch.tensor([[[True, True], [True, False]]])
    normalized = operation.norm(msa)
    left, right = operation.left(normalized), operation.right(normalized)
    expected = torch.zeros(1, 2, 2, 3)
    for i in range(2):
        for j in range(2):
            count = 0
            outer_sum = torch.zeros(1, 2, 2)
            for sequence in range(2):
                if mask[0, sequence, i] and mask[0, sequence, j]:
                    outer_sum += torch.einsum(
                        "bh,be->bhe",
                        left[:, sequence, i],
                        right[:, sequence, j],
                    )
                    count += 1
            if count:
                projected = operation.output(outer_sum.reshape(1, 4))
                expected[:, i, j] = projected / (count + operation.epsilon)

    assert torch.allclose(operation(msa, mask), expected, atol=1e-6, rtol=1e-6)


def test_opm_uses_float32_contractions_under_cpu_autocast():
    torch.manual_seed(372)
    operation = OuterProductMean(4, 3, 2, 1, chunk_size=1).eval()
    msa = torch.randn(1, 2, 3, 4)
    mask = torch.ones(1, 2, 3, dtype=torch.bool)
    expected = operation(msa, mask)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_result = operation(msa, mask)
    bfloat_result = operation(msa.to(torch.bfloat16), mask)

    assert torch.isfinite(autocast_result).all()
    assert torch.allclose(autocast_result, expected, atol=1e-6, rtol=1e-6)
    assert bfloat_result.dtype == torch.bfloat16
    assert torch.isfinite(bfloat_result).all()


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


def test_triangle_multiplication_matches_independent_index_equations():
    torch.manual_seed(38)
    pair = torch.randn(1, 3, 3, 4)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    for operation, outgoing in (
        (TriangleMultiplicationOutgoing(4, 3, 1, 0.0).eval(), True),
        (TriangleMultiplicationIncoming(4, 3, 1, 0.0).eval(), False),
    ):
        normalized = operation.norm(pair)
        left = operation.left(normalized) * torch.sigmoid(operation.left_gate(normalized))
        right = operation.right(normalized) * torch.sigmoid(operation.right_gate(normalized))
        product = torch.zeros(1, 3, 3, 3)
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    if outgoing:
                        product[:, i, j] += left[:, i, k] * right[:, j, k]
                    else:
                        product[:, i, j] += left[:, k, i] * right[:, k, j]
        expected = operation.output(operation.output_norm(product))
        expected = expected * torch.sigmoid(operation.output_gate(normalized))

        assert torch.allclose(operation(pair, mask), expected, atol=1e-6, rtol=1e-6)


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


def test_triangle_attention_pair_bias_uses_each_key_pair_not_query_pair():
    operation = TriangleAttentionStartingNode(4, 2, 1, 0.0).eval()
    with torch.no_grad():
        operation.q.weight.zero_()
        operation.k.weight.zero_()
        operation.bias.weight.zero_()
        operation.bias.weight[:, 0] = 1.0
    pair = torch.zeros(1, 3, 3, 4)
    pair[0, 0, 1] = torch.tensor([1.0, 0.0, -1.0, 0.0])
    pair[0, 0, 2] = torch.tensor([0.0, 1.0, 0.0, -1.0])
    normalized = operation.norm(pair)
    rows = normalized[:, 0:1]
    q = operation.q(rows).reshape(1, 1, 3, 2, 2).permute(0, 1, 3, 2, 4)
    k = operation.k(rows).reshape(1, 1, 3, 2, 2).permute(0, 1, 3, 2, 4)

    logits = operation._attention_logits(q, k, rows)
    key_bias = operation._project_key_bias(rows)
    wrong_query_broadcast = operation.bias(rows).permute(0, 1, 3, 2).unsqueeze(-1)
    wrong_logits = (q @ k.transpose(-1, -2)) * (operation.c_head ** -0.5)
    wrong_logits = wrong_logits + wrong_query_broadcast

    assert logits.shape == (1, 1, 2, 3, 3)
    assert not torch.allclose(key_bias[0, 0, :, 0, 1], key_bias[0, 0, :, 0, 2])
    assert not torch.allclose(logits[0, 0, :, 0, 1], logits[0, 0, :, 0, 2])
    assert torch.allclose(wrong_logits[0, 0, :, 0, 1], wrong_logits[0, 0, :, 0, 2])


def test_triangle_attention_ending_bias_uses_transposed_key_pairs():
    operation = TriangleAttentionEndingNode(4, 2, 1, 0.0).eval()
    with torch.no_grad():
        operation.q.weight.zero_()
        operation.k.weight.zero_()
        operation.bias.weight.zero_()
        operation.bias.weight[:, 0] = 1.0
    pair = torch.zeros(1, 3, 3, 4)
    pair[0, 1, 0] = torch.tensor([1.0, 0.0, -1.0, 0.0])
    pair[0, 2, 0] = torch.tensor([0.0, 1.0, 0.0, -1.0])
    oriented = pair.transpose(1, 2)
    normalized = operation.norm(oriented)
    rows = normalized[:, 0:1]
    q = operation.q(rows).reshape(1, 1, 3, 2, 2).permute(0, 1, 3, 2, 4)
    k = operation.k(rows).reshape(1, 1, 3, 2, 2).permute(0, 1, 3, 2, 4)
    logits = operation._attention_logits(q, k, rows)
    key_bias = operation._project_key_bias(rows)

    assert not torch.allclose(key_bias[0, 0, :, 0, 1], key_bias[0, 0, :, 0, 2])
    assert not torch.allclose(logits[0, 0, :, 0, 1], logits[0, 0, :, 0, 2])


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


def test_canonical_msa_pair_streams_exchange_information_in_both_directions():
    torch.manual_seed(81)
    model = _small_stack(n_blocks=1).eval()
    tokens = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
    pair = torch.randn(1, 3, 3, 8)
    altered_pair = pair.clone()
    altered_pair[:, 0, 1, 0] += 3.0
    msa_changed = tokens.clone()
    msa_changed[:, 1, 0] = 12

    baseline_single, baseline_pair = model(tokens, pair)
    pair_conditioned_single, _ = model(tokens, altered_pair)
    _, msa_conditioned_pair = model(msa_changed, pair)

    assert not torch.allclose(baseline_single, pair_conditioned_single)
    assert not torch.allclose(baseline_pair, msa_conditioned_pair)


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
    single, _ = model(tokens, pair)
    loss = loss + 0.1 * single.square().mean()
    loss.backward()

    assert torch.isfinite(loss)
    assert loss_parts["masked_msa_loss"] > 0
    assert loss_parts["pair_coevolution_loss"] >= 0
    gradients = model.gradient_diagnostics()
    assert all(value > 0 for value in gradients.values()), gradients
    assert model.relative_position_embed.weight.grad is not None
    assert model.relative_position_embed.weight.grad.abs().sum() > 0
    assert model.single_projection.weight.grad is not None
    assert model.single_projection.weight.grad.abs().sum() > 0


def test_dropout_masks_share_the_reference_semantic_axes():
    torch.manual_seed(39)
    msa_dropout = _SharedDropout(0.5, shared_axis=1).train()
    pair_dropout = _SharedDropout(0.5, shared_axis=1).train()
    msa_mask = msa_dropout(torch.ones(2, 5, 4, 8))
    pair_mask = pair_dropout(torch.ones(2, 6, 6, 8))

    assert all(torch.equal(msa_mask[:, 0], msa_mask[:, row]) for row in range(1, 5))
    assert all(torch.equal(pair_mask[:, 0], pair_mask[:, row]) for row in range(1, 6))
    block = _small_stack(n_blocks=1).blocks[0]
    assert block.msa_row_attention.dropout.shared_axis == 1
    assert block.msa_column_attention.dropout.shared_axis == 1
    assert block.triangle_multiplication_outgoing.dropout.shared_axis == 1
    assert block.triangle_attention_starting.dropout.shared_axis == 1


def test_pair_dropout_sharing_is_independent_of_attention_chunking():
    chunked = _small_stack(n_blocks=1, attention_chunk_size=1, opm_chunk_size=1).eval()
    unchunked = _small_stack(
        n_blocks=1, attention_chunk_size=None, opm_chunk_size=None
    ).eval()
    unchunked.load_state_dict(chunked.state_dict())
    chunked.train()
    unchunked.train()
    msa = torch.randint(0, 20, (1, 3, 4))
    pair = torch.randn(1, 4, 4, 8)

    torch.manual_seed(40)
    single_chunked, pair_chunked = chunked(msa, pair)
    torch.manual_seed(40)
    single_full, pair_full = unchunked(msa, pair)

    assert torch.allclose(single_chunked, single_full, atol=1e-5, rtol=1e-5)
    assert torch.allclose(pair_chunked, pair_full, atol=1e-5, rtol=1e-5)


def test_configuration_records_transition_factor_and_chunking_choices():
    model = _small_stack(transition_factor=3, attention_chunk_size=None, opm_chunk_size=None)
    configuration = model.configuration
    assert configuration["transition_factor"] == 3
    assert configuration["attention_chunk_size"] is None
    assert configuration["opm_chunk_size"] is None
    assert configuration["opm_epsilon"] == 1e-3
    assert model.blocks[0].msa_transition.input.out_features == 16 * 3
    assert model.blocks[0].pair_transition.input.out_features == 8 * 3
    assert model.blocks[0].msa_row_attention.output.bias is not None
    assert model.blocks[0].triangle_attention_starting.output.bias is not None
    assert model.blocks[0].triangle_multiplication_outgoing.output.bias is not None


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
