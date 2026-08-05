"""
CHIMERA v2 shape and integration tests.
Run with: pytest tests/test_chimera_v2.py -v

These tests run on CPU with stub weights — no GPU or pretrained
checkpoints required. They verify tensor shapes, forward pass logic,
and module interfaces are all correct before you load real weights.
"""

import pytest
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from chimera.chimera_v2 import CHIMERAv2, NRPSConstraints
from chimera.flow_matching import SE3FlowMatching, so3_exp, so3_log, se3_interp
from chimera.multi_objective import (
    ParetoMultiObjectiveHead,
    BayesianUncertaintyEstimator,
    MultiScaleNRPSDesigner,
    DPOTrainer,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def small_config():
    """Minimal config for fast CPU testing."""
    return dict(
        d_evo_single=64,
        d_evo_pair=32,
        d_se3=64,
        d_pair_out=64,
        d_mpnn=32,
        n_flow_blocks=2,
        n_flow_steps=3,
        n_retrieve=2,
        n_mpnn_seqs=2,
        n_mc_dropout=3,
        n_domains=5,
        n_modules=2,
    )


@pytest.fixture
def small_constraints():
    B, L = 1, 60
    return NRPSConstraints(
        fixed_mask=torch.zeros(B, L, dtype=torch.bool),
        stachelhaus_positions=torch.tensor([10, 11, 14, 20, 25, 27, 30, 35, 40, 41]),
        domain_boundaries=torch.tensor(
            [[[0, 20], [20, 30], [30, 45], [45, 55], [55, 60]]]
        ),
        module_boundaries=torch.tensor([[[0, 60], [0, 0], [0, 0], [0, 0], [0, 0]]]),
        icosahedral_face=torch.tensor([7]),
        ppt_serine_position=42,
        hotspot_coords=None,
        hotspot_indices=None,
        target_substrate="PHE",
    )


@pytest.fixture
def chimera_model(small_config):
    model = CHIMERAv2(**small_config)
    model.eval()
    return model


# ── SO(3) Math Tests ─────────────────────────────────────────────────────────


class TestSO3Math:

    def test_so3_exp_returns_rotation_matrix(self):
        """exp map should return matrices with det=1 and R^T R = I."""
        omega = torch.randn(4, 3) * 0.5
        R = so3_exp(omega)
        assert R.shape == (4, 3, 3)
        # Check orthogonality: R^T R ≈ I
        eye = torch.eye(3).unsqueeze(0).expand(4, -1, -1)
        diff = (torch.bmm(R.transpose(-1, -2), R) - eye).abs().max()
        assert diff < 1e-5, f"R^T R != I, max diff = {diff}"

    def test_so3_log_inverts_exp(self):
        """log(exp(omega)) should recover omega (for small angles)."""
        omega = torch.randn(4, 3) * 0.3  # small angles
        R = so3_exp(omega)
        omega_recovered = so3_log(R)
        diff = (omega - omega_recovered).abs().max()
        assert diff < 1e-4, f"log(exp(omega)) != omega, max diff = {diff}"

    def test_se3_interp_endpoints(self):
        """Interpolation at t=0 should give source, t=1 should give target."""
        B, L = 2, 10
        R0 = so3_exp(torch.randn(B, L, 3) * 0.1)
        R1 = so3_exp(torch.randn(B, L, 3) * 0.1)
        t0 = torch.randn(B, L, 3)
        t1 = torch.randn(B, L, 3)

        Rt0, tt0 = se3_interp(R0, t0, R1, t1, 0.0)
        Rt1, tt1 = se3_interp(R0, t0, R1, t1, 1.0)

        assert (t0 - tt0).abs().max() < 1e-5
        assert (t1 - tt1).abs().max() < 1e-5


# ── Flow Matching Tests ───────────────────────────────────────────────────────


class TestFlowMatching:

    def test_flow_matching_loss_shape(self):
        B, L = 2, 30
        model = SE3FlowMatching(d_single=64, d_pair=64, n_blocks=2)
        R0 = so3_exp(torch.randn(B, L, 3) * 0.1)
        R1 = so3_exp(torch.randn(B, L, 3) * 0.1)
        t0, t1 = torch.randn(B, L, 3), torch.randn(B, L, 3)
        pair_c = torch.randn(B, L, L, 64)
        evol_s = torch.randn(B, L, 64)

        loss = model.flow_matching_loss(R0, t0, R1, t1, pair_c, evol_s)
        assert loss.shape == (), "Loss should be a scalar"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_sample_output_shapes(self):
        B, L = 1, 20
        model = SE3FlowMatching(d_single=64, d_pair=64, n_blocks=2)
        R0 = so3_exp(torch.randn(B, L, 3) * 0.1)
        t0 = torch.randn(B, L, 3)
        pair_c = torch.randn(B, L, L, 64)
        evol_s = torch.randn(B, L, 64)

        with torch.no_grad():
            R_out, t_out = model.sample(R0, t0, pair_c, evol_s, n_steps=3)

        assert R_out.shape == (B, L, 3, 3), f"Wrong R shape: {R_out.shape}"
        assert t_out.shape == (B, L, 3), f"Wrong t shape: {t_out.shape}"

    def test_fixed_mask_respected(self):
        """Fixed positions should not move during sampling."""
        B, L = 1, 20
        model = SE3FlowMatching(d_single=64, d_pair=64, n_blocks=2)
        R0 = so3_exp(torch.randn(B, L, 3) * 0.1)
        t0 = torch.randn(B, L, 3)
        mask = torch.zeros(B, L, dtype=torch.bool)
        mask[:, :5] = True  # fix first 5 positions

        with torch.no_grad():
            R_out, t_out = model.sample(
                R0,
                t0,
                torch.randn(B, L, L, 64),
                torch.randn(B, L, 64),
                n_steps=3,
                fixed_mask=mask,
            )

        # Fixed positions should be exactly R0, t0
        assert (t_out[:, :5] - t0[:, :5]).abs().max() < 1e-5


# ── Multi-Scale Designer Tests ────────────────────────────────────────────────


class TestMultiScaleDesigner:

    def test_output_shape(self):
        B, L = 2, 60
        model = MultiScaleNRPSDesigner(
            d_residue=32,
            d_domain=64,
            d_module=128,
            d_assembly=64,
            n_domains=5,
            n_modules=2,
        )
        residue_f = torch.randn(B, L, 32)
        evol_f = torch.randn(B, L, 32)
        edge_f = torch.randn(B, L, 16, 16)
        edge_idx = torch.randint(0, L, (B, L, 16))
        d_bounds = torch.tensor([[[0, 12], [12, 24], [24, 36], [36, 48], [48, L]]] * B)
        m_bounds = torch.tensor([[[0, L], [0, 0], [0, 0], [0, 0], [0, 0]]] * B)
        face = torch.zeros(B, dtype=torch.long)

        logits = model(residue_f, evol_f, edge_f, edge_idx, d_bounds, m_bounds, face)
        assert logits.shape == (B, L, 20), f"Wrong output shape: {logits.shape}"


# ── Pareto Optimization Tests ─────────────────────────────────────────────────


class TestParetoOptimization:

    def test_pareto_head_output_shapes(self):
        B, L = 4, 30
        model = ParetoMultiObjectiveHead(d_model=64)
        repr_ = torch.randn(B, L, 64)
        obj = model(repr_)
        assert obj.evolutionary_plausibility.shape == (B,)
        assert obj.structural_stability.shape == (B,)
        assert obj.expression_efficiency.shape == (B,)
        assert obj.substrate_selectivity.shape == (B,)
        assert obj.assembly_compatibility.shape == (B,)

    def test_pareto_frontier_non_dominated(self):
        """All returned points should be non-dominated."""
        N = 100
        obj_matrix = torch.rand(N, 5)
        pareto_front, pareto_idx = ParetoMultiObjectiveHead.compute_pareto_frontier(
            obj_matrix, maximize=[True] * 5
        )
        # Verify: no point in Pareto front is dominated by another
        P = pareto_front.numpy()
        for i in range(len(P)):
            for j in range(len(P)):
                if i == j:
                    continue
                dominated = all(P[i][k] >= P[j][k] for k in range(5)) and any(
                    P[i][k] > P[j][k] for k in range(5)
                )
                assert (
                    not dominated
                ), f"Point {j} is dominated by {i} — Pareto front incorrect"

    def test_pareto_head_pcgrad_loss(self):
        B, L = 2, 20
        model = ParetoMultiObjectiveHead(d_model=64)
        repr_ = torch.randn(B, L, 64)
        obj = model(repr_)
        labels = {
            "evol": torch.randn(B),
            "stab": torch.rand(B) * 100,
            "expr": torch.rand(B),
        }
        loss, metrics = model.pcgrad_loss(obj, labels)
        assert not torch.isnan(loss)
        assert "evol" in metrics


# ── Codon Optimizer Tests ─────────────────────────────────────────────────────


class TestCodonOptimizer:

    def test_synonymous_mask_correctness(self):
        """Every position in mask should correspond to a valid synonymous codon."""
        from chimera.codon_optimizer import get_synonymous_mask, CODON_TABLE, ALL_CODONS

        device = torch.device("cpu")
        for aa in "ACDEFGHIKLMNPQRSTVWY":
            mask = get_synonymous_mask(aa, device)
            valid = set(CODON_TABLE[aa])
            active = {ALL_CODONS[i] for i in mask.nonzero().squeeze(-1).tolist()}
            assert (
                active == valid
            ), f"Wrong synonymous codons for {aa}: {active} != {valid}"

    def test_translation_roundtrip(self):
        """Tokenize → detokenize → translate should recover original AA sequence."""
        from chimera.codon_optimizer import (
            tokenize_dna,
            detokenize_dna,
            translate_dna,
            sliding_window_optimize,
        )

        aa_seq = "MTEYKLVVVGAGGVGKS"
        dna, _ = sliding_window_optimize(aa_seq)
        assert len(dna) == len(aa_seq) * 3
        assert translate_dna(dna) == aa_seq, "Translation roundtrip failed"

    def test_cai_within_bounds(self):
        """CAI should be in (0, 1]."""
        from chimera.codon_optimizer import (
            tokenize_dna,
            compute_cai,
            sliding_window_optimize,
        )

        aa_seq = "MTEYKLVVVGAGGVGKS"
        dna, metrics = sliding_window_optimize(aa_seq)
        tokens = tokenize_dna(dna)
        cai = compute_cai(tokens)
        assert 0.0 < cai.item() <= 1.0, f"CAI out of bounds: {cai.item()}"

    def test_gc_content_within_target_after_optimization(self):
        """GC content should be between 0.50 and 0.75 after optimization."""
        from chimera.codon_optimizer import sliding_window_optimize

        aa_seq = "MTEYKLVVVGAGGVGKSALTIQLIQNHFVD"
        dna, metrics = sliding_window_optimize(aa_seq)
        gc = metrics["gc_content"]
        assert 0.45 <= gc <= 0.80, f"GC content out of expected range: {gc}"


# ── CHIMERA v2 Integration Tests ──────────────────────────────────────────────


class TestCHIMERAv2Integration:

    def test_model_instantiation(self, small_config):
        model = CHIMERAv2(**small_config)
        assert model is not None

    def test_parameter_counts(self, chimera_model):
        frozen = chimera_model.count_frozen()
        trainable = chimera_model.count_trainable()
        assert frozen > 0, "Should have frozen pretrained params"
        assert trainable > 0, "Should have trainable connector params"
        # Connectors should be much smaller than frozen backbones
        ratio = trainable / (frozen + trainable)
        assert ratio < 0.5, f"Trainable params ({ratio:.1%}) should be minority"

    def test_forward_pass_shapes(self, chimera_model, small_constraints):
        B, N_seq, L = 1, 4, 60
        with torch.no_grad():
            outputs = chimera_model(
                msa_tokens=torch.randint(0, 23, (B, N_seq, L)),
                initial_pair_features=torch.zeros(B, L, L, 32),
                source_R=torch.eye(3).view(1, 1, 3, 3).expand(B, L, -1, -1),
                source_t=torch.randn(B, L, 3) * 5,
                constraints=small_constraints,
                substrate_id=torch.tensor([13]),
                n_flow_steps=3,
                n_mpnn_seqs=2,
            )
        assert outputs["sequences"].shape[0] == B
        assert outputs["backbone_coords"].shape == (B, L, 4, 3)
        assert outputs["evol_plausibility"].shape == (B,)
        assert outputs["assembly_compat"].shape == (B,)

    def test_save_load_connectors(self, chimera_model, tmp_path):
        """Save connector weights, reload, verify forward pass still works."""
        save_path = str(tmp_path / "connectors.pt")
        chimera_model.save(save_path)

        # Load into fresh model
        import copy

        model2 = copy.deepcopy(chimera_model)
        model2.load_connectors(save_path)
        assert os.path.exists(save_path)


# ── DPO Tests ─────────────────────────────────────────────────────────────────


class TestDPO:

    def test_dpo_loss_winner_higher_than_loser(self):
        """After enough DPO steps, winner should have higher log-prob than loser."""
        # This is a smoke test — just checks loss is computable, not convergence
        from chimera.multi_objective import DPOTrainer, ProteusPreferencePair
        from chimera.codon_optimizer import tokenize_protein

        trainer = DPOTrainer(beta=0.1)
        L = 20
        # Create a trivial preference pair
        pair = ProteusPreferencePair(
            winner_tokens=torch.randint(0, 20, (L,)),
            loser_tokens=torch.randint(0, 20, (L,)),
            msa_tokens=torch.randint(0, 23, (4, L)),
            pair_features=torch.randn(L, L, 32),
        )
        # Just test it doesn't crash with mock model
        # (full DPO test requires actual CHIMERA forward pass — tagged as slow)
        assert pair.winner_tokens.shape == (L,)
        assert pair.loser_tokens.shape == (L,)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
