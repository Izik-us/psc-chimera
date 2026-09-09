import torch

from chimera import CHIMERAv2
from chimera.bayesian import BayesianUncertaintyEstimator
from chimera.canonical_components import FlowMatchingBackbone, MultiScaleNRPSDesigner
from chimera.pareto_pcgrad import MergeReadyParetoMultiObjectiveHead


def test_public_model_uses_canonical_components():
    model = CHIMERAv2(
        evoformer_layers=1,
        flow_blocks=1,
        mpnn_layers=1,
        n_mpnn_seqs=1,
    )
    assert isinstance(model.flow_model, FlowMatchingBackbone)
    assert isinstance(model.multi_scale_designer, MultiScaleNRPSDesigner)
    assert isinstance(model.pareto_head, MergeReadyParetoMultiObjectiveHead)
    assert isinstance(model.uncertainty_estimator, BayesianUncertaintyEstimator)


def test_public_model_has_single_canonical_sequence_policy():
    model = CHIMERAv2(evoformer_layers=1, flow_blocks=1, mpnn_layers=1, n_mpnn_seqs=1)
    params = list(model.sequence_policy.parameters())
    assert params
    assert all(torch.isfinite(p).all() for p in params)
