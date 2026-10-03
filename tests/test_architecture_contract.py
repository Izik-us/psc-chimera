import ast
import subprocess
import sys
from pathlib import Path

import torch

from chimera import CHIMERAv2, AutoregressiveSequencePolicy


def test_package_boundary_has_no_runtime_compat_shim():
    root = Path(__file__).resolve().parents[1]
    assert not (root / "chimera" / "runtime_compat.py").exists()
    init_text = (root / "chimera" / "__init__.py").read_text(encoding="utf-8").lower()
    assert "runtime_compat" not in init_text


def test_canonical_imports_do_not_depend_on_compatibility_modules():
    root = Path(__file__).resolve().parents[1] / "chimera"
    forbidden = {
        "chimera_v1",
        "chimera_v2",
        "evoformer",
        "flow_matching",
        "legacy_flow",
        "legacy_optimization",
        "multi_objective",
        "se3_diffusion",
    }
    for module_name in ("architecture.py", "training.py", "se3_flow.py"):
        tree = ast.parse((root / module_name).read_text(encoding="utf-8"))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
                imported.add(node.module.split(".", 1)[0])
            elif isinstance(node, ast.Import):
                imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        assert not imported.intersection(forbidden), (module_name, imported & forbidden)


def test_canonical_flow_uses_the_extracted_module():
    from chimera.se3_flow import FlowMatchingBackbone as CanonicalFlowMatchingBackbone

    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=32,
        n_flow_blocks=1,
        n_domains=1,
        n_modules=1,
        n_mpnn_seqs=1,
        n_mc_dropout=2,
    )
    assert isinstance(model.flow_model, CanonicalFlowMatchingBackbone)
    assert type(model.flow_model).__module__ == "chimera.se3_flow"
    assert any(
        key.startswith("flow_model.flow_model.velocity_field.")
        for key in model.state_dict()
    )


def test_package_import_does_not_load_legacy_flow_or_external_models(tmp_path):
    root = Path(__file__).resolve().parents[1]
    code = (
        "import sys; "
        "sys.path.insert(0, sys.argv[1]); "
        "import chimera; "
        "assert 'chimera.legacy_flow' not in sys.modules; "
        "assert 'chimera.flow_matching' not in sys.modules; "
        "assert not any(name == 'esm' or name.startswith('esm.') for name in sys.modules); "
        "assert not any(name == 'openfold' or name.startswith('openfold.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_historical_flow_package_export_is_lazy_and_compatible():
    from chimera import SE3FlowMatching

    assert SE3FlowMatching.__module__ == "chimera.legacy_flow"


def test_public_chimera_is_canonical_and_has_causal_policy():
    model = CHIMERAv2(
        d_evo_single=32,
        d_evo_pair=16,
        d_se3=32,
        d_pair_out=32,
        d_mpnn=32,
        n_flow_blocks=1,
        n_domains=1,
        n_modules=1,
        n_mpnn_seqs=1,
        n_mc_dropout=2,
    )
    assert getattr(model, "_is_canonical_composition", False)
    assert isinstance(model.sequence_policy, AutoregressiveSequencePolicy)


def test_autoregressive_policy_has_finite_logprob():
    policy = AutoregressiveSequencePolicy(context_dim=32, vocab_size=20, layers=1, heads=4)
    context = torch.randn(2, 5, 32)
    tokens = torch.randint(0, 20, (2, 5))
    logp = policy.logprob(context, tokens)
    assert logp.shape == (2,)
    assert torch.isfinite(logp).all()
