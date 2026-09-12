from pathlib import Path

import pytest
import torch
import torch.nn as nn

from chimera.backends import (
    BackendSpec,
    BackendUnavailable,
    freeze_eval,
    load_strict_state,
)


def test_backend_spec_expands_checkpoint_path(tmp_path):
    spec = BackendSpec("demo", "demo.module", Path("~") / tmp_path.name)
    assert spec.normalized_checkpoint().is_absolute()


def test_missing_checkpoint_fails_closed(tmp_path):
    model = nn.Linear(3, 2)
    with pytest.raises(BackendUnavailable, match="checkpoint not found"):
        load_strict_state(model, tmp_path / "missing.pt", "demo")


def test_strict_loader_rejects_wrong_schema(tmp_path):
    model = nn.Linear(3, 2)
    path = tmp_path / "bad.pt"
    torch.save({"state_dict": {"wrong.weight": torch.zeros(2, 3)}}, path)
    with pytest.raises(BackendUnavailable, match="does not match"):
        load_strict_state(model, path, "demo")


def test_freeze_eval_is_deterministic_boundary():
    model = nn.Sequential(nn.Linear(3, 3), nn.Dropout(0.5))
    freeze_eval(model)
    assert model.training is False
    assert all(not p.requires_grad for p in model.parameters())
