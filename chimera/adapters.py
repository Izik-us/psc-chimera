"""Stable boundaries for optional native upstream model integrations.

The adapters in this module deliberately do not fall back to CHIMERA's local
approximation models. Native upstream models must be installed and wired to
these interfaces before they can be used in a production design run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Mapping

import torch

from .backends import BackendUnavailable


class BackboneEncoder(ABC):
    """Encode an MSA and pair features into single and pair representations."""

    @abstractmethod
    def forward(
        self, msa_tokens: torch.Tensor, pair_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class StructureGenerator(ABC):
    """Generate a backbone from source frames and conditioning features."""

    @abstractmethod
    def sample(
        self,
        source_R: torch.Tensor,
        source_t: torch.Tensor,
        conditioning: Any,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SequenceDesigner(ABC):
    """Sample sequences and per-residue log probabilities from a backbone."""

    @abstractmethod
    def design(
        self,
        backbone_coords: torch.Tensor,
        fixed_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


def _require_file(path: str | Path, name: str) -> Path:
    checkpoint = Path(path)
    if not checkpoint.is_file():
        raise BackendUnavailable(f"{name} checkpoint not found: {checkpoint}")
    return checkpoint


class OpenFoldAdapter(BackboneEncoder):
    """Boundary for a native OpenFold Evoformer implementation."""

    def __init__(self, checkpoint: str | Path, **_: Any) -> None:
        self.checkpoint = _require_file(checkpoint, "OpenFold")
        raise BackendUnavailable(
            "OpenFoldAdapter is not wired to an OpenFold model in this prototype; "
            "install OpenFold and implement its batch/trunk conversion first"
        )

    def forward(
        self, msa_tokens: torch.Tensor, pair_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("OpenFoldAdapter was not initialized")


class OpenFoldCLIAdapter(BackboneEncoder):
    """Run an OpenFold Evoformer runner in its native environment.

    The runner receives a Torch input file containing ``msa_tokens`` and
    ``pair_features``. It must write a Torch output file containing
    ``single_repr`` and ``pair_repr``. This keeps OpenFold's environment out
    of the CHIMERA process while making the representation contract explicit.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        runner_script: str | Path,
        python_executable: str = "python",
        timeout_seconds: int = 3600,
    ) -> None:
        self.checkpoint = _require_file(checkpoint, "OpenFold")
        self.runner_script = _require_file(runner_script, "OpenFold runner script")
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds

    def forward(
        self, msa_tokens: torch.Tensor, pair_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if msa_tokens.ndim != 3:
            raise ValueError("msa_tokens must have shape (B, N_seq, L)")
        if pair_features.ndim != 4:
            raise ValueError("pair_features must have shape (B, L, L, C)")
        if msa_tokens.shape[0] != pair_features.shape[0] or msa_tokens.shape[2] != pair_features.shape[1] or pair_features.shape[1] != pair_features.shape[2]:
            raise ValueError("MSA and pair feature batch/length dimensions do not match")

        with tempfile.TemporaryDirectory(prefix="chimera-openfold-") as workdir:
            work = Path(workdir)
            input_path = work / "input.pt"
            output_path = work / "output.pt"
            torch.save(
                {"msa_tokens": msa_tokens.cpu(), "pair_features": pair_features.cpu()},
                input_path,
            )
            command = [
                self.python_executable,
                str(self.runner_script),
                "--checkpoint",
                str(self.checkpoint),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ]
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip()
                raise BackendUnavailable(
                    f"OpenFold runner failed with exit code {completed.returncode}: {detail}"
                )
            if not output_path.is_file():
                raise BackendUnavailable("OpenFold runner did not produce its output file")
            result = torch.load(output_path, map_location="cpu")

        if not isinstance(result, Mapping) or "single_repr" not in result or "pair_repr" not in result:
            raise BackendUnavailable(
                "OpenFold runner output must contain single_repr and pair_repr"
            )
        single_repr = result["single_repr"]
        pair_repr = result["pair_repr"]
        if single_repr.ndim != 3 or pair_repr.ndim != 4:
            raise BackendUnavailable("OpenFold representations have invalid ranks")
        if single_repr.shape[:2] != msa_tokens.shape[::2] or pair_repr.shape[:3] != (
            msa_tokens.shape[0], msa_tokens.shape[2], msa_tokens.shape[2]
        ):
            raise BackendUnavailable("OpenFold representations have invalid batch/length dimensions")
        return single_repr, pair_repr


class RFdiffusionAdapter(StructureGenerator):
    """Boundary for native RFdiffusion inference."""

    def __init__(self, checkpoint: str | Path, **_: Any) -> None:
        self.checkpoint = _require_file(checkpoint, "RFdiffusion")
        raise BackendUnavailable(
            "RFdiffusionAdapter is not wired to RFdiffusion inference in this "
            "prototype; use RFdiffusion's native API instead of SE3FlowMatching"
        )

    def sample(
        self,
        source_R: torch.Tensor,
        source_t: torch.Tensor,
        conditioning: Any,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise RuntimeError("RFdiffusionAdapter was not initialized")


class RFdiffusionCLIAdapter(StructureGenerator):
    """Run native RFdiffusion through its upstream inference script.

    This adapter is useful when RFdiffusion lives in WSL or another isolated
    environment. ``conditioning`` must contain ``input_pdb``, ``contigs``,
    and ``output_prefix`` paths understood by the native CLI.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        inference_script: str | Path,
        python_executable: str = "python",
        timeout_seconds: int = 3600,
    ) -> None:
        self.checkpoint = _require_file(checkpoint, "RFdiffusion")
        self.inference_script = _require_file(inference_script, "RFdiffusion inference script")
        self.python_executable = python_executable
        self.timeout_seconds = timeout_seconds

    def sample(
        self,
        source_R: torch.Tensor,
        source_t: torch.Tensor,
        conditioning: Any,
        steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del source_R, source_t, steps
        if not isinstance(conditioning, Mapping):
            raise ValueError(
                "RFdiffusionCLIAdapter conditioning must be a mapping with "
                "input_pdb, contigs, and output_prefix"
            )
        required = ("input_pdb", "contigs", "output_prefix")
        missing = [key for key in required if key not in conditioning]
        if missing:
            raise ValueError(f"RFdiffusion conditioning missing: {', '.join(missing)}")

        input_pdb = _require_file(conditioning["input_pdb"], "RFdiffusion input PDB")
        output_prefix = Path(conditioning["output_prefix"])
        output_prefix.parent.mkdir(parents=True, exist_ok=True)
        command = [
            self.python_executable,
            str(self.inference_script),
            f"inference.ckpt_override_path={self.checkpoint}",
            f"inference.input_pdb={input_pdb}",
            f"inference.output_prefix={output_prefix}",
            f"inference.num_designs={int(conditioning.get('num_designs', 1))}",
            f"contigmap.contigs={conditioning['contigs']}",
        ]
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise BackendUnavailable(
                f"RFdiffusion inference failed with exit code {completed.returncode}: {detail}"
            )

        generated_pdb = Path(f"{output_prefix}_0.pdb")
        if not generated_pdb.is_file():
            raise BackendUnavailable(
                f"RFdiffusion completed without producing {generated_pdb}"
            )
        from .structure_utils import load_backbone_pdb

        return load_backbone_pdb(generated_pdb)


class ProteinMPNNAdapter(SequenceDesigner):
    """Boundary for native ProteinMPNN sampling."""

    def __init__(
        self,
        checkpoint: str | Path,
        proteinmpnn_repo: str | Path | None = None,
        device: str | torch.device = "cpu",
        temperature: float = 0.1,
    ) -> None:
        self.checkpoint = _require_file(checkpoint, "ProteinMPNN")
        repo = Path(proteinmpnn_repo) if proteinmpnn_repo else self.checkpoint.parent.parent
        if not (repo / "protein_mpnn_utils.py").is_file():
            raise BackendUnavailable(
                f"ProteinMPNN source not found under {repo}; provide proteinmpnn_repo"
            )
        sys.path.insert(0, str(repo))
        try:
            from protein_mpnn_utils import ProteinMPNN
        except ImportError as exc:
            raise BackendUnavailable(
                "ProteinMPNN dependencies are not installed in the active environment"
            ) from exc

        payload = torch.load(self.checkpoint, map_location="cpu")
        if not isinstance(payload, dict) or "model_state_dict" not in payload:
            raise BackendUnavailable(
                "ProteinMPNN checkpoint is missing the native model_state_dict key"
            )
        self.device = torch.device(device)
        self.temperature = temperature
        self.model = ProteinMPNN(
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0,
            k_neighbors=int(payload.get("num_edges", 48)),
        )
        try:
            self.model.load_state_dict(payload["model_state_dict"], strict=True)
        except RuntimeError as exc:
            raise BackendUnavailable(
                "ProteinMPNN checkpoint does not match the native adapter architecture"
            ) from exc
        self.model.to(self.device).eval()

    def design(
        self,
        backbone_coords: torch.Tensor,
        fixed_positions: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if backbone_coords.ndim != 4 or backbone_coords.shape[-2:] != (4, 3):
            raise ValueError("backbone_coords must have shape (B, L, 4, 3)")
        batch_size, length = backbone_coords.shape[:2]
        coords = backbone_coords.to(self.device, dtype=torch.float32)
        mask = torch.isfinite(coords).all(dim=(-1, -2)).float()
        coords = torch.nan_to_num(coords)
        if fixed_positions is None:
            design_mask = mask.clone()
        else:
            if fixed_positions.shape != (batch_size, length):
                raise ValueError("fixed_positions must have shape (B, L)")
            design_mask = mask * (~fixed_positions.to(self.device).bool()).float()

        zeros = torch.zeros((batch_size, length), device=self.device)
        residue_idx = torch.arange(length, device=self.device).unsqueeze(0).expand(batch_size, -1)
        chain_encoding = torch.zeros_like(residue_idx)
        random_order = torch.randn((batch_size, length), device=self.device)
        sequence_seed = torch.zeros((batch_size, length), dtype=torch.long, device=self.device)
        bias_by_res = torch.zeros((batch_size, length, 21), device=self.device)
        with torch.no_grad():
            sampled = self.model.sample(
                X=coords,
                randn=random_order,
                S_true=sequence_seed,
                chain_mask=design_mask,
                chain_encoding_all=chain_encoding,
                residue_idx=residue_idx,
                mask=mask,
                temperature=self.temperature,
                omit_AAs_np=[0.0] * 21,
                bias_AAs_np=[0.0] * 21,
                chain_M_pos=torch.ones_like(mask),
                bias_by_res=bias_by_res,
            )
        sequences = sampled["S"]
        log_probs = sampled["probs"].clamp_min(torch.finfo(torch.float32).tiny).log()
        return sequences, log_probs
