"""External folder boundary (ESMFold / OpenFold / any CLI that writes a PDB).

No upstream Python API is assumed: the caller supplies a command template, so
this works with whatever runner is installed. Fails closed (``ConfigurationError``)
when the command or its output is unavailable; never fabricates a structure.
"""

from __future__ import annotations

import hashlib
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import Sequence

import torch

from chimera.errors import ConfigurationError, InferenceError
from .frames import backbone_to_frames, frames_to_backbone
from .types import FoldingBackend, FoldPrediction, detokenize


def parse_backbone_pdb(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``coords (L,4,3)`` (N/CA/C/O) and per-residue pLDDT (CA B-factor, scaled to 0-100)."""
    residues: dict[tuple, dict] = {}
    order: list[tuple] = []
    for line in Path(path).read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        name = line[12:16].strip()
        if name not in ("N", "CA", "C", "O"):
            continue
        key = (line[21], line[22:27])
        if key not in residues:
            residues[key] = {}
            order.append(key)
        residues[key][name] = (float(line[30:38]), float(line[38:46]), float(line[46:54]), float(line[60:66]))
    if not order:
        raise InferenceError("no backbone ATOM records found in folder output")
    coords = torch.full((len(order), 4, 3), float("nan"))
    plddt = torch.zeros(len(order))
    for i, key in enumerate(order):
        for j, nm in enumerate(("N", "CA", "C", "O")):
            if nm in residues[key]:
                coords[i, j] = torch.tensor(residues[key][nm][:3])
        if "CA" in residues[key]:
            plddt[i] = residues[key]["CA"][3]
    if not torch.isfinite(coords[:, :3]).all():
        raise InferenceError("folder output is missing N/CA/C atoms for some residues")
    if not torch.isfinite(coords[:, 3]).all():  # O absent: place with ideal geometry
        R, t = backbone_to_frames(coords[None, :, :3])
        ideal = frames_to_backbone(R, t)[0]
        coords[:, 3] = torch.where(torch.isfinite(coords[:, 3]), coords[:, 3], ideal[:, 3])
    if float(plddt.max()) <= 1.0 + 1e-6:  # ESMFold writes 0-1
        plddt = plddt * 100.0
    return coords, plddt


class ExternalFoldingBackend(FoldingBackend):
    """Run ``command`` once per sequence. Placeholders: ``{fasta}`` ``{out_pdb}`` ``{seed}``."""

    independence_class = "external_pretrained"

    def __init__(
        self,
        command: Sequence[str],
        *,
        name: str,
        checkpoint_path: str | Path | None = None,
        timeout_s: float = 3600.0,
        deterministic: bool = False,
    ):
        if not command:
            raise ConfigurationError("external folding command is empty")
        self.command = list(command)
        self.name = name
        self.timeout_s = timeout_s
        self.deterministic = deterministic
        self.checkpoint_sha256 = None
        if checkpoint_path is not None:
            p = Path(checkpoint_path)
            if not p.is_file():
                raise ConfigurationError(
                    f"{name} checkpoint missing: {p}", corrective_action="download and verify the checkpoint"
                )
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
            self.checkpoint_sha256 = h.hexdigest()

    def predict(self, tokens, mask=None, *, seed: int = 0, msa_tokens=None) -> FoldPrediction:
        seqs = detokenize(tokens)
        coords, plddt = [], []
        with tempfile.TemporaryDirectory() as tmp:
            for i, seq in enumerate(seqs):
                fasta, pdb = Path(tmp) / f"s{i}.fasta", Path(tmp) / f"s{i}.pdb"
                fasta.write_text(f">s{i}\n{seq}\n")
                cmd = [c.format(fasta=fasta, out_pdb=pdb, seed=seed) for c in self.command]
                try:
                    subprocess.run(cmd, check=True, capture_output=True, timeout=self.timeout_s)
                except (OSError, subprocess.SubprocessError) as exc:
                    raise ConfigurationError(
                        f"{self.name} runner failed: {shlex.join(cmd)} ({exc})",
                        corrective_action="install/verify the folding runner; no structure was fabricated",
                    ) from exc
                c, p = parse_backbone_pdb(pdb)
                if c.shape[0] != len(seq):
                    raise InferenceError(f"{self.name} returned {c.shape[0]} residues for a length-{len(seq)} sequence")
                coords.append(c), plddt.append(p)
        L = tokens.shape[1]
        C = torch.zeros(len(seqs), L, 4, 3)
        P = torch.zeros(len(seqs), L)
        for i, (c, p) in enumerate(zip(coords, plddt)):
            C[i, : c.shape[0]], P[i, : p.shape[0]] = c, p
        m = mask if mask is not None else torch.ones(len(seqs), L, dtype=torch.bool)
        R, t = backbone_to_frames(C)
        return FoldPrediction(
            tokens=tokens, mask=m, coords=C, plddt=P, frames=(R, t), backend=self.name,
            independence_class=self.independence_class, checkpoint_sha256=self.checkpoint_sha256, seed=seed,
            provenance={"command": self.command[0], "deterministic": self.deterministic},
        )
