"""Small, dependency-light structure and MSA input adapters."""

from pathlib import Path
from typing import List, Tuple
import torch
from .flow_matching import so3_exp

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_TOKEN = {aa: i for i, aa in enumerate(AA)}
AA_TO_TOKEN.update({"X": 20, "-": 21, "B": 20, "Z": 20, "J": 20, "U": 20, "O": 20})


def load_msa(path: str | Path) -> torch.Tensor:
    """Load aligned FASTA/A3M records as ``(1, N, L)`` integer tokens."""
    sequences: List[str] = []
    current = ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                sequences.append(current)
                current = ""
        else:
            current += "".join(ch for ch in line if not ch.islower())
    if current:
        sequences.append(current)
    if not sequences or len({len(seq) for seq in sequences}) != 1:
        raise ValueError("MSA must contain at least one aligned sequence of equal length")
    tokens = [[AA_TO_TOKEN.get(ch.upper(), 20) for ch in seq] for seq in sequences]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)


def load_backbone_pdb(path: str | Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract N/CA/C/O atoms and construct per-residue frames from a PDB."""
    coords = load_backbone_coords_pdb(path)
    n, ca, c = coords[0, :, 0], coords[0, :, 1], coords[0, :, 2]
    x = torch.nn.functional.normalize(c - ca, dim=-1)
    y = torch.nn.functional.normalize(n - ca, dim=-1)
    z = torch.nn.functional.normalize(torch.cross(x, y, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    frames = torch.stack((x, y, z), dim=-1).unsqueeze(0)
    return frames, ca.unsqueeze(0)


def load_backbone_coords_pdb(path: str | Path) -> torch.Tensor:
    """Extract complete N/CA/C/O coordinates as ``(1, L, 4, 3)``."""
    atoms = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        atom = line[12:16].strip()
        if atom not in {"N", "CA", "C", "O"}:
            continue
        key = (line[21].strip(), line[22:26].strip(), line[26].strip())
        atoms.setdefault(key, {})[atom] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
    residues = [entry for entry in atoms.values() if {"N", "CA", "C"}.issubset(entry)]
    if not residues:
        raise ValueError(f"No complete backbone residues found in {path}")
    coords = torch.tensor(
        [[entry[name] for name in ("N", "CA", "C", "O")] for entry in residues],
        dtype=torch.float32,
    ).unsqueeze(0)
    if coords.shape[2] != 4:
        raise ValueError("Every residue must contain N, CA, C, and O atoms")
    return coords
