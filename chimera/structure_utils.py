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


def load_backbone_pdb(
    path: str | Path,
    chain_id: str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract N/CA/C/O atoms and construct per-residue frames from a PDB."""
    coords = load_backbone_coords_pdb(path, chain_id=chain_id)
    n, ca, c = coords[0, :, 0], coords[0, :, 1], coords[0, :, 2]
    x = torch.nn.functional.normalize(c - ca, dim=-1)
    y = torch.nn.functional.normalize(n - ca, dim=-1)
    z = torch.nn.functional.normalize(torch.cross(x, y, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)
    frames = torch.stack((x, y, z), dim=-1).unsqueeze(0)
    return frames, ca.unsqueeze(0)


def load_backbone_coords_pdb(
    path: str | Path,
    chain_id: str | None = None,
) -> torch.Tensor:
    """Extract complete N/CA/C/O coordinates as ``(1, L, 4, 3)``.

    Deterministic policy: use the first MODEL, optionally filter a chain,
    prefer blank-altLoc atoms, otherwise choose one residue-wide conformer by
    highest mean occupancy (ties prefer A then lexical order), and sort by
    residue number plus insertion code. Alternate conformers are never mixed.
    """
    residues: dict[tuple[str, int, str], dict] = {}
    in_first_model = True
    model_seen = False
    selected_chain = chain_id
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("MODEL "):
            if model_seen:
                break
            model_seen = True
            in_first_model = True
            continue
        if line.startswith("ENDMDL") and model_seen:
            break
        if not in_first_model:
            continue
        if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
            continue
        atom_chain = line[21].strip()
        if selected_chain is None:
            selected_chain = atom_chain
        if atom_chain != selected_chain:
            continue
        atom = line[12:16].strip()
        if atom not in {"N", "CA", "C", "O"}:
            continue
        try:
            residue_number = int(line[22:26])
            occupancy = float(line[54:60].strip() or 0.0)
            xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
        except ValueError as exc:
            raise ValueError(f"Malformed PDB atom record: {line!r}") from exc
        insertion_code = line[26].strip()
        alt_loc = line[16].strip()
        residue_key = (atom_chain, residue_number, insertion_code)
        residue = residues.setdefault(residue_key, {"atoms": {}, "alt_occupancies": {}})
        residue["atoms"].setdefault(atom, []).append((alt_loc, occupancy, xyz))
        if alt_loc:
            residue["alt_occupancies"].setdefault(alt_loc, []).append(occupancy)

    if not residues:
        raise ValueError(f"No complete N/CA/C/O backbone residues found in {path}")

    selected_residues = []
    for residue_key, residue in residues.items():
        occupancy_by_alt = residue["alt_occupancies"]
        chosen_alt = ""
        if occupancy_by_alt:
            chosen_alt = min(
                occupancy_by_alt,
                key=lambda alt: (
                    -sum(occupancy_by_alt[alt]) / len(occupancy_by_alt[alt]),
                    alt != "A",
                    alt,
                ),
            )
        selected_atoms = {}
        for atom_name, choices in residue["atoms"].items():
            compatible = [choice for choice in choices if choice[0] in {"", chosen_alt}]
            if not compatible:
                continue
            blank = [choice for choice in compatible if choice[0] == ""]
            selected = max(blank or compatible, key=lambda choice: choice[1])
            selected_atoms[atom_name] = selected[2]
        if not {"N", "CA", "C", "O"}.issubset(selected_atoms):
            raise ValueError(
                f"Incomplete N/CA/C/O backbone residue {residue_key} in {path}"
            )
        selected_residues.append((residue_key, selected_atoms))
    selected_residues.sort(key=lambda item: (item[0][0], item[0][1], item[0][2]))

    coords = torch.tensor(
        [[entry[name] for name in ("N", "CA", "C", "O")] for _, entry in selected_residues],
        dtype=torch.float32,
    ).unsqueeze(0)
    return coords
