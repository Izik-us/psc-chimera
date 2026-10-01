import pytest

from chimera.structure_utils import load_backbone_coords_pdb


def _atom_line(serial, atom, residue, chain, number, x, alt="", occupancy=1.0, insertion=""):
    element = atom[0]
    return (
        f"ATOM  {serial:5d} {atom:>4s}{alt:1s}{residue:>3s} {chain}"
        f"{number:4d}{insertion:1s}   {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
        f"{occupancy:6.2f}{20.0:6.2f}          {element:>2s}"
    )


def _residue_lines(serial, chain, number, x, insertion="", alt="", occupancy=1.0):
    atoms = []
    for atom, offset in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
        serial += 1
        atoms.append(_atom_line(serial, atom, "ALA", chain, number, x + offset, alt, occupancy, insertion))
    return atoms


def test_pdb_uses_first_model_and_insertion_code_order(tmp_path):
    lines = ["MODEL        1"]
    lines.extend(_residue_lines(0, "A", 10, 10.0, insertion="B"))
    lines.extend(_residue_lines(4, "A", 10, 20.0, insertion="A"))
    lines.append("ENDMDL")
    lines.append("MODEL        2")
    lines.extend(_residue_lines(8, "A", 1, 99.0))
    lines.append("ENDMDL")
    path = tmp_path / "models.pdb"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    coords = load_backbone_coords_pdb(path)

    assert coords.shape == (1, 2, 4, 3)
    assert coords[0, 0, 0, 0].item() == pytest.approx(20.0)
    assert coords[0, 1, 0, 0].item() == pytest.approx(10.0)


def test_pdb_selects_one_residue_conformer_by_occupancy(tmp_path):
    lines = ["MODEL        1"]
    serial = 0
    for atom, offset in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
        serial += 1
        lines.append(_atom_line(serial, atom, "ALA", "A", 1, 50.0 + offset, "A", 0.3))
        serial += 1
        lines.append(_atom_line(serial, atom, "ALA", "A", 1, 5.0 + offset, "B", 0.7))
    lines.append("ENDMDL")
    path = tmp_path / "altloc.pdb"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    coords = load_backbone_coords_pdb(path)

    assert coords.shape == (1, 1, 4, 3)
    assert coords[0, 0, 0, 0].item() == pytest.approx(5.0)


def test_pdb_prefers_blank_altloc_and_supports_chain_filter(tmp_path):
    lines = ["MODEL        1"]
    lines.extend(_residue_lines(0, "A", 1, 10.0, alt="", occupancy=0.1))
    for atom, offset in (("N", 0.0), ("CA", 1.0), ("C", 2.0), ("O", 3.0)):
        lines.append(_atom_line(5 + int(offset), atom, "ALA", "A", 1, 80.0 + offset, "A", 0.9))
    lines.extend(_residue_lines(10, "B", 1, 30.0))
    lines.append("ENDMDL")
    path = tmp_path / "chains.pdb"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    first_chain = load_backbone_coords_pdb(path)
    chain_b = load_backbone_coords_pdb(path, chain_id="B")

    assert first_chain.shape == (1, 1, 4, 3)
    assert first_chain[0, 0, 0, 0].item() == pytest.approx(10.0)
    assert chain_b[0, 0, 0, 0].item() == pytest.approx(30.0)