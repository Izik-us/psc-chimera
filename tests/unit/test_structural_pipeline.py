import pytest

from data.structures import StructuralQCConfig, parse_mmcif_structure


def _write_mmcif(tmp_path, *, omit_second_residue=False, include_ligand=False):
    path = tmp_path / "fixture.cif"
    lines = [
        "data_TEST",
        "_entry.id TEST",
        "loop_",
        "_entity_poly.entity_id",
        "_entity_poly.type",
        "_entity_poly.pdbx_strand_id",
        "_entity_poly.pdbx_seq_one_letter_code",
        "_entity_poly.pdbx_seq_one_letter_code_can",
        "1 'polypeptide(L)' A GAV GAV",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_asym_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    atom_id = 1
    names = {"G": "GLY", "A": "ALA", "V": "VAL"}
    for index, amino_acid in enumerate("GAV", start=1):
        if omit_second_residue and index == 2:
            continue
        x = (index - 1) * 3.8
        atoms = {
            "N": (x, 0.0, 0.0),
            "CA": (x + 1.45, 0.0, 0.0),
            "C": (x + 2.6, 0.0, 0.0),
            "O": (x + 3.0, 1.0, 0.0),
        }
        if amino_acid != "G":
            atoms["CB"] = (x + 1.45, 1.4, 0.0)
        if amino_acid == "V":
            atoms["CG1"] = (x + 1.8, 2.2, 0.1)
            atoms["CG2"] = (x + 1.8, -1.0, 0.1)
        for atom_name, coordinates in atoms.items():
            if index == 3 and atom_name == "CA":
                for alt, occupancy, shift in (("A", 0.5, 0.0), ("B", 0.5, 0.2)):
                    xyz = (coordinates[0] + shift, *coordinates[1:])
                    lines.append(
                        f"ATOM {atom_id} C {atom_name} {alt} {names[amino_acid]} A 1 {index} A "
                        f"{xyz[0]} {xyz[1]} {xyz[2]} {occupancy} 42.0 {index + 10} X 1"
                    )
                    atom_id += 1
                continue
            lines.append(
                f"ATOM {atom_id} C {atom_name} . {names[amino_acid]} A 1 {index} "
                f"{'A' if index == 3 else '.'} {coordinates[0]} {coordinates[1]} {coordinates[2]} "
                f"1.0 42.0 {index + 10} X 1"
            )
            atom_id += 1
    if include_ligand:
        lines.append("HETATM 100 C C1 . ATP L . . ? 1.45 0.0 1.0 1.0 18.0 501 A 1")
        lines.extend([
            "#",
            "_chem_comp.id ATP",
            "_chem_comp.name 'ADENOSINE TRIPHOSPHATE'",
            "_chem_comp.formula 'C10 H16 N5 O13 P3'",
            "#",
        ])
    lines.extend(["#", "_exptl.method 'X-RAY DIFFRACTION'", "_refine.ls_d_res_high 2.1", "#"])
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def test_parser_preserves_entity_chain_numbering_altloc_and_atom_masks(tmp_path):
    structure = parse_mmcif_structure(
        _write_mmcif(tmp_path), config=StructuralQCConfig(minimum_chain_length=1)
    )

    assert structure.structure_id == "TEST"
    assert structure.experimental_method == "X-RAY DIFFRACTION"
    assert structure.resolution_angstrom == pytest.approx(2.1)
    assert len(structure.chains) == 1
    chain = structure.chains[0]
    assert (chain.entity_id, chain.label_asym_id, chain.auth_asym_id) == ("1", "A", "X")
    assert chain.sequence == "GAV"
    assert chain.coordinates.shape == (3, 5, 3)
    assert chain.atom_mask[:, :4].all()
    assert not chain.atom_mask[0, 4]
    assert chain.atom_mask[1:, 4].all()
    assert chain.mappings[2].auth_seq_id == "13"
    assert chain.mappings[2].insertion_code == "A"
    assert chain.alternate_locations[2] == "A"
    assert chain.quality.status == "accepted"


def test_parser_retains_unresolved_polymer_positions_in_sequence_mapping(tmp_path):
    structure = parse_mmcif_structure(
        _write_mmcif(tmp_path, omit_second_residue=True),
        config=StructuralQCConfig(minimum_chain_length=1),
    )
    chain = structure.chains[0]

    assert chain.sequence == "GAV"
    assert chain.residue_mask.tolist() == [True, False, True]
    assert chain.sequence_mask.tolist() == [True, True, True]
    assert len(chain.mappings) == 3
    assert chain.mappings[1].sequence_index == 1
    assert "EXCESSIVE_UNRESOLVED_RESIDUES" in chain.quality.reasons


def test_ligand_contacts_map_to_protein_chain_and_ccd_metadata(tmp_path):
    structure = parse_mmcif_structure(
        _write_mmcif(tmp_path, include_ligand=True),
        config=StructuralQCConfig(minimum_chain_length=1),
    )
    ligand = structure.ligands[0]

    assert ligand.chemical_component_id == "ATP"
    assert ligand.name == "ADENOSINE TRIPHOSPHATE"
    assert ligand.binding_residue_refs
    assert ligand.binding_residue_refs[0]["label_asym_id"] == "A"
    assert ligand.minimum_contact_distance == pytest.approx(1.0)