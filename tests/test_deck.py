"""Test lsdyna_mesh_reader deck reader."""

from typing import List
from pathlib import Path
import os

import pytest
import numpy as np
import pyvista as pv

import lsdyna_mesh_reader
from lsdyna_mesh_reader._deck import _Deck
from lsdyna_mesh_reader import examples

NODE_SECTION = """*NODE
       1 2.309401035E+00-2.309401035E+00-2.309401035E+00       0       0
       2-2.039600611E+00 2.039600611E+00-2.039600611E+00       0       3
       3-1.769800305E+00-1.769800305E+00 1.769800305E+00       1       4
       4-1.500000000E+00-1.500000000E+00-1.500000000E+00       2       0
       5-2.593648434E+00-1.595611572E+00-2.593648434E+00       0       0
*END
"""

NODE_SECTION_COORD_EXPECTED = np.array(
    [
        [2.309401035e00, -2.309401035e00, -2.309401035e00],
        [-2.039600611e00, 2.039600611e00, -2.039600611e00],
        [-1.769800305e00, -1.769800305e00, 1.769800305e00],
        [-1.500000000e00, -1.500000000e00, -1.500000000e00],
        [-2.593648434e00, -1.595611572e00, -2.593648434e00],
    ]
)

ELEMENT_SOLID_SECTION = """*ELEMENT_SOLID
       1       1       1       2       6       5      17      18      22      21
       2       1       5       6      10       9       9       9       9       9
       3       1       6       7      11      10      22      23      23      23
*END
"""

ELEMENT_SOLID_SECTION_ELEMS = [
    [1, 2, 6, 5, 17, 18, 22, 21],
    [5, 6, 10, 9, 9, 9, 9, 9],
    [6, 7, 11, 10, 22, 23, 23, 23],
]


ELEMENT_SHELL_SECTION = """*ELEMENT_SHELL
       1       2     377     388     389     378
       2       2     378     389     390     379
       3       2     379     390     391     380
       4       2     380     391     392     381
       5       2     381     392     393     393
*END
"""

ELEMENT_SHELL_SECTION_ELEMS = [
    [377, 388, 389, 378],
    [378, 389, 390, 379],
    [379, 390, 391, 380],
    [380, 391, 392, 381],
    [381, 392, 393, 393],
]


def get_example_files() -> List[str]:
    file_paths: List[str] = []
    for attr in dir(examples):
        eval_attr = getattr(examples, attr)
        if isinstance(eval_attr, str) and os.path.isfile(eval_attr):
            if eval_attr.endswith(".key") or eval_attr.endswith(".k"):
                file_paths.append(eval_attr)

    return file_paths


def test_node_section(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.k")
    with open(filename, "w") as fid:
        fid.write(NODE_SECTION)

    deck = _Deck(filename)
    deck.read_line()
    deck.read_node_section()

    assert len(deck.node_sections) == 1
    node_section = deck.node_sections[0]

    assert "NodeSection containing 5 nodes" in str(node_section)

    assert np.allclose(node_section.coordinates, NODE_SECTION_COORD_EXPECTED)
    assert np.allclose(node_section.nid, range(1, 6))
    assert np.allclose(node_section.tc, [0, 0, 1, 2, 0])
    assert np.allclose(node_section.rc, [0, 3, 4, 0, 0])


def test_element_solid_section(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.k")
    with open(filename, "w") as fid:
        fid.write(ELEMENT_SOLID_SECTION)

    deck = _Deck(filename)
    deck.read_line()
    deck.read_element_solid_section()

    assert len(deck.element_solid_sections) == 1
    section = deck.element_solid_sections[0]
    assert "ElementSolidSection containing 3 elements" in str(section)

    assert np.allclose(section.eid, range(1, 4))
    assert np.allclose(section.pid, [1] * 3)
    assert np.allclose(section.node_ids, np.array(ELEMENT_SOLID_SECTION_ELEMS).ravel())

    cells, offset, celltypes = section.to_vtk()
    expected_cells = np.hstack(
        [
            ELEMENT_SOLID_SECTION_ELEMS[0],  # hex
            ELEMENT_SOLID_SECTION_ELEMS[1][:4],  # tet
            [6, 7, 22, 10, 11, 23],  # remapped to vtk style wedge
        ]
    )
    assert np.allclose(cells, expected_cells)
    assert np.allclose(offset, [0, 8, 12, 18])
    assert np.allclose(celltypes, [pv.CellType.HEXAHEDRON, pv.CellType.TETRA, pv.CellType.WEDGE])

    offsets = np.cumsum([0] + [len(element) for element in ELEMENT_SOLID_SECTION_ELEMS])
    assert np.allclose(section.node_id_offsets, offsets)


def test_element_shell_section(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.k")
    with open(filename, "w") as fid:
        fid.write(ELEMENT_SHELL_SECTION)

    deck = _Deck(filename)
    deck.read_line()
    deck.read_element_shell_section()

    assert len(deck.element_shell_sections) == 1
    section = deck.element_shell_sections[0]

    assert "ElementShellSection containing 5 elements" in str(section)

    assert np.allclose(section.eid, range(1, 6))
    assert np.allclose(section.pid, [2] * 5)
    assert np.allclose(section.node_ids, np.array(ELEMENT_SHELL_SECTION_ELEMS).ravel())

    cells, offset, celltypes = section.to_vtk()
    assert np.allclose(cells, section.node_ids[:-1])  # last excluded as it's a triangle
    offsets_expected = section.node_id_offsets.copy()
    offsets_expected[-1] -= 1
    assert np.allclose(offset, offsets_expected)
    assert np.allclose(celltypes, [pv.CellType.QUAD] * 4 + [pv.CellType.TRIANGLE])

    offsets = np.cumsum([0] + [len(element) for element in ELEMENT_SHELL_SECTION_ELEMS])
    assert np.allclose(section.node_id_offsets, offsets)


def test_read_birdball() -> None:
    deck = lsdyna_mesh_reader.Deck(examples.birdball)
    assert "  Node sections:              1" in str(deck)
    assert "  Element Solid sections:     1" in str(deck)

    assert len(deck.node_sections) == 1
    node_section = deck.node_sections[0]
    assert len(node_section) == 1281
    assert len(node_section.coordinates) == 1281
    assert node_section.nid[-1] == 1344

    assert len(deck.element_solid_sections) == 1
    element_solid_section = deck.element_solid_sections[0]
    assert len(element_solid_section) == 816
    assert len(element_solid_section.eid) == 816
    assert len(element_solid_section.pid) == 816
    assert len(element_solid_section.node_id_offsets) == 817
    assert len(element_solid_section.node_ids) == 816 * 8

    assert len(deck.element_shell_sections) == 1
    element_shell_section = deck.element_shell_sections[0]
    assert len(element_shell_section) == 100
    assert len(element_shell_section.eid) == 100
    assert len(element_shell_section.pid) == 100
    assert len(element_shell_section.node_id_offsets) == 101
    assert len(element_shell_section.node_ids) == 100 * 4


@pytest.mark.parametrize("file_path", get_example_files())
def test_examples(file_path: str) -> None:
    assert os.path.isfile(file_path)
    deck = lsdyna_mesh_reader.Deck(file_path)
    grid = deck.to_grid()
    grid._check_for_consistency()


def test_element_tshell() -> None:
    """Verify we can read in ELEMENT_TSHELL."""
    deck = lsdyna_mesh_reader.Deck(examples.simple_plate)
    assert len(deck.element_solid_sections) == 1
    assert len(deck.element_shell_sections) == 0

    # node section has partial constraints, ensure that it's been populated correctly
    node_section = deck.node_sections[0]
    assert len(node_section) == 324

    grid = deck.to_grid()


def test_overwrite_node_section(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.k")
    with open(filename, "w") as fid:
        fid.write(NODE_SECTION)

    deck = lsdyna_mesh_reader.Deck(filename)

    new_filename = str(tmp_path / "tmp_overwrite.k")
    node_section = deck.node_sections[0]
    new_nodes = node_section.coordinates + np.random.random(node_section.coordinates.shape)
    deck.overwrite_node_section(new_filename, new_nodes)

    deck_new = lsdyna_mesh_reader.Deck(new_filename)
    assert np.allclose(deck_new.node_sections[0].coordinates, new_nodes)
    assert np.allclose(deck_new.node_sections[0].nid, deck.node_sections[0].nid)
