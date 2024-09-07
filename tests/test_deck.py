from pathlib import Path

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

NODE_SECTION_NODES_EXPECTED = np.array(
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
       2       1       2       3       7       6      18      19      23      22
       3       1       3       4       8       7      19      20      24      23
       4       1       5       6      10       9      21      22      26      25
       5       1       6       7      11      10      22      23      27      26
*END
"""

ELEMENT_SOLID_SECTION_ELEMS = [
    [1, 2, 6, 5, 17, 18, 22, 21],
    [2, 3, 7, 6, 18, 19, 23, 22],
    [3, 4, 8, 7, 19, 20, 24, 23],
    [5, 6, 10, 9, 21, 22, 26, 25],
    [6, 7, 11, 10, 22, 23, 27, 26],
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

    assert np.allclose(node_section.nodes, NODE_SECTION_NODES_EXPECTED)
    assert np.allclose(node_section.nnum, range(1, 6))
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
    assert "ElementSolidSection containing 5 elements" in str(section)

    assert np.allclose(section.eid, range(1, 6))
    assert np.allclose(section.pid, [1] * 5)
    assert np.allclose(section.node_ids, np.array(ELEMENT_SOLID_SECTION_ELEMS).ravel())

    cells, offset, celltypes = section.to_vtk()
    assert np.allclose(cells, section.node_ids)  # expect no change
    assert np.allclose(offset, section.node_id_offsets)  # expect no change
    assert np.allclose(celltypes, [pv.CellType.HEXAHEDRON] * 5)

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
    assert len(node_section.nodes) == 1281
    assert node_section.nnum[-1] == 1344

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
