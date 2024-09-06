from pathlib import Path
import numpy as np

from lsdyna_mesh_reader._deck import Deck


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


def test_node_card(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.k")
    with open(filename, "w") as fid:
        fid.write(NODE_CARD_BLOCK)

    deck = Deck(filename)
    deck.read_line()
    deck.read_node_card()

    assert len(deck.node_cards) == 1
    node_card = deck.node_cards[0]
    assert np.allclose(node_card.nodes, NODE_CARD_NODES_EXPECTED)
    assert np.allclose(node_card.nnum, range(1, 6))


ELEMENT_SECTION = """*ELEMENT_SOLID
       1       1       1       2       6       5      17      18      22      21
       2       1       2       3       7       6      18      19      23      22
       3       1       3       4       8       7      19      20      24      23
       4       1       5       6      10       9      21      22      26      25
       5       1       6       7      11      10      22      23      27      26
*END
"""


def test_node_section(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.k")
    with open(filename, "w") as fid:
        fid.write(NODE_SECTION)

    deck = Deck(filename)
    deck.read_line()
    deck.read_node_section()

    assert len(deck.node_sections) == 1
    node_section = deck.node_sections[0]
    assert np.allclose(node_section.nodes, NODE_SECTION_NODES_EXPECTED)
    assert np.allclose(node_section.nnum, range(1, 6))
    assert np.allclose(node_section.tc, [0, 0, 1, 2, 0])
    assert np.allclose(node_section.rc, [0, 3, 4, 0, 0])


def test_element_section(tmp_path: Path) -> None:
    filename = str(tmp_path / "tmp.k")
    with open(filename, "w") as fid:
        fid.write(NODE_SECTION)

    deck = Deck(filename)
    deck.read_line()
    deck.read_element_section()

    assert len(deck.element_sections) == 1
    element_section = deck.element_sections[0]

    # assert np.allclose(element_section.element_id,
