from pathlib import Path
import numpy as np

from lsdyna_mesh_reader._deck import Deck


NODE_CARD_BLOCK = """*NODE
       1 2.309401035E+00-2.309401035E+00-2.309401035E+00       0       0
       2-2.039600611E+00 2.039600611E+00-2.039600611E+00       0       0
       3-1.769800305E+00-1.769800305E+00 1.769800305E+00       0       0
       4-1.500000000E+00-1.500000000E+00-1.500000000E+00       0       0
       5-2.593648434E+00-1.595611572E+00-2.593648434E+00       0       0
*END
"""

NODE_CARD_NODES_EXPECTED = np.array(
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
