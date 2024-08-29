from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", np.float32, np.float64)

def write_nblock(
    filename: str,
    max_node_id: int,
    node_id: NDArray[np.int32],
    nodes: NDArray[np.float64],
    angles: NDArray[np.float64],
    sig_digits: int,
    mode: str,
) -> None: ...
def write_eblock(
    filename: str,
    n_elem: int,
    elem_id: NDArray[np.int32],
    etype: NDArray[np.int32],
    mtype: NDArray[np.int32],
    rcon: NDArray[np.int32],
    elem_nnodes: NDArray[np.int32],
    celltypes: NDArray[np.uint8],
    offset: NDArray[np.int64],
    cells: NDArray[np.int64],
    typenum: NDArray[np.int32],
    nodenum: NDArray[np.int32],
    mode: str,
) -> None: ...
def cmblock_items_from_array(array: NDArray[np.int32]) -> NDArray[np.int32]: ...
def reset_midside(
    celltypes: NDArray[np.uint8],
    cells: NDArray[np.int64],
    offset: NDArray[np.int64],
    points: NDArray[T],
) -> None: ...
