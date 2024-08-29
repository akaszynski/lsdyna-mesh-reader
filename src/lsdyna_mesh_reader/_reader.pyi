from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

class Archive:
    elem: NDArray[np.int32]
    elem_off: NDArray[np.int32]
    n_elem: int
    keyopt: Dict[int, List[List[int]]]
    rdat: List[List[float]]
    rnum: List[int]
    elem_type: List[List[int]]

    # Components
    elem_comps: Dict[str, NDArray[np.int32]]
    node_comps: Dict[str, NDArray[np.int32]]

    # node block
    n_nodes: int
    nblock_start: int
    nblock_end: int
    nnum: NDArray[np.int32]
    nodes: NDArray[np.double]
    node_angles: NDArray[np.double]

    def __init__(
        self, fname: str, read_params: bool = False, debug: bool = False, read_eblock: bool = True
    ) -> None: ...
    def get_element_types(self) -> NDArray[np.int32]: ...
    def read_eblock(self) -> None: ...
    def read_et_line(self) -> None: ...
    def read_etblock(self) -> None: ...
    def read_keyopt_line(self) -> None: ...
    def read_nblock(self) -> None: ...
    def read_rlblock(self) -> None: ...
    def read_cmblock(self) -> None: ...
    def read(self) -> None: ...
    def to_vtk(
        self, type_map: NDArray[np.int32]
    ) -> Tuple[NDArray[int], NDArray[np.uint8], NDArray[int]]: ...

def ans_to_vtk(
    elem: NDArray[np.int32],
    elem_off: NDArray[np.int32],
    type_ref: NDArray[np.int32],
    nnum: NDArray[np.int32],
) -> Tuple[NDArray[np.int64], NDArray[np.uint8], NDArray[np.int64]]: ...
