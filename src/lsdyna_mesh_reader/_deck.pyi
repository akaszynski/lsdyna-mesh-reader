from typing import List

import numpy as np
from numpy.typing import NDArray

# shape typing added in 2.1, not adding it here yet
IntArray = NDArray[np.int32]
FloatArray1D = NDArray[np.float64]
FloatArray2D = NDArray[np.float64]

class NodeSection:
    def __init__(self) -> None: ...
    @property
    def nnum(self) -> IntArray: ...
    @property
    def nodes(self) -> FloatArray2D: ...
    @property
    def tc(self) -> IntArray: ...
    @property
    def rc(self) -> IntArray: ...
    def __len__(self) -> int: ...

class ElementSection:
    def __init__(self) -> None: ...
    @property
    def eid(self) -> IntArray: ...
    @property
    def pid(self) -> IntArray: ...
    @property
    def node_ids(self) -> IntArray: ...
    @property
    def node_id_offsets(self) -> IntArray: ...
    def __len__(self) -> int: ...

class ElementShellSection(ElementSection): ...
class ElementSolidSection(ElementSection): ...

class _Deck:
    def __init__(self, fname: str) -> None: ...
    @property
    def node_sections(self) -> List[NodeSection]: ...
    @property
    def element_solid_sections(self) -> List[ElementSolidSection]: ...
    @property
    def element_shell_sections(self) -> List[ElementShellSection]: ...
    def read_line(self) -> int: ...
    def read_element_solid_section(self) -> None: ...
    def read_element_shell_section(self) -> None: ...
    def read_node_section(self) -> None: ...
    def read(self) -> None: ...
