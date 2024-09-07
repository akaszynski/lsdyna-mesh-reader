from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import numpy as np
from numpy.typing import NDArray

from lsdyna_mesh_reader._deck import ElementShellSection, ElementSolidSection, NodeSection, _Deck

if TYPE_CHECKING:
    try:
        from pyvista.core.pointset import UnstructuredGrid
    except ImportError:
        pass


class Deck(_Deck):
    """LS-DYNA deck."""

    def __init__(self, filename: Union[str, Path]) -> None:
        """Initialize the deck object."""
        self._deck = _Deck(str(filename))
        self._deck.read()

    @property
    def element_solid_sections(self) -> List[ElementSolidSection]:
        """Return the element_solid sections."""
        return self._deck.element_solid_sections

    @property
    def element_shell_sections(self) -> List[ElementShellSection]:
        """Return the element_shell sections."""
        return self._deck.element_shell_sections

    @property
    def node_sections(self) -> List[NodeSection]:
        """Return the node sections."""
        return self._deck.node_sections

    def to_grid(self) -> "UnstructuredGrid":
        """Convert the mesh within the deck to a pyvista.UnstructuredGrid."""
        try:
            import pyvista as pv
            from pyvista import ID_TYPE, CellArray
            from pyvista.core.pointset import UnstructuredGrid
            from vtkmodules.util.numpy_support import numpy_to_vtk
        except ImportError:
            pass

        if not self.node_sections:
            raise RuntimeError("Missing node sections. Unable to generate UnstructuredGrid.")

        node_section = self.node_sections[0]

        # map between the node ids and a sequential index
        id_map = np.empty(node_section.nnum[-1] + 1, dtype=ID_TYPE)
        id_map[node_section.nnum] = np.arange(len(node_section), dtype=ID_TYPE)

        element_sections = self.element_shell_sections + self.element_solid_sections

        if not element_sections:
            raise NotImplementedError("Deck missing element sections")

        # add shell sections
        offsets: List[NDArray[np.int32]] = []
        celltypes: List[NDArray[np.uint8]] = []
        cells: List[NDArray[np.int32]] = []
        part_ids = []
        for section in element_sections:
            if offsets:
                # we need to shift by the last value of the last offset
                offsets.append(section.node_id_offsets[1:] + offsets[-1][-1])
            else:
                offsets.append(section.node_id_offsets)

            if isinstance(section, ElementSolidSection):
                celltype = pv.CellType.HEXAHEDRON
            elif isinstance(section, ElementShellSection):
                celltype = pv.CellType.QUAD
            else:
                raise NotImplementedError(f"Unsupported section {type(section)}")

            shell_celltypes = np.full(len(section), celltype, dtype=np.uint8)
            celltypes.append(shell_celltypes)

            cells.append(id_map[section.node_ids])
            part_ids.append(section.pid)

        offsets_arr = np.hstack(offsets, dtype=ID_TYPE)
        cells_arr = np.hstack(cells, dtype=ID_TYPE)
        celltypes_arr = np.hstack(celltypes, dtype=np.uint8)

        grid = UnstructuredGrid()
        grid.points = pv.pyvista_ndarray(node_section.nodes)

        vtk_cells = CellArray.from_arrays(offsets_arr, cells_arr, deep=False)
        vtk_cell_type = numpy_to_vtk(celltypes_arr, deep=False)
        grid.SetCells(vtk_cell_type, vtk_cells)

        # add part and node ids
        grid.cell_data["Part ID"] = np.hstack(part_ids)
        grid.point_data["Node ID"] = node_section.nnum

        return grid

    def __repr__(self) -> str:
        lines = ["LSDYNA Deck with:"]
        lines.append(f"  Node sections:              {len(self.node_sections)}")
        lines.append(f"  Element Solid sections:     {len(self.element_solid_sections)}")
        lines.append(f"  Element Shell sections:     {len(self.element_shell_sections)}")

        return "\n".join(lines)
