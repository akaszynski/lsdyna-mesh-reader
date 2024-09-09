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


class Deck:
    r"""LS-DYNA deck.

    Parameters
    ----------
    filename : str | pathlib.Path
        Path to the keyword file (`*.k`, `*.key`, `*.dyn`).

    Examples
    --------
    >>> import lsdyna_mesh_reader
    >>> from lsdyna_mesh_reader import examples
    >>> deck = lsdyna_mesh_reader.Deck(examples.birdball)
    LSDYNA Deck with:
      Node sections:              1
      Element Solid sections:     1
      Element Shell sections:     1

    """

    def __init__(self, filename: Union[str, Path]) -> None:
        """Initialize the deck object."""
        self._deck = _Deck(str(filename))
        self._deck.read()

    @property
    def element_solid_sections(self) -> List[ElementSolidSection]:
        """Return the element_solid sections.

        Returns
        -------
        List[ElementSolidSection]

        Examples
        --------
        Load an example keyword deck and output the element IDs.

        >>> import lsdyna_mesh_reader
        >>> from lsdyna_mesh_reader import examples
        >>> deck = lsdyna_mesh_reader.Deck(examples.birdball)
        >>> section = deck.element_solid_sections[0]
        >>> section.eid
        array([1, 2, 3, ..., 814, 815, 816], dtype=int32)

        Output a list of individual elements.

        >>> import numpy as np
        >>> elements = np.split(section.node_ids, section.node_id_offsets[1:-1])
        >>> elements[:5]  # show the first 5
        [array([ 1,  2,  6,  5, 17, 18, 22, 21], dtype=int32),
         array([ 2,  3,  7,  6, 18, 19, 23, 22], dtype=int32),
         array([ 3,  4,  8,  7, 19, 20, 24, 23], dtype=int32),
         array([ 5,  6, 10,  9, 21, 22, 26, 25], dtype=int32),
         array([ 6,  7, 11, 10, 22, 23, 27, 26], dtype=int32)]

        """
        return self._deck.element_solid_sections

    @property
    def element_shell_sections(self) -> List[ElementShellSection]:
        """Return the element_shell sections.

        Returns
        -------
        List[ElementShellSection]

        Examples
        --------
        >>> import lsdyna_mesh_reader
        >>> from lsdyna_mesh_reader import examples
        >>> deck = lsdyna_mesh_reader.Deck(examples.birdball)
        >>> section = deck.element_shell_sections[0]
        >>> section.eid
        array([  1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,
                27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,
                40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
                53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,
                66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,
                79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,
                92,  93,  94,  95,  96,  97,  98,  99, 100], dtype=int32)

        Output a list of individual elements.

        >>> import numpy as np
        >>> elements = np.split(section.node_ids, section.node_id_offsets[1:-1])
        >>> elements[:5]  # output the first 5
        [array([377, 388, 389, 378], dtype=int32),
         array([378, 389, 390, 379], dtype=int32),
         array([379, 390, 391, 380], dtype=int32),
         array([380, 391, 392, 381], dtype=int32),
         array([381, 392, 393, 382], dtype=int32)]

        """
        return self._deck.element_shell_sections

    @property
    def node_sections(self) -> List[NodeSection]:
        """Return the node sections.

        Returns
        -------
        List[NodeSection]

        Examples
        --------
        Load an example keyword deck and output the node IDs of the first node
        section.

        >>> import lsdyna_mesh_reader
        >>> from lsdyna_mesh_reader import examples
        >>> deck = lsdyna_mesh_reader.Deck(examples.birdball)
        >>> node_section = deck.node_sections[0]
        >>> node_section.nid
        array([   1,    2,    3, ..., 1342, 1343, 1344], dtype=int32)

        Output the node coordinates.

        >>> node_section.coordinates
        array([[ -2.30940104,  -2.30940104,  -2.30940104],
               [ -2.03960061,  -2.03960061,  -2.03960061],
               [ -1.76980031,  -1.76980031,  -1.76980031],
               ...,
               [ -4.        , -10.        ,   0.        ],
               [ -2.        , -10.        ,   0.        ],
               [  0.        , -10.        ,   0.        ]])

        """
        return self._deck.node_sections

    def to_grid(self) -> "UnstructuredGrid":
        """Convert the mesh within the deck to a pyvista.UnstructuredGrid.

        Returns
        -------
        pyvista.UnstructuredGrid

        Notes
        -----
        This requires ``pyvista`` to be installed.

        Examples
        --------
        Load an example keyword deck and convert it to an unstructured grid.

        >>> import lsdyna_mesh_reader
        >>> from lsdyna_mesh_reader import examples
        >>> deck = lsdyna_mesh_reader.Deck(examples.birdball)
        >>> ugrid = deck.to_grid()
        UnstructuredGrid (0x70d5caaed1e0)
          N Cells:    916
          N Points:   1281
          X Bounds:   -2.000e+01, 2.220e-15
          Y Bounds:   -1.000e+01, 4.000e+00
          Z Bounds:   -2.000e+01, 4.441e-15
          N Arrays:   2

        Save the unstructured grid to disk.

        >>> ugrid.save("grid.vtu")

        Plot it.

        >>> ugrid.plot(color="w", show_edges=True)

        """
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
        id_map = np.empty(node_section.nid[-1] + 1, dtype=ID_TYPE)
        id_map[node_section.nid] = np.arange(len(node_section), dtype=ID_TYPE)

        element_sections = self.element_shell_sections + self.element_solid_sections

        if not element_sections:
            raise NotImplementedError("Deck missing element sections")

        # add shell sections
        offsets: List[NDArray[np.int64]] = []
        celltypes: List[NDArray[np.uint8]] = []
        cells: List[NDArray[np.int64]] = []
        part_ids = []
        for section in element_sections:
            section_cells, section_offset, section_celltypes = section.to_vtk()
            if offsets:
                # we need to shift by the last value of the last offset
                offsets.append(section_offset[1:] + offsets[-1][-1])
            else:
                offsets.append(section_offset)

            celltypes.append(section_celltypes)
            cells.append(id_map[section_cells])
            part_ids.append(section.pid)

        offsets_arr = np.hstack(offsets, dtype=ID_TYPE)
        cells_arr = np.hstack(cells, dtype=ID_TYPE)
        celltypes_arr = np.hstack(celltypes, dtype=np.uint8)

        grid = UnstructuredGrid()
        grid.points = pv.pyvista_ndarray(node_section.coordinates)

        vtk_cells = CellArray.from_arrays(offsets_arr, cells_arr, deep=False)
        vtk_cell_type = numpy_to_vtk(celltypes_arr, deep=False)
        grid.SetCells(vtk_cell_type, vtk_cells)

        # add part and node ids
        grid.cell_data["Part ID"] = np.hstack(part_ids)
        grid.point_data["Node ID"] = node_section.nid

        return grid

    def __repr__(self) -> str:
        lines = ["LSDYNA Deck with:"]
        lines.append(f"  Node sections:              {len(self.node_sections)}")
        lines.append(f"  Element Solid sections:     {len(self.element_solid_sections)}")
        lines.append(f"  Element Shell sections:     {len(self.element_shell_sections)}")

        return "\n".join(lines)
