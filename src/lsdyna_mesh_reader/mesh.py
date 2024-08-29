"""Contains the Mesh class used by Archive."""

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TypeVar, Union, cast

import numpy as np
from numpy.typing import NDArray
from pyvista import ID_TYPE, CellArray
from pyvista.core.pointset import PolyData, UnstructuredGrid
from vtkmodules.util.numpy_support import numpy_to_vtk

from mapdl_archive import _archive, _reader
from mapdl_archive.elements import ETYPE_MAP

COMP_DICT = Dict[str, NDArray[np.int32]]

INVALID_ALLOWABLE_TYPES = TypeError(
    "`allowable_types` must be an array of ANSYS element types from 1 and 300"
)

# map MESH200 elements to a mapdl_archive/VTK element type (see elements.py)
MESH200_MAP: Dict[int, int] = {
    0: 2,  # line
    1: 2,  # line
    2: 2,  # line
    3: 2,  # line
    4: 3,  # triangle
    5: 3,  # triangle
    6: 3,  # quadrilateral
    7: 3,  # quadrilateral
    8: 5,  # tetrahedron with 4 nodes
    9: 5,  # tetrahedron with 10 nodes
    10: 4,  # hex with 8 nodes
    11: 4,
}  # hex with 8 nodes

SHAPE_MAP: Dict[int, str] = {  # from ELIST definition
    0: "",
    1: "LINE",
    2: "PARA",
    3: "ARC ",
    4: "CARC",
    5: "",
    6: "TRIA",
    7: "QUAD",
    8: "TRI6",
    9: "QUA8",
    10: "POIN",
    11: "CIRC",
    12: "",
    13: "",
    14: "CYLI",
    15: "CONE",
    16: "SPHE",
    17: "",
    18: "",
    19: "PILO",
}
# element type to VTK conversion function call map
# 0: skip
# 1: Point
# 2: Line (linear or quadratic)
# 3: Shell
# 4: 3D Solid (Hexahedral, wedge, pyramid, tetrahedral)
# 5: Tetrahedral
# 6: Line (always linear)
TARGE170_MAP = {
    "TRI": 3,  # 3-Node Triangle
    "QUAD": 3,  # 4-Node Quadrilateral
    "CYLI": 0,  # Not supported (NS)  # Cylinder
    "CONE": 0,  # NS  # Cone
    "TRI6": 3,  # 6-Node triangle
    "SPHE": 0,  # NS  # Sphere
    "PILO": 1,  # Pilot Node
    "QUAD8": 3,  # 8-Node Quadrilateral
    "LINE": 2,  # Line
    "PARA": 2,  # Parabola
    "POINT": 1,  # Point
}

T = TypeVar("T", np.float32, np.float64)


def unique_rows(a: NDArray[T]) -> Tuple[NDArray[T], NDArray[int], NDArray[int]]:
    """Return unique rows of an array and the indices of those rows."""
    if not a.flags.c_contiguous:
        a = np.ascontiguousarray(a)

    b = a.view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    _, idx, idx2 = np.unique(b, True, True)

    return a[idx], idx, idx2.ravel()


class Mesh:
    """Common class between Archive, and result mesh."""

    def __init__(
        self,
        nnum: NDArray[np.int32],
        nodes: NDArray[np.float64],
        node_angles: NDArray[np.float64],
        elem: NDArray[np.int32],
        elem_off: NDArray[np.int32],
        ekey: NDArray[np.int32],
        rnum: NDArray[np.int64],
        node_comps: COMP_DICT = {},
        elem_comps: COMP_DICT = {},
        rdat: List[List[float]] = [],
        keyopt: Dict[int, List[List[int]]] = {},
    ):
        """Initialize the mesh."""
        self._etype: Optional[NDArray[np.int32]] = None  # internal element type reference
        self._grid: Optional[UnstructuredGrid] = None
        self._surf_cache: Optional[PolyData] = None  # cached external surface
        self._enum: Optional[NDArray[np.int32]] = None  # cached element numbering
        self._etype_cache: Optional[NDArray[np.int32]] = None  # cached ansys ETYPE num
        self._rcon: Optional[NDArray[np.int32]] = None  # ansys element real constant
        self._mtype: Optional[NDArray[np.int32]] = None  # cached ansys material type
        self._cached_elements: Optional[List[NDArray[np.int32]]] = None
        self._secnum: Optional[NDArray[np.int32]] = None  # cached section number
        self._esys: Optional[NDArray[np.int32]] = None  # cached element coordinate system
        self._etype_id: Optional[NDArray[np.int32]] = None  # cached element type id

        # Always set on init
        self._nnum: NDArray[np.int32] = nnum
        self._nodes: NDArray[np.float64] = nodes
        self._node_angles: NDArray[np.float64] = node_angles
        self._elem: NDArray[np.int32] = elem
        self._elem_off: NDArray[np.int32] = elem_off
        self._ekey: NDArray[np.int32] = ekey

        # optional
        self._node_comps: COMP_DICT = node_comps
        self._elem_comps: COMP_DICT = elem_comps
        self._rdat: List[List[float]] = rdat
        self._rnum: NDArray[np.int64] = rnum
        self._keyopt: Dict[int, List[List[int]]] = keyopt
        self._tshape: Optional[NDArray[np.int32]] = None
        self._tshape_key: Optional[NDArray[np.int32]] = None

    @property
    def _surf(self) -> PolyData:
        """Return the external surface."""
        if self._surf_cache is None:
            if self._grid is None:
                raise AttributeError("Missing grid.")
            self._surf_cache = self._grid.extract_surface()
        return self._surf_cache

    @property
    def _has_nodes(self) -> bool:
        """Return ``True`` when has nodes."""
        if self.nodes is None:
            return False
        return bool(len(self.nodes))

    @property
    def _has_elements(self) -> bool:
        """Return ``True`` when geometry has elements."""
        if self._elem is None:
            return False

        if isinstance(self._elem, np.ndarray):
            return bool(self._elem.size)

        return bool(len(self._elem))

    def _parse_vtk(
        self,
        allowable_types: Optional[Union[List[str], List[int]]] = None,
        force_linear: bool = False,
        null_unallowed: bool = False,
        fix_midside: bool = True,
        additional_checking: bool = False,
    ) -> UnstructuredGrid:
        """Convert raw ANSYS nodes and elements to an UnstructuredGrid.

        Parameters
        ----------
        fix_midside : bool, optional
            Adds additional midside nodes when ``True``. When ``False``,
            missing ANSYS cells will simply point to the first node.

        """
        if not self._has_nodes or not self._has_elements:
            # warnings.warn('Missing nodes or elements.  Unable to parse to vtk')
            return UnstructuredGrid()

        etype_map = ETYPE_MAP
        if allowable_types is not None:
            try:
                allowable_types_arr: NDArray[np.int_] = np.asarray(allowable_types)
            except Exception as e:
                warnings.warn(f"{e}")
                raise INVALID_ALLOWABLE_TYPES

            if not issubclass(allowable_types_arr.dtype.type, np.integer):
                raise TypeError("Element types must be an integer array-like")

            if allowable_types_arr.min() < 1 or allowable_types_arr.max() > 300:
                raise INVALID_ALLOWABLE_TYPES

            etype_map = np.zeros_like(ETYPE_MAP)
            etype_map[allowable_types] = ETYPE_MAP[allowable_types_arr]

        # ANSYS element type to VTK map
        type_ref = np.empty(2 << 16, np.int32)  # 131072
        try:
            type_ref[self._ekey[:, 0]] = etype_map[self._ekey[:, 1]]
        except:
            print(self._ekey[:, 1])  # (debugging)
            raise

        if allowable_types is None or 200 in allowable_types:
            for etype_ind, etype in self._ekey:
                # MESH200
                if etype == 200 and etype_ind in self.key_option:
                    # keyoption 1 contains various cell types
                    # map them to the corresponding type (see elements.py)
                    mapped = MESH200_MAP[self.key_option[etype_ind][0][1]]
                    type_ref[etype_ind] = mapped

                # TARGE170 specifics
                if etype == 170:
                    # edge case where missing element within the tshape_key
                    if etype_ind not in self.tshape_key:  # pragma: no cover
                        continue
                    tshape_num = cast(int, self.tshape_key[etype_ind])
                    if tshape_num >= 19:  # weird bug when 'PILO' can be 99 instead of 19.
                        tshape_num = 19
                    tshape_label = SHAPE_MAP[tshape_num]
                    type_ref[etype_ind] = TARGE170_MAP.get(tshape_label, 0)

        offset, celltypes, cells = _reader.ans_to_vtk(
            self._elem,
            self._elem_off,
            type_ref,
            self.nnum,
        )  # for reset_midside

        nodes, angles, nnum = self.nodes, self.node_angles, self.nnum

        # fix missing midside
        if np.any(cells == -1):
            if fix_midside:
                nodes, angles, nnum = fix_missing_midside(
                    cells, nodes, celltypes, offset, angles, nnum
                )
            else:
                cells[cells == -1] = 0

        if additional_checking:
            cells[cells < 0] = 0
            # cells[cells >= nodes.shape[0]] = 0  # fails when n_nodes < 20

        grid = UnstructuredGrid()
        grid.points = nodes

        # Warn when type mismatch as it results in copying setting cells
        if cells.dtype != ID_TYPE:
            warnings.warn(f"Mismatch between cell dtype {cells.dtype} and VTK ID_TYPE {ID_TYPE}")

        vtk_cells = CellArray.from_arrays(offset, cells, deep=False)
        vtk_cell_type = numpy_to_vtk(celltypes, deep=False)
        grid.SetCells(vtk_cell_type, vtk_cells)

        # Store original ANSYS element and node information
        grid.point_data["ansys_node_num"] = nnum
        grid.cell_data["ansys_elem_num"] = self.enum
        grid.cell_data["ansys_real_constant"] = self.elem_real_constant
        grid.cell_data["ansys_material_type"] = self.material_type
        grid.cell_data["ansys_etype"] = self._ans_etype
        grid.cell_data["ansys_elem_type_num"] = self.etype

        # add components
        # Add element components to unstructured grid
        for key, item in self.element_components.items():
            mask = np.isin(self.enum, item, assume_unique=True)
            grid.cell_data[key] = mask

        # Add node components to unstructured grid
        for key, item in self.node_components.items():
            mask = np.isin(nnum, item, assume_unique=True)
            grid.point_data[key] = mask

        # store node angles
        if angles is not None:
            if angles.shape[1] == 3:
                grid.point_data["angles"] = angles

        if not null_unallowed:
            grid = grid.extract_cells(grid.celltypes != 0)

        if force_linear:
            # only run if the grid has points or cells
            if grid.n_points:
                grid = grid.linear_copy()

        # map over element types
        # Add tracker for original node numbering
        ind = np.arange(grid.n_points)
        grid.point_data["origid"] = ind
        grid.point_data["VTKorigID"] = ind
        return grid

    @property
    def key_option(self) -> Dict[int, List[List[int]]]:
        """Return additional key options for element types.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.key_option
        {1: [[1, 11]]}
        """
        return self._keyopt

    @property
    def material_type(self) -> NDArray[np.int32]:
        """Return the material type index of each element in the archive.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.material_type
        array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1], dtype=int32)
        """
        # FIELD 0 : material reference number
        if self._mtype is None:
            self._mtype = self._elem[self._elem_off[:-1]]
        return self._mtype

    @property
    def element_components(self) -> COMP_DICT:
        """Return Element components for the archive.

        Output is a dictionary of element components. Each entry is an array of
        MAPDL element numbers corresponding to the element component. The keys
        are element component names.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.element_components
        {'ECOMP1 ': array([17, 18, 21, 22, 23, 24, 25, 26, 27, 28, 29,
                           30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
                           dtype=int32),
        'ECOMP2 ': array([ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                          14, 15, 16, 17, 18, 19, 20, 23, 24], dtype=int32)}
        """
        return self._elem_comps

    @property
    def node_components(self) -> COMP_DICT:
        """Return the node components for the archive.

        Output is a dictionary of node components. Each entry is an array of
        MAPDL node numbers corresponding to the node component. The keys are
        node component names.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.node_components
        {'NCOMP2  ': array([  1,   2,   3,   4,   5,   6,   7,   8,
                             14, 15, 16, 17, 18, 19, 20, 21, 43, 44,
                             62, 63, 64, 81, 82, 90, 91, 92, 93, 94,
                             118, 119, 120, 121, 122, 123, 124, 125,
                             126, 137, 147, 148, 149, 150, 151, 152,
                             153, 165, 166, 167, 193, 194, 195, 202,
                             203, 204, 205, 206, 207, 221, 240, 258,
                             267, 268, 276, 277, 278, 285, 286, 287,
                             304, 305, 306, 313, 314, 315, 316
                             ], dtype=int32),
        ...,
        }
        """
        return self._node_comps

    @property
    def elem_real_constant(self) -> NDArray[np.int32]:
        """Return the real constant reference for each element.

        Use the data within ``rlblock`` and ``rlblock_num`` to get the
        real constant datat for each element.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.elem_real_constant
        array([ 1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
                ...,
                1,  1,  1,  1,  1,  1,  1,  1,  1,  1, 61, 61, 61, 61,
               61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61,
               61], dtype=int32)

        """
        # FIELD 2 : real constant reference number
        if self._rcon is None:
            self._rcon = self._elem[self._elem_off[:-1] + 2]
        return self._rcon

    @property
    def etype(self) -> NDArray[np.int32]:
        """Return the element type of each element.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.etype
        array([ 45,  45,  45,  45,  45,  45,  45,  45,  45,  45,  45,
                45,  45,  45,  45,  45,  45,  45,  45,  92,  92,  92,
                92,  92,  92,  92,  92,  92,  92,  92,  92,  92,  92,
                ...,
                92,  92,  92,  92,  92, 154, 154, 154, 154, 154, 154,
               154, 154, 154, 154, 154, 154, 154, 154, 154, 154, 154,
               154], dtype=int32)

        Notes
        -----
        Element types are listed below. Please see the APDL Element Reference
        for more details:

        https://www.mm.bme.hu/~gyebro/files/vem/ansys_14_element_reference.pdf
        """
        if self._etype is None:
            arr = np.empty(self._ekey[:, 0].max() + 1, np.int32)
            arr[self._ekey[:, 0]] = self._ekey[:, 1]
            self._etype = arr[self._ans_etype]
        return self._etype

    @property
    def _ans_etype(self) -> NDArray[np.int32]:
        """Return field 1, the element type number."""
        if self._etype_cache is None:
            self._etype_cache = self._elem[self._elem_off[:-1] + 1]
        return self._etype_cache

    @property
    def section(self) -> NDArray[np.int32]:
        """Return the section number.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.section
        array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1], dtype=int32)
        """
        if self._secnum is None:
            self._secnum = self._elem[self._elem_off[:-1] + 3]  # FIELD 3
        return self._secnum

    def element_coord_system(self) -> NDArray[np.int32]:
        """Return the element coordinate system number.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.element_coord_system
        array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0], dtype=int32)
        """
        if self._esys is None:
            self._esys = self._elem[self._elem_off[:-1] + 4]  # FIELD 4
        return self._esys

    @property
    def elem(self) -> List[NDArray[np.int32]]:
        """Return the list of elements containing raw element information.

        Each element contains 10 items plus the nodes belonging to the
        element.  The first 10 items are:

        - FIELD 0 : material reference number
        - FIELD 1 : element type number
        - FIELD 2 : real constant reference number
        - FIELD 3 : section number
        - FIELD 4 : element coordinate system
        - FIELD 5 : death flag (0 - alive, 1 - dead)
        - FIELD 6 : solid model reference
        - FIELD 7 : coded shape key
        - FIELD 8 : element number
        - FIELD 9 : base element number (applicable to reinforcing elements only)
        - FIELDS 10 - 30 : The nodes belonging to the element in ANSYS numbering.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.elem
        [array([  1,   4,  19,  15,  63,  91, 286, 240,   3,  18,  17,
                 16,  81, 276, 267, 258,  62,  90, 285, 239],
         array([  4,   2,   8,  19,  91,  44, 147, 286,   5,   7,  21,
                 18, 109, 137, 313, 276,  90,  43, 146, 285],
         array([ 15,  19,  12,  10, 240, 286, 203, 175,  17,  20,  13,
                 14, 267, 304, 221, 230, 239, 285, 202, 174],
        ...
        """
        if self._cached_elements is None:
            self._cached_elements = np.split(self._elem, self._elem_off[1:-1])
        return self._cached_elements

    @property
    def enum(self) -> NDArray[np.int32]:
        """Return the MAPDl element numbers.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.enum
        array([    1,     2,     3, ...,  9998,  9999, 10000])
        """
        if self._enum is None:
            self._enum = self._elem[self._elem_off[:-1] + 8]
        return self._enum

    @property
    def nnum(self) -> NDArray[np.int32]:
        """Return the array of node numbers.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.nnum
        array([    1,     2,     3, ..., 19998, 19999, 20000])
        """
        return self._nnum

    @property
    def ekey(self) -> NDArray[np.int32]:
        """Return the element type key.

        Array containing element type numbers in the first column and
        the element types (like SURF154) in the second column.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.ekey
        array([[  1,  45],
               [  2,  95],
               [  3,  92],
               [ 60, 154]], dtype=int32)
        """
        return self._ekey

    @property
    def rlblock(self) -> List[List[float]]:
        """Return the real constant data from the RLBLOCK.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.rlblock
        [[0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.02 ],
         [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.01 ],
         [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.005],
         [0.   , 0.   , 0.   , 0.   , 0.   , 0.   , 0.005]]
        """
        return self._rdat

    @property
    def rlblock_num(self) -> NDArray[np.int64]:
        """Return the indices from the real constant data.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.rnum
        array([60, 61, 62, 63])
        """
        return self._rnum

    @property
    def nodes(self) -> NDArray[np.float64]:
        """Return the array of nodes of the mesh.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.nodes
        [[0.   0.   0.  ]
         [1.   0.   0.  ]
         [0.25 0.   0.  ]
         ...,
         [0.75 0.5  3.5 ]
         [0.75 0.5  4.  ]
         [0.75 0.5  4.5 ]]
        """
        return self._nodes

    @property
    def node_angles(self) -> Optional[NDArray[np.float64]]:
        """Return the node angles from the archive file.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.nodes
        [[0.   0.   0.  ]
         [0.   0.   0.  ]
         [0.   0.   0.  ]
         ...,
         [0.   0.   0.  ]
         [0.   0.   0.  ]
         [0.   0.   0.  ]]

        """
        return self._node_angles

    def __repr__(self) -> str:
        """Return the representation of the mesh."""
        txt: str = "ANSYS Mesh\n"
        txt += f"  Number of Nodes:              {len(self.nnum)}\n"
        txt += f"  Number of Elements:           {len(self.enum)}\n"
        txt += f"  Number of Element Types:      {len(self.ekey)}\n"
        txt += f"  Number of Node Components:    {len(self.node_components)}\n"
        txt += f"  Number of Element Components: {len(self.element_components)}\n"
        return txt

    def save(
        self,
        filename: Union[str, Path],
        binary: bool = True,
        force_linear: bool = False,
        allowable_types: Optional[Union[List[int], List[str]]] = None,
        null_unallowed: bool = False,
    ) -> None:
        """Save the geometry as a vtk file.

        Parameters
        ----------
        filename : str, pathlib.Path
            Filename of output file. Writer type is inferred from
            the extension of the filename.
        binary : bool, optional
            If ``True``, write as binary, else ASCII.
        force_linear : bool, optional
            This parser creates quadratic elements if available.  Set
            this to True to always create linear elements.  Defaults
            to False.
        allowable_types : list, optional
            Allowable element types.  Defaults to all valid element
            types in ``mapdl_archive.elements.valid_types``

            See ``help(mapdl_archive.elements)`` for available element types.
        null_unallowed : bool, optional
            Elements types not matching element types will be stored
            as empty (null) elements.  Useful for debug or tracking
            element numbers.  Default False.

        Examples
        --------
        >>> geom.save("mesh.vtk")

        Notes
        -----
        Binary files write much faster than ASCII and have a smaller
        file size.

        """
        grid = self._parse_vtk(
            allowable_types=allowable_types,
            force_linear=force_linear,
            null_unallowed=null_unallowed,
        )
        grid.save(str(filename), binary=binary)
        return None

    @property
    def n_node(self) -> int:
        """Return the number of nodes."""
        if not self._has_nodes:
            return 0
        return self.nodes.shape[0]

    @property
    def n_elem(self) -> int:
        """Return the number of nodes."""
        if not self._has_elements:
            return 0
        return len(self.enum)

    @property
    def et_id(self) -> NDArray[np.int32]:
        """Element type id (ET) for each element."""
        if self._etype_id is None:
            etype_elem_id = self._elem_off[:-1] + 1
            self._etype_id = self._elem[etype_elem_id]
        return self._etype_id

    @property
    def tshape(self) -> NDArray[np.int32]:
        """Tshape of contact elements."""
        if self._tshape is None:
            shape_elem_id = self._elem_off[:-1] + 7
            self._tshape = self._elem[shape_elem_id]
        return self._tshape

    @property
    def tshape_key(
        self, as_array: bool = False
    ) -> Union[NDArray[np.int32], Dict[int, NDArray[np.int32]]]:
        """Return a dictionary with the mapping between element type and element shape.

        TShape is only applicable to contact elements.
        """
        if self._tshape_key is None:
            self._tshape_key = np.unique(np.vstack((self.et_id, self.tshape)), axis=1).T

        if as_array:
            return self._tshape_key
        return {elem_id: tshape for elem_id, tshape in self._tshape_key}


def fix_missing_midside(
    cells: NDArray[np.int64],
    nodes: NDArray[np.double],
    celltypes: NDArray[np.uint8],
    offset: NDArray[np.int64],
    angles: Optional[NDArray[np.float64]],
    nnum: NDArray[np.int32],
) -> Tuple[NDArray[np.float64], Optional[NDArray[np.float64]], NDArray[np.int32]]:
    """Add missing midside nodes to cells.

    ANSYS sometimes does not add midside nodes, and this is denoted in
    the element array with a ``0``.  When translated to VTK, this is
    saved as a ``-1``.  If this is not corrected, VTK will segfault.

    This function creates missing midside nodes for the quadratic
    elements.
    """
    # Check for missing midside nodes
    mask = cells == -1
    nnodes = nodes.shape[0]

    nextra = mask.sum()
    cells[mask] = np.arange(nnodes, nnodes + nextra)

    nodes_new = np.empty((nnodes + nextra, 3))
    nodes_new[:nnodes] = nodes
    nodes_new[nnodes:] = 0  # otherwise, segfault disaster

    # Set new midside nodes directly between their edge nodes
    temp_nodes = nodes_new.copy()
    _archive.reset_midside(celltypes, cells, offset, temp_nodes)

    # merge midside nodes
    node_slice = temp_nodes[nnodes:]
    unique_nodes, idx_a, idx_b = unique_rows(node_slice)

    # rewrite node numbers
    cells[mask] = idx_b + nnodes
    nextra = idx_a.shape[0]  # extra unique nodes
    nodes_new = nodes_new[: nnodes + nextra]
    nodes_new[nnodes:] = unique_nodes

    if angles is not None:
        new_angles = np.empty((nnodes + nextra, 3))
        new_angles[:nnodes] = angles
        new_angles[nnodes:] = 0
    else:
        new_angles = None

    # Add extra node numbers
    nnum_new = np.empty(nnodes + nextra, np.int32)
    nnum_new[:nnodes] = nnum
    nnum_new[nnodes:] = -1
    return nodes_new, new_angles, nnum_new
