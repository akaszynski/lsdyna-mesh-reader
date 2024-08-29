"""Module to read MAPDL ASCII block formatted CDB files."""

import io
import logging
import os
import pathlib
import shutil
from typing import Any, Dict, List, Optional, Sequence, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from pyvista import ID_TYPE, CellType, UnstructuredGrid

from mapdl_archive import _archive, _reader
from mapdl_archive.mesh import Mesh

# types
NPArray_FLOAT32 = NDArray[np.float32]
NPArray_FLOAT64 = NDArray[np.float64]

VTK_VOXEL = 11

log = logging.getLogger(__name__)
log.setLevel("CRITICAL")

U = TypeVar("U", np.int32, np.int64)
T = TypeVar("T", np.float32, np.float64)


class Archive(Mesh):
    """Read a blocked MAPDL archive file or input file.

    Reads a blocked CDB file and optionally parses it to a vtk grid.
    This can be used to read in files written from MAPDL using the
    ``CDWRITE`` command or input files (``'.dat'``) files written from
    ANSYS Workbench.

    Write the archive file using ``CDWRITE, DB, archive.cdb``

    Parameters
    ----------
    filename : string, pathlib.Path
        Filename of block formatted cdb file

    read_parameters : bool, default: False
        Optionally read parameters from the archive file.

    parse_vtk : bool, default: True
        When ``True``, parse the raw data into to VTK format.

    force_linear : bool, default: False
        This parser creates quadratic elements if available. Set
        this to ``True`` to always create linear elements.

    allowable_types : list, optional
        Allowable element types.  Defaults to all valid element
        types in ``mapdl_archive.elements.valid_types``

        See ``help(mapdl_archive.elements)`` for available element types.

    null_unallowed : bool, default: False
        Elements types not matching element types will be stored
        as empty (null) elements.  Useful for debug or tracking
        element numbers.

    verbose : bool, optional
        Print out each step when reading the archive file.  Used for
        debug purposes and defaults to ``False``.

    name : str, optional
        Internally used parameter used to have a custom ``__repr__``.

    read_eblock : bool, default: True
        Read the element block.

    Examples
    --------
    >>> import mapdl_archive
    >>> from mapdl_archive import examples
    >>> hex_beam = mapdl_archive.Archive(examples.hexarchivefile)
    >>> print(hex_beam)
    ANSYS Archive File HexBeam.cdb
      Number of Nodes:              40
      Number of Elements:           321
      Number of Element Types:      1
      Number of Node Components:    2
      Number of Element Components: 2

    Print the node array.

    >>> hex_beam.nodes
    array([[0.  , 0.  , 0.  ],
           [1.  , 0.  , 0.  ],
           [0.25, 0.  , 0.  ],
           ...,
           [0.75, 0.5 , 3.5 ],
           [0.75, 0.5 , 4.  ],
           [0.75, 0.5 , 4.5 ]])

    Read an Ansys mechanical input file.

    >>> my_archive = mapdl_archive.Archive("C:/Users/user/ds.dat")

    Notes
    -----
    This class only reads EBLOCK records with SOLID records.  For
    example, the record ``EBLOCK,19,SOLID,,3588`` will be read, but
    ``EBLOCK,10,,,3588`` will not be read. Generally, MAPDL will only
    write SOLID records and Mechanical Workbench may write SOLID
    records. These additional records will be ignored.
    """

    def __init__(
        self,
        filename: Union[str, pathlib.Path],
        read_parameters: bool = False,
        parse_vtk: bool = True,
        force_linear: bool = False,
        allowable_types: Optional[Union[List[str], List[int]]] = None,
        null_unallowed: bool = False,
        verbose: bool = False,
        name: str = "",
        read_eblock: bool = True,
    ):
        """Initialize an instance of the archive class."""
        self._read_parameters: bool = read_parameters
        self._filename: pathlib.Path = pathlib.Path(filename)
        self._name: str = name
        self._archive = _reader.Archive(
            self.filename,
            read_params=read_parameters,
            debug=verbose,
            read_eblock=read_eblock,
        )
        self._archive.read()

        super().__init__(
            self._archive.nnum,
            self._archive.nodes,
            self._archive.node_angles,
            self._archive.elem,
            self._archive.elem_off,
            np.array(self._archive.elem_type),
            np.array(self._archive.rnum),
            node_comps=self._archive.node_comps,
            elem_comps=self._archive.elem_comps,
            rdat=self._archive.rdat,
            keyopt=self._archive.keyopt,
        )

        self._allowable_types: Optional[Union[List[str], List[int]]] = allowable_types
        self._force_linear: bool = force_linear
        self._null_unallowed: bool = null_unallowed

        if parse_vtk:
            self._grid = self._parse_vtk(allowable_types, force_linear, null_unallowed)

    @property
    def filename(self) -> str:
        """Return the filename."""
        return str(self._filename)

    @property
    def pathlib_filename(self) -> pathlib.Path:
        """Return the filename as a ``pathlib.Path``."""
        return self._filename

    @property
    def parameters(self) -> Dict[str, NDArray[np.double]]:
        """Return the parameters stored in the archive file.

        Parameters have been deprecated.

        """
        raise RuntimeError("Parameters have been deprecated.")

    def __repr__(self) -> str:
        """Return the representation of the archive."""
        if self._name:
            txt: str = f"MAPDL {self._name}\n"
        else:
            basename = os.path.basename(self._filename)
            txt = f"ANSYS Archive File {basename}\n"

        txt += f"  Number of Nodes:              {len(self.nnum)}\n"
        txt += f"  Number of Elements:           {len(self.enum)}\n"
        txt += f"  Number of Element Types:      {len(self.ekey)}\n"
        txt += f"  Number of Node Components:    {len(self.node_components)}\n"
        txt += f"  Number of Element Components: {len(self.element_components)}\n"
        return txt

    @property
    def grid(self) -> UnstructuredGrid:
        """Return a ``pyvista.UnstructuredGrid`` of the archive file.

        Examples
        --------
        >>> import mapdl_archive
        >>> from mapdl_archive import examples
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> archive.grid
        UnstructuredGrid (0x7ffa237f08a0)
          N Cells:      40
          N Points:     321
          X Bounds:     0.000e+00, 1.000e+00
          Y Bounds:     0.000e+00, 1.000e+00
          Z Bounds:     0.000e+00, 5.000e+00
          N Arrays:     13
        """
        # parse the grid using the cached parameters
        if self._grid is None:
            self._grid = self._parse_vtk(
                self._allowable_types, self._force_linear, self._null_unallowed
            )
        return self._grid

    def plot(self, *args: Any, **kwargs: Any) -> Any:
        """Plot the mesh.

        See ``help(pyvista.plot)`` for all optional kwargs.

        """
        if self._grid is None:  # pragma: no cover
            raise AttributeError("Archive must be parsed as a vtk grid.\n Set `parse_vtk=True`")
        kwargs.setdefault("color", "w")
        kwargs.setdefault("show_edges", True)
        return self.grid.plot(*args, **kwargs)

    @property
    def _nblock_start(self) -> int:
        """Return the start of the node block in the original file."""
        return self._archive.nblock_start

    @property
    def _nblock_end(self) -> int:
        """Return the end of the node block in the original file."""
        return self._archive.nblock_end

    def overwrite_nblock(
        self,
        filename: Union[str, pathlib.Path],
        node_id: NDArray[int],
        pos: NDArray[T],
        angles: Optional[NDArray[T]] = None,
        sig_digits: int = 13,
    ) -> None:
        """Write out an archive file to disk while replacing its NBLOCK.

        Parameters
        ----------
        filename : str | pathlib.Path
            Filename to write node block to.
        node_id : numpy.ndarray
            ANSYS node numbers.
        pos : np.ndarray
            Array of node coordinates.
        angles : numpy.ndarray, optional
            Writes the node angles for each node when included.
        sig_digits : int, default: 13
            Number of significant digits to use when writing the nodes. Must be
            greater than 0.

        Examples
        --------
        Write a new archive file that overwrites the NBLOCK with random nodes
        and reuse the existing node numbers.

        >>> import numpy as np
        >>> import mapdl_archive
        >>> archive = mapdl_archive.Archive(examples.hexarchivefile)
        >>> new_nodes = np.random.random(archive.nodes.shape)
        >>> archive.overwrite_nblock("new_archive.cdb", archive.nnum, new_nodes)

        """
        filename = str(filename)
        with open(self._filename, "rb") as src_file, open(filename, "wb") as dest_file:
            # Copy the beginning of the file up to _nblock_start
            dest_file.write(src_file.read(self._nblock_start))

        # Write new nblock
        write_nblock(filename, node_id, pos, angles=angles, mode="a", sig_digits=sig_digits)

        # Copy the rest of the original file
        with open(self._filename, "rb") as src_file, open(filename, "ab") as dest_file:
            src_file.seek(self._nblock_end)
            dest_file.seek(0, io.SEEK_END)

            shutil.copyfileobj(src_file, dest_file)

    def save_as_numpy(self, filename: str) -> None:
        """Save this archive as a numpy "npz" file.

        This reduces the file size by around 50% compared with the Ansys
        blocked file format.

        """
        if not filename.endswith("npz"):
            raise ValueError("Filename must end with '.npz'")

        raise RuntimeError("save_as_numpy as been deprecated")


def save_as_archive(
    filename: Union[pathlib.Path, str],
    grid: UnstructuredGrid,
    mtype_start: int = 1,
    etype_start: int = 1,
    real_constant_start: int = 1,
    mode: str = "w",
    enum_start: int = 1,
    nnum_start: int = 1,
    include_etype_header: bool = True,
    reset_etype: bool = False,
    allow_missing: bool = True,
    include_surface_elements: bool = True,
    include_solid_elements: bool = True,
    include_components: bool = True,
    exclude_missing: bool = False,
    node_sig_digits: int = 13,
) -> None:
    """Write FEM as an ANSYS APDL archive file.

    This function supports the following element types:

        - ``vtk.VTK_HEXAHEDRON``
        - ``vtk.VTK_PYRAMID``
        - ``vtk.VTK_QUADRATIC_HEXAHEDRON``
        - ``vtk.VTK_QUADRATIC_PYRAMID``
        - ``vtk.VTK_QUADRATIC_TETRA``
        - ``vtk.VTK_QUADRATIC_WEDGE``
        - ``vtk.VTK_QUAD``
        - ``vtk.VTK_TETRA``
        - ``vtk.VTK_TRIANGLE``
        - ``vtk.VTK_VOXEL``
        - ``vtk.VTK_WEDGE``

    Will automatically renumber nodes and elements if the FEM does not
    contain ANSYS node or element numbers.  Node numbers are stored as
    a point array ``"ansys_node_num"``, and cell numbers are stored as
    cell array ``"ansys_elem_num"``.

    Parameters
    ----------
    filename : str, pathlib.Path
       Filename to write archive file.

    grid : pyvista.DataSet
        Any :class:`pyvista.DataSet` that can be cast to a
        :class:`pyvista.UnstructuredGrid`.

    mtype_start : int, optional
        Material number to assign to elements.  Can be set manually by
        adding the cell array "mtype" to the unstructured grid.

    etype_start : int, optional
        Starting element type number.  Can be manually set by adding
        the cell array "ansys_etype" to the unstructured grid.

    real_constant_start : int, optional
        Starting real constant to assign to unset cells.  Can be
        manually set by adding the cell array "ansys_real_constant" to
        the unstructured grid.

    mode : str, optional
        File mode.  See ``help(open)``

    enum_start : int, optional
        Starting element number to assign to unset cells.  Can be
        manually set by adding the cell array "ansys_elem_num" to the
        unstructured grid.

    nnum_start : int, optional
        Starting element number to assign to unset points.  Can be
        manually set by adding the point array "ansys_node_num" to the
        unstructured grid.

    include_etype_header : bool, optional
        For each element type, includes element type command
        (e.g. "ET, 1, 186") in the archive file.

    reset_etype : bool, optional
        Resets element type.  Element types will automatically be
        determined by the shape of the element (i.e. quadradic
        tetrahedrals will be saved as SOLID187, linear hexahedrals as
        SOLID185).  Default True.

    include_surface_elements : bool, optional
        Includes surface elements when writing the archive file and
        saves them as SHELL181.

    include_solid_elements : bool, optional
        Includes solid elements when writing the archive file and
        saves them as SOLID185, SOLID186, or SOLID187.

    include_components : bool, optional
        Writes note components to file.  Node components must be
        stored within the unstructured grid as uint8 or bool arrays.

    exclude_missing : bool, default: False
        When ``allow_missing=True``, write ``0`` instead of renumbering
        nodes. This allows you to exclude midside nodes for certain element
        types (e.g. ``SOLID186``). Missing midside nodes are identified as
        ``-1`` in the ``"ansys_node_num"`` array.

    node_sig_digits : int, default: 13
        Number of significant digits to use when writing the nodes. Must be
        greater than 0.

    Examples
    --------
    Write a ``pyvista.UnstructuredGrid`` to ``"archive.cdb"``.

    >>> import mapdl_archive
    >>> from pyvista import examples
    >>> grid = examples.load_hexbeam()
    >>> pymapdl_reader.save_as_archive("archive.cdb", grid)

    """
    if hasattr(grid, "cast_to_unstructured_grid"):
        grid = grid.cast_to_unstructured_grid()

    if not isinstance(grid, UnstructuredGrid):
        raise TypeError(f"``grid`` argument must be an UnstructuredGrid, not {type(grid)}")

    allowable = []
    if include_solid_elements:
        allowable.extend(
            [
                CellType.VOXEL,
                CellType.TETRA,
                CellType.QUADRATIC_TETRA,
                CellType.PYRAMID,
                CellType.QUADRATIC_PYRAMID,
                CellType.WEDGE,
                CellType.QUADRATIC_WEDGE,
                CellType.HEXAHEDRON,
                CellType.QUADRATIC_HEXAHEDRON,
            ]
        )

    if include_surface_elements:
        allowable.extend([CellType.TRIANGLE, CellType.QUAD])
        # VTK_QUADRATIC_TRIANGLE,
        # VTK_QUADRATIC_QUAD

    # extract allowable cell types
    mask = np.isin(grid.celltypes, allowable)
    if not mask.any():
        ucelltypes = np.unique(grid.celltypes)
        allowable.sort()
        raise RuntimeError(
            f"`grid` contains no allowable cell types. Contains types {ucelltypes} "
            f"and only {allowable} are allowed.\n\n"
            "See https://vtk.org/doc/nightly/html/vtkCellType_8h_source.html "
            "for more details."
        )
    grid = grid.extract_cells(mask)

    header = "/PREP7\n"

    # node numbers
    if "ansys_node_num" in grid.point_data:
        nodenum = grid.point_data["ansys_node_num"]
    else:
        log.info("No ANSYS node numbers set in input. Adding default range")
        nodenum = np.arange(1, grid.n_points + 1, dtype=np.int32)

    missing_mask = nodenum == -1
    if np.any(missing_mask):
        if not allow_missing:
            raise RuntimeError('Missing node numbers.  Exiting due "allow_missing=False"')
        elif exclude_missing:
            log.info("Excluding missing nodes from archive file.")
            nodenum = nodenum.copy()
            nodenum[missing_mask] = 0
        else:
            start_num = nodenum.max() + 1
            if nnum_start > start_num:
                start_num = nnum_start
            nadd = np.sum(nodenum == -1)
            end_num = start_num + nadd
            log.info(
                "FEM missing some node numbers.  Adding node numbering " "from %d to %d",
                start_num,
                end_num,
            )
            nodenum[missing_mask] = np.arange(start_num, end_num, dtype=np.int32)

    # element block
    ncells = grid.n_cells
    if "ansys_elem_num" in grid.cell_data:
        enum = grid.cell_data["ansys_elem_num"]
    else:
        if not allow_missing:
            raise RuntimeError('Missing node numbers. Exiting due "allow_missing=False"')
        log.info(
            "No ANSYS element numbers set in input. " "Adding default range starting from %d",
            enum_start,
        )
        enum = np.arange(1, ncells + 1, dtype=np.int32)

    if np.any(enum == -1):
        if not allow_missing:
            raise RuntimeError(
                '-1 encountered in "ansys_elem_num".\nExiting due "allow_missing=False"'
            )

        start_num = enum.max() + 1
        if enum_start > start_num:
            start_num = enum_start
        nadd = np.sum(enum == -1)
        end_num = start_num + nadd
        log.info(
            "FEM missing some cell numbers.  Adding numbering from %d to %d",
            start_num,
            end_num,
        )
        enum[enum == -1] = np.arange(start_num, end_num, dtype=np.int32)

    # material type
    if "ansys_material_type" in grid.cell_data:
        mtype = grid.cell_data["ansys_material_type"]
    else:
        log.info(
            "No ANSYS element numbers set in input.  " "Adding default range starting from %d",
            mtype_start,
        )
        mtype = np.arange(1, ncells + 1, dtype=np.int32)

    if np.any(mtype == -1):
        log.info("FEM missing some material type numbers.  Adding...")
        mtype[mtype == -1] = mtype_start

    # real constant
    if "ansys_real_constant" in grid.cell_data:
        rcon = grid.cell_data["ansys_real_constant"]
    else:
        log.info(
            "No ANSYS element numbers set in input.  " + "Adding default range starting from %d",
            real_constant_start,
        )
        rcon = np.arange(1, ncells + 1, dtype=np.int32)

    if np.any(rcon == -1):
        log.info("FEM missing some material type numbers.  Adding...")
        rcon[rcon == -1] = real_constant_start

    # element type
    invalid = False
    if "ansys_etype" in grid.cell_data and not reset_etype:
        missing = False
        typenum = grid.cell_data["ansys_elem_type_num"]
        etype = grid.cell_data["ansys_etype"]
        if np.any(etype == -1):
            log.warning("Some elements are missing element type numbers.")
            invalid = True

        if include_etype_header and not invalid:
            _, ind = np.unique(etype, return_index=True)
            for idx in ind:
                header += "ET, %d, %d\n" % (etype[idx], typenum[idx])
    else:
        missing = True

    # check if valid
    if not missing:
        mask = grid.celltypes < 20
        if np.any(grid.cell_data["ansys_elem_type_num"][mask] == 186):
            invalid = True
            log.warning("Invalid ANSYS element types.")

    if invalid or missing:
        if not allow_missing:
            raise RuntimeError(
                'Invalid or missing data in "ansys_elem_type_num"'
                ' or "ansys_etype".  Exiting due "allow_missing=False"'
            )

        log.info(
            "No ANSYS element type or invalid data input.  "
            + "Adding default range starting from %d" % etype_start
        )

        etype = np.empty(grid.n_cells, np.int32)

        # VTK to SOLID186 mapping
        # TETRA delegated to SOLID187
        etype_186 = etype_start
        etype_186_types = [
            CellType.QUADRATIC_HEXAHEDRON,
            CellType.QUADRATIC_WEDGE,
            CellType.QUADRATIC_PYRAMID,
        ]
        etype[np.isin(grid.celltypes, etype_186_types)] = etype_186

        etype_187 = etype_start + 1
        etype[grid.celltypes == CellType.QUADRATIC_TETRA] = etype_187

        # VTK to SOLID185 mapping
        etype_185 = etype_start + 2
        etype_185_types = [
            CellType.VOXEL,
            CellType.TETRA,
            CellType.HEXAHEDRON,
            CellType.WEDGE,
            CellType.PYRAMID,
        ]
        etype[np.isin(grid.celltypes, etype_185_types)] = etype_185

        # Surface elements
        etype_181 = etype_start + 3
        etype_181_types = [
            CellType.TRIANGLE,
            CellType.QUAD,
        ]
        etype[np.isin(grid.celltypes, etype_181_types)] = etype_181

        typenum = np.empty_like(etype)
        typenum[etype == etype_185] = 185
        typenum[etype == etype_186] = 186
        typenum[etype == etype_187] = 187
        typenum[etype == etype_181] = 181

        header += f"ET,{etype_185},185\n"
        header += f"ET,{etype_186},186\n"
        header += f"ET,{etype_187},187\n"
        header += f"ET,{etype_181},181\n"

    # number of nodes written per element
    elem_nnodes = np.empty(etype.size, np.int32)
    elem_nnodes[typenum == 181] = 4
    elem_nnodes[typenum == 185] = 8
    elem_nnodes[typenum == 186] = 20
    elem_nnodes[typenum == 187] = 10

    if not reset_etype:
        unsup = np.setdiff1d(typenum, [181, 185, 186, 187])
        if unsup.any():
            raise RuntimeError(
                f"Unsupported element types {unsup}. Either set ``reset_etype=True``"
                " or remove (or relabel) the unsupported element types."
            )

    # edge case where element types are unsupported

    # write the EBLOCK
    filename = str(filename)
    with open(filename, mode) as f:
        f.write(header)

    if exclude_missing:
        log.info("Excluding missing nodes from archive file.")
        write_nblock(
            filename,
            nodenum[~missing_mask],
            grid.points[~missing_mask],
            mode="a",
            sig_digits=node_sig_digits,
        )
    else:
        write_nblock(filename, nodenum, grid.points, mode="a", sig_digits=node_sig_digits)

    # write remainder of eblock
    _write_eblock(
        filename,
        enum,
        etype,
        mtype,
        rcon,
        elem_nnodes,
        grid.cell_connectivity,
        grid.offset,
        grid.celltypes,
        typenum,
        nodenum,
        mode="a",
    )

    if include_components:
        with open(filename, "a") as fid:
            # write node components
            for node_key in grid.point_data:
                arr = grid.point_data[node_key]
                if arr.dtype in [np.uint8, np.bool_]:
                    items = nodenum[arr.view(np.bool_)]
                    write_cmblock(fid, items, node_key, "NODE")

            # write element components
            for node_key in grid.cell_data:
                arr = grid.cell_data[node_key]
                if arr.dtype in [np.uint8, np.bool_]:
                    items = enum[arr.view(np.bool_)]
                    write_cmblock(fid, items, node_key, "ELEM")


def write_nblock(
    filename: Union[str, pathlib.Path],
    node_id: NDArray[int],
    pos: NDArray[T],
    angles: Optional[NDArray[T]] = None,
    mode: str = "w",
    sig_digits: int = 13,
) -> None:
    """Write nodes and node angles to file.

    Parameters
    ----------
    filename : str, pathlib.Path
        Filename to write node block to.
    node_id : numpy.ndarray
        ANSYS node numbers.
    pos : np.ndarray
        Array of node coordinates.
    angles : numpy.ndarray, optional
        Writes the node angles for each node when included.
    mode : str, default: "w"
        Write mode.
    sig_digits : int, default: 13
        Number of significant digits to use when writing the nodes. Must be
        greater than 0.

    Examples
    --------
    Write random points as nodes for MAPDL.

    >>> import numpy as np
    >>> import mapdl_archive
    >>> points = np.random.random((100, 3))
    >>> point_ids = np.arange(1, 101)
    >>> mapdl_archive.write_nblock("nblock.inp", point_ids, points)

    """
    if sig_digits < 1:
        raise ValueError(f"`sig_digits` must be greater than 0, got {sig_digits}")
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError(f"Invalid position array shape {pos.shape}. Should be shaped `(n, 3)`.")
    if angles is not None:
        if angles.ndim != 2 or angles.shape[1] != 3:
            raise ValueError(
                f"Invalid angles array shape {angles.shape}. Should be shaped `(n, 3)`."
            )

    node_id_32: NDArray[np.int32] = node_id.astype(np.int32, copy=False)

    # node array must be sorted
    # note, this is sort check is most suited for pre-sorted arrays
    # see https://stackoverflow.com/questions/3755136/
    if not np.all(node_id_32[:-1] <= node_id_32[1:]):
        sidx = np.argsort(node_id_32)
        node_id_32 = node_id_32[sidx]
        pos = pos[sidx]

    if angles is not None:
        angles = angles.astype(pos.dtype, copy=False)
    else:
        angles = np.empty((0, 0), dtype=pos.dtype)

    # void WriteNblock(
    #     std::string &filename,
    #     const int max_node_id,
    #     const NDArray<const int, 1> node_id_arr,
    #     const NDArray<const double, 2> nodes_arr,
    #     const NDArray<const double, 2> angles_arr,
    #     int sig_digits,
    #     std::string &mode) {

    _archive.write_nblock(
        str(filename),
        node_id_32[-1],
        node_id_32,
        pos.astype(np.float64, copy=False),
        angles.astype(np.float64, copy=False),
        sig_digits,
        mode,
    )

    return None


def write_cmblock(
    filename: Union[str, io.TextIOBase, pathlib.Path],
    items: Union[Sequence[int], NDArray[int]],
    comp_name: str,
    comp_type: str,
    digit_width: int = 10,
    mode: str = "w",
) -> None:
    """Write a component block (CMBLOCK) to a file.

    Parameters
    ----------
    filename : str or file handle
        File to write CMBLOCK component to.
    items : sequence
        Element or node numbers to write.
    comp_name : str
        Name of the component.
    comp_type : str
        Component type to write.  Should be either ``'ELEM'`` or ``'NODE'``.
    digit_width : int, default: 10
        Digit width.
    mode : str, default: "w"
        Write mode.

    Examples
    --------
    Write a node component to disk.

    >>> import write_cmblock
    >>> items = [1, 20, 50, 51, 52, 53]
    >>> mapdl_archive.write_cmblock("cmblock_elem.inp", items, "MY_NODE_COMP", "NODE")

    Write an element component to disk.

    >>> mapdl_archive.write_cmblock("cmblock_elem.inp", items, "MY_NODE_COMP", "ELEM")

    """
    comp_name = comp_name.upper()
    comp_type = comp_type.upper()
    comp_type = "ELEM" if comp_type == "ELEMENT" else comp_type
    if comp_type.upper() not in ["ELEM", "NODE"]:
        raise ValueError("`comp_type` must be either 'ELEM' or 'NODE'")

    opened_file = False
    if isinstance(filename, io.TextIOBase):
        fid = filename
    elif isinstance(filename, (str, pathlib.Path)):
        fid = open(str(filename), mode)  # type: ignore
        opened_file = True
    else:
        raise TypeError(
            "Invalid type {type(filename)} for `filename`. Should be file handle or string."
        )

    if not isinstance(items, np.ndarray):
        items_arr = np.array(items, dtype=np.int32)
    else:
        items_arr = items.astype(np.int32, copy=False)

    # All this python writing could be a bottleneck for non-contiguous CMBLOCKs.
    # consider cythonizing this in the future
    cmblock_items = _archive.cmblock_items_from_array(items_arr)
    nitems = len(cmblock_items)
    print(f"CMBLOCK,{comp_name},{comp_type},{nitems:8d}", file=fid)
    print(f"(8i{digit_width})", file=fid)
    digit_formatter = f"%{digit_width}d"

    # use np savetxt here as it's faster than looping through and
    # writing each line.
    # nearest multiple of 8
    up_to = len(cmblock_items) % 8
    if up_to:  # deal with the zero case
        np.savetxt(fid, cmblock_items[:-up_to].reshape(-1, 8), digit_formatter * 8)

        # write the final line
        chunk = cmblock_items[-up_to:]
        print("".join([digit_formatter] * len(chunk)) % tuple(chunk), file=fid)
    else:
        np.savetxt(fid, cmblock_items.reshape(-1, 8), digit_formatter * 8)

    if opened_file:
        fid.close()


def _write_eblock(
    filename: str,
    elem_id: NDArray[U],
    etype: NDArray[U],
    mtype: NDArray[U],
    rcon: NDArray[U],
    elem_nnodes: NDArray[U],
    cells: NDArray[U],
    offset: NDArray[U],
    celltypes: NDArray[np.uint8],
    typenum: NDArray[U],
    nodenum: NDArray[U],
    mode: str = "a",
) -> None:
    """Write EBLOCK to disk."""
    _archive.write_eblock(
        filename,
        elem_id.size,
        elem_id.astype(np.int32, copy=False),
        etype.astype(np.int32, copy=False),
        mtype.astype(np.int32, copy=False),
        rcon.astype(np.int32, copy=False),
        elem_nnodes.astype(np.int32, copy=False),
        celltypes.astype(np.uint8, copy=False),
        offset.astype(ID_TYPE, copy=False),
        cells.astype(ID_TYPE, copy=False),
        typenum.astype(np.int32, copy=False),
        nodenum.astype(np.int32, copy=False),
        mode,
    )
