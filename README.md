# LS-DYNA Mesh Reader

This library can be used to read in LS-DYNA meshes stored within keyword
(`*.k`, `*.key`, `*.dyn`) files, also known as keyword format "input
decks". Full documentation for this repository can be found at [lsdyna-mesh-reader Documentation](https://akaszynski.github.io/lsdyna-mesh-reader/).

Many of these example files were obtained from the excellent documentation at
[LS-DYNA Examples](https://www.dynaexamples.com/).

## Motivation

Despite its popularity, there doesn't appear to be a reader for LS-DYNA keyword
files. I need a reader for a closed source project and hope that this helps
someone else who also wishes to read in these files. It borrows from
[mapdl-archive](https://github.com/akaszynski/mapdl-archive) as MAPDL also
follows many of the same FORTRAN conventions when writing out FEMs.

## Installation

Install the fully featured reader with visualization with:

```bash
pip install lsdyna-mesh-reader[pyvista]
```

If you only need the node and element arrays and not any VTK features
(e.g. plotting or
[UnstructuredGrid](https://docs.pyvista.org/api/core/_autosummary/pyvista.unstructuredgrid)
representation), install the basic library with:

```bash
pip install lsdyna-mesh-reader
```

### Examples

Before going through a basic example, let's talk about how these "decks" are organized. Each keyword file contains "keywords" that describe the start of sections of "cards". This terminology dates back to when DYNA3D was developed in 1976 where programs were written on [punch cards](https://en.wikipedia.org/wiki/Punched_card).

To read in nodes and elements, we have to read in one or more node and element sections, each starting with `*NODE` or `*ELEMENT_SOLID`. This library loads in those raw sections as well as parsed sections as a higher level abstraction.

Let's start by loading the [Contact Eroding
I](https://www.dynaexamples.com/introduction/intro-by-a.-tabiei/contact/contact-eroding-i)
example deck.

```py

Load the birdball deck.

>>> import lsdyna_mesh_reader
>>> from lsdyna_mesh_reader import examples
>>> deck = lsdyna_mesh_reader.Deck(examples.birdball)
LSDYNA Deck with:
  Node sections:              1
  Element Solid sections:     1
  Element Shell sections:     1
```

We can now inspect one of the node sections:

```py
>>> node_section = deck.node_sections[0]
>>> node_section
NodeSection containing 1281 nodes

|  NID  |       X       |       Y       |       Z       |   tc   |   rc   |
|-------|---------------|---------------|---------------|--------|--------|
       1 -2.30940104e+00 -2.30940104e+00 -2.30940104e+00        0        0
       2 -2.03960061e+00 -2.03960061e+00 -2.03960061e+00        0        0
       3 -1.76980031e+00 -1.76980031e+00 -1.76980031e+00        0        0
       4 -1.50000000e+00 -1.50000000e+00 -1.50000000e+00        0        0
       5 -2.59364843e+00 -1.59561157e+00 -2.59364843e+00        0        0
       6 -2.22909880e+00 -1.39707434e+00 -2.22909880e+00        0        0
       7 -1.86454940e+00 -1.19853711e+00 -1.86454940e+00        0        0
       8 -1.50000000e+00 -1.00000000e+00 -1.50000000e+00        0        0
       9 -2.76911068e+00 -8.14893484e-01 -2.76911068e+00        0        0
      10 -2.34607387e+00 -7.09928930e-01 -2.34607387e+00        0        0
...

```

We can directly access the node IDs and arrays of the node section:

```py
Node IDs

>>> node_section.nid
array([   1,    2,    3, ..., 1342, 1343, 1344], dtype=int32)

Node coordinates

>>> node_section.coordinates
array([[ -2.30940104,  -2.30940104,  -2.30940104],
       [ -2.03960061,  -2.03960061,  -2.03960061],
       [ -1.76980031,  -1.76980031,  -1.76980031],
       ...,
       [ -4.        , -10.        ,   0.        ],
       [ -2.        , -10.        ,   0.        ],
       [  0.        , -10.        ,   0.        ]])
```

The same can be done for both the solid and shell element sections.

```py
>>> deck.element_solid_sections  # or deck.element_shell_sections
[ElementSolidSection containing 816 elements

 |  EID  |  PID  |  N1   |  N2   |  N3   |  N4   |  N5   |  N6   |  N7   |  N8   |
 |-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
        1       1       1       2       6       5      17      18      22      21
        2       1       2       3       7       6      18      19      23      22
        3       1       3       4       8       7      19      20      24      23
        4       1       5       6      10       9      21      22      26      25
        5       1       6       7      11      10      22      23      27      26
        6       1       7       8      12      11      23      24      28      27
        7       1       9      10      14      13      25      26      30      29
        8       1      10      11      15      14      26      27      31      30
        9       1      11      12      16      15      27      28      32      31
       10       1      17      18      22      21      33      34      38      37
 ...]

Element IDs

>>> section = deck.element_solid_sections[0]
>>> section.eid
array([  1,   2,   3, ..., 814, 815, 816], dtype=int32)

Node IDs of the elements

>>>
array([   1,    2,    6, ..., 1256, 1267, 1266], dtype=int32)

```

The elements are stored as a single contiguous array for efficiency and can be
split up via:

```py
>>> import numpy as np
>>> elements = np.split(section.node_ids, section.node_id_offsets[1:-1])
[array([ 1,  2,  6,  5, 17, 18, 22, 21], dtype=int32),
 array([ 2,  3,  7,  6, 18, 19, 23, 22], dtype=int32),
...
]
```

If you have `pyvista` installed or installed the library with `pip install
lsdyna-mesh-reader[pyvista]`, you can convert the mesh to a single unstructured
grid:

```py
>>> grid = deck.to_grid()
>>> grid
UnstructuredGrid (0x70d5d723bc40)
  N Cells:    916
  N Points:   1281
  X Bounds:   -2.000e+01, 2.220e-15
  Y Bounds:   -1.000e+01, 4.000e+00
  Z Bounds:   -2.000e+01, 4.441e-15
  N Arrays:   2
```

This lets you plot, save, or perform other operations on the mesh via
PyVista. For example, you could plot the resulting mesh. Here's a full example using the [Yaris Static Suspension System Loading Examplew](https://www.dynaexamples.com/implicit/yaris-static-suspension-system-loading).

```py
>>> filename = "YarisD_V2g_shock_abs_load_01.k"
>>> deck = Deck(filename)
>>> grid = deck.to_grid()
>>> grid.plot(color="w", smooth_shading=True, show_edges=True)
```

![Yaris Static Suspension Mesh](https://github.com/akaszynski/lsdyna-mesh-reader/blob/main/docs/source/images/yaris-mesh.png)

### Caveats and Limitations

As of now, limited testing has been performed on this library and you may find
that it fails to load complex or simple keyword decks.

Additionally, this reader only supports the following keywords:

* `*NODE`
* `*ELEMENT_SHELL`
* `*ELEMENT_SOLID`

The VTK UnstructuredGrid contains only the linear element conversion of the
underlying LS-DYNA elements, and only supports `VTK_QUAD`, `VTK_TRIANGLE`,
`VTK_TETRA`, `VTK_WEDGE`, and `VTK_HEXAHEDRAL`.


## Issues and Contributing

Feel free to open an [Issue](https://github.com/akaszynski/lsdyna-mesh-reader/issues) in this repository with any features you'd like me to add or bugs you need fixed.

If you'd like to contribute, please see [CONTRIBUTING.md](https://github.com/akaszynski/lsdyna-mesh-reader/blob/main/CONTRIBUTING.md).


## License

Source and content is under the MIT License, except for the LS-DYNA artifacts,
which retain their original license from LS-DYNA and Ansys.

Note that the example files used here were downloaded from [LS-DYNA
Examples](https://www.dynaexamples.com/) and have the following usage license
as noted on the website:

    The input files and several class notes are available for download. The
    download is free of charge, a login is not required. All examples are
    presented with a brief description. You may find an example by checking a
    specific class or by using the search functionality of the site.

    The content is prepared for educational purposes. Hence, material
    properties and other parameters might be non-physic for simplification.
