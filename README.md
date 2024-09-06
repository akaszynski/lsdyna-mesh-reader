## LS-DYNA Mesh Reader

This library can be used to read in LS-DYNA meshes stored within `*.k` files,
commonly known as keyword format "input decks".

Many of these example files were obtained from the excellent documentation at
[LS-DYNA Examples](https://www.dynaexamples.com/).

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

Let's load the [Contact Eroding
I](https://www.dynaexamples.com/introduction/intro-by-a.-tabiei/contact/contact-eroding-i)
example mesh and output the node coordinates as a numpy array

```py
>>> import lsdyna_mesh_reader
>>> from lsdyna_mesh_reader import examples
>>> deck = lsdyna_mesh_reader.Deck(examples.birdball)
>>> deck.nodes
```
