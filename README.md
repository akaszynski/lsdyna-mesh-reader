## LS-DYNA Mesh Reader

This library can be used to read in LS-DYNA meshes stored within keyword
(`*.k`, `*.key`, `*.dyn`) files, also known as keyword format "input
decks".

Many of these example files were obtained from the excellent documentation at
[LS-DYNA Examples](https://www.dynaexamples.com/).

### Motivation

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
example mesh and getting the first node section.

```py

Load the example file and get the first node section.

>>> import lsdyna_mesh_reader
>>> from lsdyna_mesh_reader import examples
>>> deck = lsdyna_mesh_reader.Deck(examples.birdball)
>>> node_section = deck.node_sections[0]
>>> node_section

```
