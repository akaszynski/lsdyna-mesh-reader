"""LS-DYNA mesh reader."""

from importlib.metadata import PackageNotFoundError, version

from lsdyna_mesh_reader import examples
from lsdyna_mesh_reader.deck import Deck

# get current version from the package metadata
try:
    __version__ = version("lsdyna_mesh_reader")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["examples", "Deck"]
