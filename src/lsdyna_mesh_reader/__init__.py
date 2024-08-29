"""MAPDL archive reader."""

from importlib.metadata import PackageNotFoundError, version

from mapdl_archive import examples
from mapdl_archive.archive import (
    Archive,
    save_as_archive,
    write_cmblock,
    write_nblock,
)

# get current version from the package metadata
try:
    __version__ = version("mapdl_archive")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["Archive", "save_as_archive", "write_cmblock", "write_nblock", "examples", "__version__"]
