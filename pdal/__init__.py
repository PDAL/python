__version__ = "2.4.2"
__all__ = ["Pipeline", "Stage", "Reader", "Filter", "Writer", "dimensions", "info"]

import os

if os.getenv("PDAL_PYTHON_PYBIND11"):
    from . import libpybind11 as libpdalpython
else:
    from . import libpdalpython

from .drivers import inject_pdal_drivers
from .pipeline import Filter, Pipeline, Reader, Stage, Writer

inject_pdal_drivers()
dimensions = libpdalpython.getDimensions()
info = libpdalpython.getInfo()

del inject_pdal_drivers, libpdalpython
