__version__ = "3.0.1"
__all__ = ["Pipeline", "Stage", "Reader", "Filter", "Writer", "dimensions", "info"]

from . import libpdalpython
from .drivers import inject_pdal_drivers
from .pipeline import Filter, Pipeline, Reader, Stage, Writer

inject_pdal_drivers()
dimensions = libpdalpython.getDimensions()
info = libpdalpython.getInfo()

del inject_pdal_drivers, libpdalpython
