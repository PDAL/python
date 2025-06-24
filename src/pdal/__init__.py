__all__ = ["Pipeline", "Stage", "Reader", "Filter", "Writer", "dimensions", "info"]
__version__ = '3.5.0'

from . import libpdalpython
from .drivers import inject_pdal_drivers
from .pipeline import Filter, Pipeline, Reader, Stage, Writer

inject_pdal_drivers()
dimensions = libpdalpython.getDimensions()
info = libpdalpython.getInfo()

del inject_pdal_drivers, libpdalpython
