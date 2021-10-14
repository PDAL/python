__version__ = "2.4.2"
__all__ = ["Pipeline", "Stage", "Reader", "Filter", "Writer", "dimensions", "info"]

from .drivers import inject_pdal_drivers
from .libpdalpython import getDimensions, getInfo
from .pipeline import Filter, Pipeline, Reader, Stage, Writer

inject_pdal_drivers()
dimensions = getDimensions()
info = getInfo()

del inject_pdal_drivers, getDimensions, getInfo
