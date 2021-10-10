__version__ = "2.4.2"
__all__ = ["Pipeline",  "Reader", "Filter", "Writer", "dimensions", "info"]

from .libpdalpython import getDimensions, getInfo
from .pipeline import Pipeline, Reader, Filter, Writer

dimensions = getDimensions()
info = getInfo()
del getDimensions, getInfo
