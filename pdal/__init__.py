from .libpdalpython import Pipeline, PipelineIterator
from .libpdalpython import getDimensions, getInfo

__version__ = "2.4.2"

dimensions = getDimensions()
info = getInfo()
del getDimensions, getInfo
