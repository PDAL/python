from .libpdalpython import getDimensions, getInfo
from .pipeline import Pipeline

__version__ = "2.4.2"

dimensions = getDimensions()
info = getInfo()
del getDimensions, getInfo
