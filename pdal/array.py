import numpy as np
from pdal import libpdalpython

class Array(object):
    """A Numpy Array that can speak PDAL"""

    def __init__(self, data):
        self.p = libpdalpython.PyArray(data)
