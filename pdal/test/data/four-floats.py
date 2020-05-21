import numpy as np


dtype = np.dtype([('X', '>f4'), ('Y', '>f4'), ('Z', '<f4'), ('GPSTime', '>f4')])

def load(filename):
    data = open(filename, 'rb').read()
    array = np.frombuffer(data, dtype=dtype)

    return array
