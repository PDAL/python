(
    Reader(filename="test/data/1.2-with-color.las", spatialreference="EPSG:2993")
    |
    Filter.python(function="filter", module="anything", source="""
import numpy as np


def filter(ins, outs):
    print("entered filter()")
    cls = ins["Classification"]
    keep_classes = [1]

    # Use the first test for our base array.
    keep = np.equal(cls, keep_classes[0])

    # For 1:n, test each predicate and join back
    # to our existing predicate array
    for k in range(1, len(keep_classes)):
        t = np.equal(cls, keep_classes[k])
        keep = keep + t

    outs["Mask"] = keep
    print("exiting filter()")
    return True
""")
    |
    Writer("out2.las")
)
