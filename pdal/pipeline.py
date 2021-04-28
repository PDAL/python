
from pdal import libpdalpython
import numpy as np

class Pipeline(object):
    """A PDAL pipeline object, defined by JSON. See http://www.pdal.io/pipeline.html for more
    information on how to define one"""

    def __init__(self, json, arrays=None):

        if arrays:
            self.p = libpdalpython.PyPipeline(json, arrays)
        else:
            self.p = libpdalpython.PyPipeline(json)

    def get_metadata(self):
        return self.p.metadata
    metadata = property(get_metadata)

    def get_schema(self):
        return self.p.schema
    schema = property(get_schema)

    def get_pipeline(self):
        return self.p.pipeline
    pipeline = property(get_pipeline)

    def get_loglevel(self):
        return self.p.loglevel

    def set_loglevel(self, v):
        self.p.loglevel = v
    loglevel = property(get_loglevel, set_loglevel)

    def get_log(self):
        return self.p.log
    log = property(get_log)

    def execute(self):
        return self.p.execute()

    def validate(self):
        return self.p.validate()

    def get_arrays(self):
        return self.p.arrays
    arrays = property(get_arrays)

    def get_meshes(self):
        return self.p.meshes
    meshes = property(get_meshes)

    def get_meshio(self, idx: int):
        try:
            from meshio import Mesh
        except ModuleNotFoundError:
            raise RuntimeError(
                "The get_meshio function can only be used if you have installed meshio. Try pip install meshio")
        array = self.arrays[idx]
        mesh = self.meshes[idx]
        if len(mesh) == 0:
            return None
        return Mesh(
            np.stack((array['X'], array['Y'], array['Z']), 1),
            [(
                "triangle",
                np.stack((mesh['A'], mesh['B'], mesh['C']), 1)
            )
            ]
        )