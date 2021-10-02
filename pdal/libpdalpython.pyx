# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

import json
from types import SimpleNamespace

cimport cython
from cpython.ref cimport Py_DECREF
from cython.operator cimport dereference as deref
from libcpp cimport bool
from libcpp.memory cimport shared_ptr
from libcpp.set cimport set as stl_set
from libcpp.string cimport string
from libcpp.vector cimport vector

import numpy as np
cimport numpy as np
np.import_array()


cdef extern from "pdal/pdal_config.hpp" namespace "pdal::Config":
    cdef int versionMajor() except +
    cdef int versionMinor() except +
    cdef int versionPatch() except +
    cdef string sha1() except+
    cdef string debugInformation() except+
    cdef string pluginInstallPath() except+
    cdef string versionString() except+

def getInfo():
    return SimpleNamespace(
        version=versionString(),
        major=versionMajor(),
        minor=versionMinor(),
        patch=versionPatch(),
        debug=debugInformation(),
        sha1=sha1(),
        plugin=pluginInstallPath(),
    )


cdef extern from "PyDimension.hpp":
    ctypedef struct Dimension:
        string name;
        string description;
        int size;
        string type;
##         string units; // Not defined by PDAL yet
    cdef vector[Dimension] getValidDimensions() except +

def getDimensions():
    output = []
    for dim in getValidDimensions():
        output.append({
            'name': dim.name,
            'description': dim.description,
            'dtype': np.dtype(dim.type + str(dim.size))
        })
    return output


cdef extern from "pdal/Mesh.hpp" namespace "pdal":
    cdef cppclass TriangularMesh:
        pass


cdef extern from "pdal/PointView.hpp" namespace "pdal":
    cdef cppclass PointView:
        bool empty()
        TriangularMesh* mesh()

    ctypedef shared_ptr[PointView] PointViewPtr
    ctypedef stl_set[PointViewPtr] PointViewSet


cdef extern from "pdal/PipelineExecutor.hpp" namespace "pdal":
    cdef cppclass PipelineExecutor:
        PipelineExecutor(const char*) except +
        int execute() except +
        bool validate() except +
        PointViewSet views() except +
        string getPipeline() except +
        string getMetadata() except +
        string getSchema() except +
        string getLog() except +
        int getLogLevel() except +
        void setLogLevel(int) except +

    cdef cppclass PipelineStreamableExecutor(PipelineExecutor):
        PipelineStreamableExecutor(const char*, int) except +
        PointViewPtr executeNext() except +


cdef extern from "PyArray.hpp" namespace "pdal::python":
    cdef cppclass Array:
        Array(np.ndarray) except +


cdef extern from "PyPipeline.hpp" namespace "pdal::python":
    void addArrayReaders(PipelineExecutor*, vector[Array *]) except +
    np.PyArrayObject* viewToNumpyArray(PointViewPtr view) except +;
    np.PyArrayObject* meshToNumpyArray(const TriangularMesh* mesh) except +;


@cython.internal
cdef class BasePipeline:
    cdef PipelineExecutor* _executor
    cdef vector[Array *] _arrays;

    def __dealloc__(self):
        for array in self._arrays:
            del array
        del self._executor

    property pipeline:
        def __get__(self):
            return self._executor.getPipeline()

    property metadata:
        def __get__(self):
            return self._executor.getMetadata()

    property loglevel:
        def __get__(self):
            return self._executor.getLogLevel()
        def __set__(self, v):
            self._executor.setLogLevel(v)

    property log:
        def __get__(self):
            return self._executor.getLog()

    property schema:
        def __get__(self):
            return json.loads(self._executor.getSchema())

    def validate(self):
        return self._executor.validate()

    cdef _add_arrays(self, list arrays):
        if arrays is not None:
            for array in arrays:
                self._arrays.push_back(new Array(array))
        addArrayReaders(self._executor, self._arrays)


cdef class Pipeline(BasePipeline):
    def __cinit__(self, unicode json, list arrays=None):
        self._executor = new PipelineExecutor(json.encode("UTF-8"))
        self._add_arrays(arrays)

    def execute(self):
        return self._executor.execute()

    property arrays:
        def __get__(self):
            output = []
            for view in self._executor.views():
                output.append(<object>viewToNumpyArray(view))
                Py_DECREF(output[-1])
            return output

    property meshes:
        def __get__(self):
            output = []
            for view in self._executor.views():
                output.append(<object>meshToNumpyArray(deref(view).mesh()))
                Py_DECREF(output[-1])
            return output

    def get_meshio(self, idx):
        try:
            from meshio import Mesh
        except ModuleNotFoundError:
            raise RuntimeError(
                "The get_meshio function can only be used if you have installed meshio. Try pip install meshio"
            )
        array = self.arrays[idx]
        mesh = self.meshes[idx]
        if len(mesh) == 0:
            return None
        return Mesh(
            np.stack((array["X"], array["Y"], array["Z"]), 1),
            [("triangle", np.stack((mesh["A"], mesh["B"], mesh["C"]), 1))],
        )


cdef class PipelineIterator(BasePipeline):
    cdef int _chunk_size

    def __cinit__(self, unicode json, list arrays=None, int chunk_size=10000):
        self._chunk_size = chunk_size
        self._executor = new PipelineStreamableExecutor(json.encode("UTF-8"), chunk_size)
        self._add_arrays(arrays)

    property chunk_size:
        def __get__(self):
            return self._chunk_size

    def __iter__(self):
        while True:
            view = (<PipelineStreamableExecutor*>self._executor).executeNext()
            if not view:
                break
            if not deref(view).empty():
                np_array = <object>viewToNumpyArray(view)
                Py_DECREF(np_array)
                yield np_array
