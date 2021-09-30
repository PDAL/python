# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

import json
from types import SimpleNamespace

from cpython.ref cimport Py_DECREF
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport int64_t
from libcpp cimport bool

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


cdef extern from "PyArray.hpp" namespace "pdal::python":
    cdef cppclass Array:
        Array(np.ndarray) except +


cdef extern from "pdal/PipelineExecutor.hpp" namespace "pdal":
    cdef cppclass PipelineExecutor:
        PipelineExecutor(const char*) except +
        bool executed() except +
        int64_t execute() except +
        bool validate() except +
        string getPipeline() except +
        string getMetadata() except +
        string getSchema() except +
        string getLog() except +
        int getLogLevel()
        void setLogLevel(int)


cdef extern from "PyPipeline.hpp" namespace "pdal::python":
    void readPipeline(PipelineExecutor*, string) except +
    void addArrayReaders(PipelineExecutor*, vector[Array *]) except +
    vector[np.PyArrayObject*] getArrays(const PipelineExecutor* executor) except +
    vector[np.PyArrayObject*] getMeshes(const PipelineExecutor* executor) except +


cdef class Pipeline:
    cdef PipelineExecutor* _executor
    cdef vector[Array *] _arrays;

    def __cinit__(self, unicode json, list arrays=None):
        self._executor = new PipelineExecutor(json.encode('UTF-8'))
        readPipeline(self._executor, json.encode('UTF-8'))
        if arrays is not None:
            for array in arrays:
                self._arrays.push_back(new Array(array))
        addArrayReaders(self._executor, self._arrays)

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

    property arrays:
        def __get__(self):
            if not self._executor.executed():
                raise RuntimeError("call execute() before fetching arrays")
            return self._vector_to_list(getArrays(self._executor))

    property meshes:
        def __get__(self):
            if not self._executor.executed():
                raise RuntimeError("call execute() before fetching the mesh")
            return self._vector_to_list(getMeshes(self._executor))

    def execute(self):
        return self._executor.execute()

    def validate(self):
        return self._executor.validate()

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

    cdef _vector_to_list(self, vector[np.PyArrayObject*] arrays):
        output = []
        for array in arrays:
            output.append(<object>array)
            Py_DECREF(output[-1])
        return output
