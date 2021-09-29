# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

import json
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport int64_t
from libcpp cimport bool
from types import SimpleNamespace

import numpy as np
cimport numpy as np
np.import_array()

from cython.operator cimport dereference as deref, preincrement as inc


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
        void *getPythonArray() except+

    cdef cppclass Mesh:
        void *getPythonArray() except +


cdef extern from "PyPipeline.hpp" namespace "pdal::python":
    cdef cppclass PyPipelineExecutor:
        PyPipelineExecutor(const char*) except +
        PyPipelineExecutor(const char*, vector[Array*]&) except +
        int64_t execute() except +
        bool validate() except +
        string getPipeline() except +
        string getMetadata() except +
        string getSchema() except +
        string getLog() except +
        vector[Array*] getArrays() except +
        vector[Mesh*] getMeshes() except +
        int getLogLevel()
        void setLogLevel(int)


cdef class Pipeline:
    cdef PyPipelineExecutor* _executor
    cdef vector[Array *] _arrays;

    def __cinit__(self, unicode json, list arrays=None):
        if arrays is not None:
            for array in arrays:
                self._arrays.push_back(new Array(array))
            self._executor = new PyPipelineExecutor(json.encode('UTF-8'), self._arrays)
        else:
            self._executor = new PyPipelineExecutor(json.encode('UTF-8'))

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
            output = []
            v = self._executor.getArrays()
            cdef vector[Array*].iterator it = v.begin()
            cdef Array* ptr
            while it != v.end():
                ptr = deref(it)
                output.append(<object>ptr.getPythonArray())
                del ptr
                inc(it)
            return output

    property meshes:
        def __get__(self):
            output = []
            v = self._executor.getMeshes()
            cdef vector[Mesh *].iterator it = v.begin()
            cdef Mesh* ptr
            while it != v.end():
                ptr = deref(it)
                output.append(<object>ptr.getPythonArray())
                del ptr
                inc(it)
            return output

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
