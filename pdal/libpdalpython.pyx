# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport uint32_t, int64_t
from libcpp cimport bool
from cpython.version cimport PY_MAJOR_VERSION
cimport numpy as np
np.import_array()

from cpython cimport PyObject, Py_INCREF
from cython.operator cimport dereference as deref, preincrement as inc

cdef extern from "pdal/pdal_config.hpp" namespace "pdal::Config":
    cdef int versionMajor() except +
    cdef int versionMinor() except +
    cdef int versionPatch() except +
    cdef string sha1() except+
    cdef string debugInformation() except+
    cdef string pluginInstallPath() except+
    cdef string versionString() except+

def getVersionString():
    return versionString()
def getVersionMajor():
    return versionMajor()
def getVersionMinor():
    return versionMinor()
def getVersionPatch():
    return versionPatch()
def getSha1():
    return sha1()
def getDebugInformation():
    return debugInformation()
def getPluginInstallPath():
    return pluginInstallPath()

cdef extern from "PyArray.hpp" namespace "pdal::python":
    cdef cppclass Array:
        Array(np.ndarray) except +
        void *getPythonArray() except+

cdef extern from "PyPipeline.hpp" namespace "pdal::python":
    cdef cppclass Pipeline:
        Pipeline(const char* ) except +
        Pipeline(const char*, vector[Array*]& ) except +
        int64_t execute() except +
        bool validate() except +
        string getPipeline() except +
        string getMetadata() except +
        string getSchema() except +
        string getLog() except +
        vector[Array*] getArrays() except +
        int getLogLevel()
        void setLogLevel(int)

cdef class PyArray:
    cdef Array *thisptr
    def __cinit__(self, np.ndarray array):
        self.thisptr = new Array(array)
    def __dealloc__(self):
        del self.thisptr

cdef extern from "PyDimension.hpp":
    ctypedef struct Dimension:
        string name;
        string description;
        int size;
        string type;
##         string units; // Not defined by PDAL yet

    cdef vector[Dimension] getValidDimensions() except +


def getDimensions():
        cdef vector[Dimension] c_dims;
        c_dims = getValidDimensions()
        output = []
        cdef vector[Dimension].iterator it = c_dims.begin()
        while it != c_dims.end():
            ptr = deref(it)
            d = {}
            d['name'] = ptr.name
            d['description'] = ptr.description
            kind = ptr.type + str(ptr.size)
            d['dtype'] = np.dtype(kind)
            ptr = deref(it)
            output.append(d)
            inc(it)
        return output


cdef class PyPipeline:
    cdef Pipeline *thisptr      # hold a c++ instance which we're wrapping


    def __cinit__(self, unicode json, list arrays=None):
        cdef char* x = NULL
        cdef Py_ssize_t n_arrays;
        if arrays:
            n_arrays = len(arrays)

        cdef vector[Array*] c_arrays;
        cdef np.ndarray np_array;
        cdef Array* a

        if arrays is not None:
            for array in arrays:
                a = new Array(array)
                c_arrays.push_back(a)

            self.thisptr = new Pipeline(json.encode('UTF-8'), c_arrays)
        else:
            self.thisptr = new Pipeline(json.encode('UTF-8'))

    def __dealloc__(self):
        del self.thisptr

    property pipeline:
        def __get__(self):
            return self.thisptr.getPipeline()

    property metadata:
        def __get__(self):
            return self.thisptr.getMetadata()

    property loglevel:
        def __get__(self):
            return self.thisptr.getLogLevel()
        def __set__(self, v):
            self.thisptr.setLogLevel(v)

    property log:
        def __get__(self):

            return self.thisptr.getLog()

    property schema:
        def __get__(self):
            import json

            j = self.thisptr.getSchema()
            return json.loads(j)

    property arrays:

        def __get__(self):
            v = self.thisptr.getArrays()
            output = []
            cdef vector[Array*].iterator it = v.begin()
            cdef Array* a
            while it != v.end():
                ptr = deref(it)
                a = ptr#.get()
                o = a.getPythonArray()
                output.append(<object>o)
                del ptr
                inc(it)
            return output


    def execute(self):
        if not self.thisptr:
            raise Exception("C++ Pipeline object not constructed!")
        return self.thisptr.execute()

    def validate(self):
        if not self.thisptr:
            raise Exception("C++ Pipeline object not constructed!")
        return self.thisptr.validate()
