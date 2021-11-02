# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

import json
from types import SimpleNamespace

cimport cython
from cython.operator cimport dereference as deref
from cpython.ref cimport Py_DECREF
from libcpp cimport bool
from libcpp.memory cimport make_shared, shared_ptr, unique_ptr
from libcpp.set cimport set as cpp_set
from libcpp.string cimport string
from libcpp.vector cimport vector

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


cdef extern from "pdal/StageFactory.hpp" namespace "pdal":
    cpdef cppclass StageFactory:
        @staticmethod
        string inferReaderDriver(string)
        @staticmethod
        string inferWriterDriver(string)


def infer_reader_driver(driver):
    return StageFactory.inferReaderDriver(driver)

def infer_writer_driver(driver):
    return StageFactory.inferWriterDriver(driver)


cdef extern from "pdal/Mesh.hpp" namespace "pdal":
    cdef cppclass TriangularMesh:
        pass


cdef extern from "pdal/PointView.hpp" namespace "pdal":
    cdef cppclass PointView:
        TriangularMesh* mesh()

    ctypedef shared_ptr[PointView] PointViewPtr
    ctypedef cpp_set[PointViewPtr] PointViewSet


cdef extern from "PyPipeline.hpp" namespace "pdal::python":
    np.PyArrayObject* viewToNumpyArray(PointViewPtr) except +
    np.PyArrayObject* meshToNumpyArray(const TriangularMesh*) except +

    cdef cppclass PipelineExecutor:
        PipelineExecutor(string, vector[shared_ptr[Array]], int) except +
        int execute() except +
        bool executed() except +
        string getPipeline() except +
        string getMetadata() except +
        string getSchema() except +
        string getLog() except +
        const PointViewSet& views() except +


cdef extern from "StreamableExecutor.hpp" namespace "pdal::python":
    cdef cppclass StreamableExecutor(PipelineExecutor):
        StreamableExecutor(string, vector[shared_ptr[Array]], int, int, int) except +
        np.PyArrayObject* executeNext() except +
        void stop() except +


cdef extern from "PyArray.hpp" namespace "pdal::python":
    cdef cppclass Array:
        Array(np.PyArrayObject*) except +


@cython.internal
cdef class PipelineResultsMixin:
    @property
    def log(self):
        return self._get_executor().getLog()

    @property
    def schema(self):
        return json.loads(self._get_executor().getSchema())

    @property
    def pipeline(self):
        return self._get_executor().getPipeline()

    @property
    def metadata(self):
        return self._get_executor().getMetadata()

    cdef PipelineExecutor* _get_executor(self) except NULL:
        raise NotImplementedError("Abstract method")



cdef class Pipeline(PipelineResultsMixin):
    cdef unique_ptr[PipelineExecutor] _executor
    cdef vector[shared_ptr[Array]] _inputs
    cdef int _loglevel

    #========= writeable properties to be set before execution ===========================

    @property
    def inputs(self):
        raise AttributeError("unreadable attribute")

    @inputs.setter
    def inputs(self, ndarrays):
        self._inputs.clear()
        for ndarray in ndarrays:
            self._inputs.push_back(make_shared[Array](<np.PyArrayObject*>ndarray))
        self._del_executor()

    @property
    def loglevel(self):
        return self._loglevel

    @loglevel.setter
    def loglevel(self, value):
        self._loglevel = value
        self._del_executor()

    #========= readable properties to be read after execution ============================

    @property
    def arrays(self):
        cdef PipelineExecutor* executor = self._get_executor()
        output = []
        for view in executor.views():
            output.append(<object>viewToNumpyArray(view))
            Py_DECREF(output[-1])
        return output

    @property
    def meshes(self):
        cdef PipelineExecutor* executor = self._get_executor()
        output = []
        for view in executor.views():
            output.append(<object>meshToNumpyArray(deref(view).mesh()))
            Py_DECREF(output[-1])
        return output

    #========= execution methods =========================================================

    def execute(self):
        return self._get_executor().execute()

    def iterator(self, int chunk_size=10000, int prefetch = 0):
        it = PipelineIterator()
        it.set_executor(new StreamableExecutor(
            self._get_json(), self._inputs, self._loglevel, chunk_size, prefetch
        ))
        return it

    #========= non-public properties & methods ===========================================

    def _get_json(self):
        raise NotImplementedError("Abstract method")

    @property
    def _has_inputs(self):
        return not self._inputs.empty()

    def _copy_inputs(self, Pipeline other):
        self._inputs = other._inputs

    def _del_executor(self):
        self._executor.reset()

    cdef PipelineExecutor* _get_executor(self) except NULL:
        if not self._executor:
            self._executor.reset(new PipelineExecutor(
                self._get_json(), self._inputs, self._loglevel)
            )
        return self._executor.get()


@cython.internal
cdef class PipelineIterator(PipelineResultsMixin):
    cdef unique_ptr[StreamableExecutor] _executor

    cdef set_executor(self, StreamableExecutor* executor):
        self._executor.reset(executor)

    def __iter__(self):
        return self

    def __next__(self):
        cdef StreamableExecutor* executor = self._executor.get()
        if executor.executed():
            raise StopIteration

        arr_ptr = executor.executeNext()
        if arr_ptr is NULL:
            executor.stop()
            raise StopIteration

        arr = <object> arr_ptr
        Py_DECREF(arr)
        return arr

    cdef PipelineExecutor* _get_executor(self) except NULL:
        return self._executor.get()
