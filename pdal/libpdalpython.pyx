# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

from types import SimpleNamespace

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


cdef extern from "pdal/PipelineManager.hpp" namespace "pdal":
    cdef cppclass PipelineManager:
        const PointViewSet& views() const


cdef extern from "pdal/PipelineExecutor.hpp" namespace "pdal":
    cdef cppclass PipelineExecutor:
        PipelineExecutor(string) except +
        const PipelineManager& getManagerConst() except +
        bool executed() except +
        void read() except +
        int execute() except +
        string getPipeline() except +
        string getMetadata() except +
        string getSchema() except +
        string getLog() except +
        void setLogLevel(int) except +


cdef extern from "StreamableExecutor.hpp" namespace "pdal::python":
    cdef cppclass StreamableExecutor(PipelineExecutor):
        StreamableExecutor(string, int, int) except +
        np.PyArrayObject* executeNext() except +
        void stop() except +


cdef extern from "PyArray.hpp" namespace "pdal::python":
    cdef cppclass Array:
        Array(np.PyArrayObject*) except +


cdef extern from "PyPipeline.hpp" namespace "pdal::python":
    void addArrayReaders(PipelineExecutor*, vector[shared_ptr[Array]]) except +
    np.PyArrayObject* viewToNumpyArray(PointViewPtr) except +
    np.PyArrayObject* meshToNumpyArray(const TriangularMesh*) except +


cdef class Pipeline:
    cdef unique_ptr[StreamableExecutor] _executor
    cdef vector[shared_ptr[Array]] _inputs
    cdef int _loglevel
    cdef int _chunk_size
    cdef int _prefetch

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

    @property
    def chunk_size(self):
        return self._chunk_size

    @chunk_size.setter
    def chunk_size(self, value):
        assert value > 0
        self._chunk_size = value
        self._del_executor()

    @property
    def prefetch(self):
        return self._prefetch

    @prefetch.setter
    def prefetch(self, value):
        assert value >= 0
        self._prefetch = value
        self._del_executor()

    #========= readable properties to be read after execution ============================

    @property
    def log(self):
        return self._get_executor().getLog()

    @property
    def schema(self):
        return self._get_executor().getSchema()

    @property
    def pipeline(self):
        return self._get_executor().getPipeline()

    @property
    def metadata(self):
        return self._get_executor().getMetadata()

    @property
    def arrays(self):
        cdef PipelineExecutor* executor = self._get_executor()
        if not executor.executed():
            raise RuntimeError("call execute() before fetching arrays")
        output = []
        for view in executor.getManagerConst().views():
            output.append(<object>viewToNumpyArray(view))
            Py_DECREF(output[-1])
        return output

    @property
    def meshes(self):
        cdef PipelineExecutor* executor = self._get_executor()
        if not executor.executed():
            raise RuntimeError("call execute() before fetching the mesh")
        output = []
        for view in executor.getManagerConst().views():
            output.append(<object>meshToNumpyArray(deref(view).mesh()))
            Py_DECREF(output[-1])
        return output

    #========= execution methods =========================================================

    def execute(self):
        return self._get_executor().execute()

    def __iter__(self):
        cdef StreamableExecutor* executor = self._get_executor()
        try:
            while True:
                arr_ptr = executor.executeNext()
                if arr_ptr is NULL:
                    break
                arr = <object>arr_ptr
                Py_DECREF(arr)
                yield arr
        finally:
            executor.stop()

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

    cdef StreamableExecutor* _get_executor(self) except NULL:
        if not self._executor:
            executor = new StreamableExecutor(self._get_json(),
                                              self._chunk_size, self._prefetch)
            executor.setLogLevel(self._loglevel)
            executor.read()
            addArrayReaders(executor, self._inputs)
            self._executor.reset(executor)
        return self._executor.get()
