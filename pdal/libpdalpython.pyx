# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

import json
from types import SimpleNamespace

from cpython.ref cimport Py_DECREF
from libcpp cimport bool
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


cdef extern from "PyArray.hpp" namespace "pdal::python":
    cdef cppclass Array:
        Array(np.ndarray) except +


cdef extern from "pdal/PipelineExecutor.hpp" namespace "pdal":
    cdef cppclass PipelineExecutor:
        PipelineExecutor(const char*) except +
        bool executed() except +
        int execute() except +
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
    vector[np.PyArrayObject*] getArrays(const PipelineExecutor*) except +
    vector[np.PyArrayObject*] getMeshes(const PipelineExecutor*) except +


cdef class Pipeline:
    cdef PipelineExecutor* _executor
    cdef vector[Array*] _inputs

    def __dealloc__(self):
        self.inputs = []

    def __copy__(self):
        cdef Pipeline clone = self.__class__()
        clone._inputs = self._inputs
        return clone

    @property
    def inputs(self):
        raise AttributeError("unreadable attribute")

    @inputs.setter
    def inputs(self, ndarrays):
        self._inputs.clear()
        for ndarray in ndarrays:
            self._inputs.push_back(new Array(ndarray))
        self._delete_executor()

    @property
    def pipeline(self):
        return self._get_executor().getPipeline()

    @property
    def metadata(self):
        return self._get_executor().getMetadata()

    @property
    def loglevel(self):
        return self._get_executor().getLogLevel()

    @loglevel.setter
    def loglevel(self, level):
        self._get_executor().setLogLevel(level)

    @property
    def log(self):
        return self._get_executor().getLog()

    @property
    def schema(self):
        return json.loads(self._get_executor().getSchema())

    @property
    def arrays(self):
        if not self._get_executor().executed():
            raise RuntimeError("call execute() before fetching arrays")
        return _vector_to_list(getArrays(self._executor))

    @property
    def meshes(self):
        if not self._get_executor().executed():
            raise RuntimeError("call execute() before fetching the mesh")
        return _vector_to_list(getMeshes(self._executor))

    def execute(self):
        return self._get_executor().execute()

    def validate(self):
        return self._get_executor().validate()

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

    @property
    def _json(self):
        raise NotImplementedError("Abstract property")

    @property
    def _num_inputs(self):
        return self._inputs.size()

    def _delete_executor(self):
        if self._executor:
            del self._executor
            self._executor = NULL

    cdef PipelineExecutor* _get_executor(self) except NULL:
        if not self._executor:
            json_bytes = self._json.encode("UTF-8")
            self._executor = new PipelineExecutor(json_bytes)
            readPipeline(self._executor, json_bytes)
            addArrayReaders(self._executor, self._inputs)
        return self._executor


cdef _vector_to_list(vector[np.PyArrayObject*] arrays):
    output = []
    for array in arrays:
        output.append(<object>array)
        Py_DECREF(output[-1])
    return output
