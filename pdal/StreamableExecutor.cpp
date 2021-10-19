/******************************************************************************
* Copyright (c) 2016, Howard Butler (howard@hobu.co)
*
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following
* conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in
*       the documentation and/or other materials provided
*       with the distribution.
*     * Neither the name of Hobu, Inc. or Flaxen Geo Consulting nor the
*       names of its contributors may be used to endorse or promote
*       products derived from this software without specific prior
*       written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
* AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
* OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
****************************************************************************/

#include "StreamableExecutor.hpp"

#include <Python.h>
#include <numpy/arrayobject.h>

#include <pdal/Stage.hpp>
#include <pdal/pdal_features.hpp>

namespace pdal
{
namespace python
{

// PythonPointTable

PythonPointTable::PythonPointTable(point_count_t limit, int prefetch) :
    StreamPointTable(m_layout, limit), m_limit(limit), m_prefetch(prefetch),
    m_curArray(nullptr), m_dtype(nullptr)
{}

PythonPointTable::~PythonPointTable()
{
    py_destroy();
}

void PythonPointTable::finalize()
{
    BasePointTable::finalize();
    py_createDescriptor();
    m_curArray = py_createArray();
}

void PythonPointTable::py_destroy()
{
    auto gil = PyGILState_Ensure();

    Py_XDECREF(m_dtype);
    Py_XDECREF(m_curArray);

    PyGILState_Release(gil);
}

PyArrayObject *PythonPointTable::py_createArray() const
{
    auto gil = PyGILState_Ensure();

    npy_intp size = (npy_intp)m_limit;
    Py_INCREF(m_dtype);
    PyArrayObject *arr = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, m_dtype,
        1, &size, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

    PyGILState_Release(gil);
    return arr;
}

void PythonPointTable::py_createDescriptor()
{
    auto gil = PyGILState_Ensure();

    if (_import_array() < 0)
        std::cerr << "Could not import array!\n";
    PyObject *dtype_dict = py_buildNumpyDescriptor();
    if (PyArray_DescrConverter(dtype_dict, &m_dtype) == NPY_FAIL)
        m_dtype = nullptr;
    Py_XDECREF(dtype_dict);

    PyGILState_Release(gil);
}

void PythonPointTable::py_resizeArray(int np)
{
    npy_intp sizes[1];
    sizes[0] = np;
    PyArray_Dims dims{ sizes, 1 };

    auto gil = PyGILState_Ensure();
    PyArray_Resize(m_curArray, &dims, true, NPY_CORDER);
    PyGILState_Release(gil);
}

PyObject *PythonPointTable::py_buildNumpyDescriptor() const
{
    // Build up a numpy dtype dictionary
    //
    // {'formats': ['f8', 'f8', 'f8', 'u2', 'u1', 'u1', 'u1', 'u1', 'u1',
    //              'f4', 'u1', 'u2', 'f8', 'u2', 'u2', 'u2'],
    // 'names': ['X', 'Y', 'Z', 'Intensity', 'ReturnNumber',
    //           'NumberOfReturns', 'ScanDirectionFlag', 'EdgeOfFlightLine',
    //           'Classification', 'ScanAngleRank', 'UserData',
    //           'PointSourceId', 'GpsTime', 'Red', 'Green', 'Blue']}
    //

    DimTypeList dims = layout()->dimTypes();
    PyObject* names = PyList_New(dims.size());
    PyObject* formats = PyList_New(dims.size());
    for (size_t i = 0; i < dims.size(); ++i)
    {
        DimType& dt = dims[i];
        std::string name = m_layout.dimName(dt.m_id);
        npy_intp stride = Dimension::size(dt.m_type);

        std::string kind;
        Dimension::BaseType b = Dimension::base(dt.m_type);
        if (b == Dimension::BaseType::Unsigned)
            kind = "u";
        else if (b == Dimension::BaseType::Signed)
            kind = "i";
        else if (b == Dimension::BaseType::Floating)
            kind = "f";
        else
            throw pdal_error("Unable to map kind '" + kind  +
                "' to PDAL dimension type");

        std::string type = kind + std::to_string(stride);
        PyList_SetItem(names, i, PyUnicode_FromString(name.c_str()));
        PyList_SetItem(formats, i, PyUnicode_FromString(type.c_str()));
    }

    PyObject* dict = PyDict_New();
    PyDict_SetItemString(dict, "names", names);
    PyDict_SetItemString(dict, "formats", formats);
    return dict;
}

void PythonPointTable::reset()
{
    point_count_t np = numPoints();

    // If this is the last chunk, the size might be less than what's expected, so
    // resize the array to match its true size.
    // ABELL - This isn't quite right. We want to know if there are any skips and deal with those
    // but I'm leaving that for the moment.
    if (np && np != m_limit)
        py_resizeArray(np);

    // This will keep putting arrays on the list until done, whether or not the consumer
    // can handle them that fast. We can modify as appropriate to block if desired.
    std::unique_lock<std::mutex> l(m_mutex);
    {
        // It's possible that this is called with 0 points processed, in which case
        // we don't push the current array.
        if (np)
        {
            m_arrays.push(m_curArray);
            m_curArray = nullptr;
        }

        bool done = np < m_limit;

        // If we just pushed the last chunk, push a nullptr so that a reader knows.
        if (done)
            m_arrays.push(nullptr);
        m_producedCv.notify_one();

        if (done)
            return;

        while (m_arrays.size() > m_prefetch)
            m_consumedCv.wait(l);
    }

    // Make a new array for data.
    m_curArray = py_createArray();
}

PyArrayObject *PythonPointTable::fetchArray()
{
    PyArrayObject *arr = nullptr;

    // Lock scope.
    Py_BEGIN_ALLOW_THREADS
    {
        std::unique_lock<std::mutex> l(m_mutex);
        while (m_arrays.empty())
            m_producedCv.wait(l);

        // Grab the array from the front of the list and notify that we did so.
        arr = m_arrays.front();
        m_arrays.pop();
    }
    Py_END_ALLOW_THREADS
    // Notify that we consumed an array.
    m_consumedCv.notify_one();
    return arr;
}

char *PythonPointTable::getPoint(PointId idx)
{
    return (char *)PyArray_GETPTR1(m_curArray, (npy_intp)idx);
}


// StreamableExecutor

StreamableExecutor::StreamableExecutor(const char *json, int chunkSize, int prefetch) :
    m_json(json), m_table(chunkSize, prefetch), m_manager(chunkSize),
    m_log(Log::makeLog("pdal_python", &m_logStream))
{
    m_manager.setLog(m_log);
}
    
StreamableExecutor::~StreamableExecutor()
{
    //ABELL - Hmmm.
    if (m_thread)
        m_thread->join();
}

PyArrayObject *StreamableExecutor::executeNext()
{
    if (!m_thread)
    {
        m_thread.reset(new std::thread([this]()
        {
            m_manager.executeStream(m_table);
        }));
    }

    // Blocks until something is ready.
    PyArrayObject *arr = m_table.fetchArray();
    if (arr == nullptr)
    {
        Py_BEGIN_ALLOW_THREADS
        m_thread->join();
        Py_END_ALLOW_THREADS
        m_thread.reset();
    }
    return arr;
}

std::string StreamableExecutor::getMetadata() const
{
    return pdal::Utils::toJSON(m_manager.getMetadata().clone("metadata"));
}

std::string StreamableExecutor::getPipeline() const
{
    std::stringstream strm;
    pdal::PipelineWriter::writePipeline(m_manager.getStage(), strm);
    return strm.str();
}

int StreamableExecutor::getLogLevel() const
{
    return static_cast<int>(m_log->getLevel());
}

std::string StreamableExecutor::getLog() const
{
    return m_logStream.str();
}

// Returns the active log level.
int StreamableExecutor::setLogLevel(int level)
{
    if (level < 0 || level > 8)
        return getLogLevel();

    m_log->setLevel(static_cast<pdal::LogLevel>(level));
    return level;
}

bool StreamableExecutor::validate()
{
    std::stringstream strm;
    strm << m_json;
    m_manager.readPipeline(strm);
    m_manager.prepare();

    return true;
}

} // namespace python
} // namespace pdal
