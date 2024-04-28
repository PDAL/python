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

#include "PyPipeline.hpp"
#include "StreamableExecutor.hpp"

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PDAL_ARRAY_API

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
    StreamPointTable(m_layout, limit), m_prefetch(prefetch),
    m_curArray(nullptr), m_dtype(nullptr)
{}

PythonPointTable::~PythonPointTable()
{
    auto gil = PyGILState_Ensure();
    Py_XDECREF(m_dtype);
    Py_XDECREF(m_curArray);
    PyGILState_Release(gil);
}

void PythonPointTable::finalize()
{
    BasePointTable::finalize();

    // create dtype
    auto gil = PyGILState_Ensure();

    PyObject *dtype_dict = buildNumpyDescriptor(&m_layout);
    if (PyArray_DescrConverter(dtype_dict, &m_dtype) == NPY_FAIL)
        throw pdal_error("Unable to create numpy dtype");
    Py_XDECREF(dtype_dict);
    PyGILState_Release(gil);

    py_createArray();
}

void PythonPointTable::py_createArray()
{
    auto gil = PyGILState_Ensure();
    npy_intp size = capacity();
    Py_INCREF(m_dtype);
    m_curArray = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, m_dtype,
        1, &size, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);
    PyGILState_Release(gil);
}

void PythonPointTable::py_resizeArray(point_count_t np)
{
    npy_intp sizes[1];
    sizes[0] = np;
    PyArray_Dims dims{ sizes, 1 };

    auto gil = PyGILState_Ensure();
    // copy the non-skipped elements to the beginning
    npy_intp dest_idx = 0;
    for (PointId src_idx = 0; src_idx < numPoints(); src_idx++)
        if (!skip(src_idx))
        {
            if (src_idx != dest_idx)
            {
                PyObject* src_item = PyArray_GETITEM(m_curArray, (const char*) PyArray_GETPTR1(m_curArray, src_idx));
                PyArray_SETITEM(m_curArray, (char*) PyArray_GETPTR1(m_curArray, dest_idx), src_item);
                Py_XDECREF(src_item);
            }
            dest_idx++;
        }
    PyArray_Resize(m_curArray, &dims, true, NPY_CORDER);
    PyGILState_Release(gil);
}

void PythonPointTable::reset()
{
    point_count_t np = 0;
    for (PointId idx = 0; idx < numPoints(); idx++)
        if (!skip(idx))
            np++;

    if (np && np != capacity())
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
            py_createArray();
            m_producedCv.notify_one();
        }
        while (m_arrays.size() > m_prefetch)
            m_consumedCv.wait(l);
    }
}

void PythonPointTable::disable()
{
    // TODO: uncomment the next line when/if StreamPointTable.m_capacity
    // changes from private to protected
    // m_capacity = 0;
}

void PythonPointTable::done()
{
    m_arrays.push(nullptr);
    m_producedCv.notify_one();
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
    return (char *)PyArray_GETPTR1(m_curArray, idx);
}


// StreamableExecutor

StreamableExecutor::StreamableExecutor(std::string const& json,
                                       std::vector<std::shared_ptr<Array>> arrays,
                                       int level,
                                       point_count_t chunkSize,
                                       int prefetch)
    : PipelineExecutor(json, arrays, level)
    , m_table(chunkSize, prefetch)
    , m_exc(nullptr)
{
    m_thread.reset(new std::thread([this]()
    {
        try {
            m_manager.executeStream(m_table);
        } catch (...) {
            m_exc = std::current_exception();
        }
        m_table.done();
    }));
}

StreamableExecutor::~StreamableExecutor()
{
    if (!m_executed)
    {
        m_table.disable();
        auto gil = PyGILState_Ensure();
        while (PyArrayObject* arr = m_table.fetchArray())
            Py_XDECREF(arr);
        PyGILState_Release(gil);
    }
    Py_BEGIN_ALLOW_THREADS
    m_thread->join();
    Py_END_ALLOW_THREADS
}

PyArrayObject *StreamableExecutor::executeNext()
{
    PyArrayObject* arr = nullptr;
    if (!m_executed)
    {
        arr = m_table.fetchArray();
        if (arr == nullptr)
            m_executed = true;
        if (m_exc)
            std::rethrow_exception(m_exc);
    }
    return arr;
}

} // namespace python
} // namespace pdal
