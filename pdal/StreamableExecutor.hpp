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

#pragma once

#include <condition_variable>
#include <thread>

#include "PyPipeline.hpp"

namespace pdal
{
namespace python
{

class PythonPointTable : public StreamPointTable
{
public:
    PythonPointTable(point_count_t size, int prefetch);
    ~PythonPointTable();

    virtual void finalize();
    void disable();
    void done();
    PyArrayObject *fetchArray();

protected:
    virtual void reset();
    virtual char *getPoint(PointId idx);

private:
    // All functions starting with py_ call Python things that need the GIL locked.
    void py_createArray();
    void py_resizeArray(point_count_t np);

    int m_prefetch;
    PointLayout m_layout;
    PyArrayObject *m_curArray;
    PyArray_Descr *m_dtype;
    std::mutex m_mutex;
    std::condition_variable m_producedCv;
    std::condition_variable m_consumedCv;
    std::queue<PyArrayObject *> m_arrays;
};

class StreamableExecutor : public PipelineExecutor
{
public:
    StreamableExecutor(std::string const& json,
                       std::vector<std::shared_ptr<Array>> arrays,
                       int level,
                       point_count_t chunkSize,
                       int prefetch);
    ~StreamableExecutor();

    MetadataNode getMetadata() { return m_table.metadata(); }
    PyArrayObject* executeNext();

private:
    ConstPointTableRef pointTable() const { return m_table; }

    PythonPointTable m_table;
    std::unique_ptr<std::thread> m_thread;
    std::exception_ptr m_exc;
};

} // namespace python
} // namespace pdal
