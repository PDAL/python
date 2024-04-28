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

#include <pdal/PipelineManager.hpp>

#define NPY_TARGET_VERSION NPY_1_22_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PDAL_ARRAY_API

#include <numpy/arrayobject.h>

namespace pdal
{
namespace python
{

PyObject* buildNumpyDescriptor(PointLayoutPtr layout);
PyArrayObject* viewToNumpyArray(PointViewPtr view);
PyArrayObject* meshToNumpyArray(const TriangularMesh* mesh);

class Array;

class PDAL_DLL PipelineExecutor {
public:
    PipelineExecutor(std::string const& json, std::vector<std::shared_ptr<Array>> arrays, int level);
    virtual ~PipelineExecutor() = default;

    point_count_t execute();
    point_count_t executeStream(point_count_t streamLimit);

    const PointViewSet& views() const;
    std::string getPipeline() const;
    std::string getMetadata() const;
    std::string getQuickInfo() const;
    std::string getSchema() const;
    std::string getSrsWKT2() const;
    PipelineManager const& getManager() const { return m_manager; }
    std::string getLog() const { return m_logStream.str(); }

protected:
    virtual ConstPointTableRef pointTable() const { return m_manager.pointTable(); }

    pdal::PipelineManager m_manager;
    bool m_executed = false;

private:
    void addArrayReaders(std::vector<std::shared_ptr<Array>> arrays);

    std::stringstream m_logStream;
};

class CountPointTable : public FixedPointTable
{
public:
    CountPointTable(point_count_t capacity) : FixedPointTable(capacity), m_count(0) {}
    point_count_t count() const { return m_count; }

protected:
    virtual void reset();

private:
    point_count_t m_count;
};

} // namespace python
} // namespace pdal
