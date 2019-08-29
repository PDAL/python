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

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <Python.h>
#include <numpy/arrayobject.h>

#include <pdal/Stage.hpp>
#include <pdal/pdal_features.hpp>

#include "PyArray.hpp"

namespace pdal
{
namespace python
{

// Create a pipeline for writing data to PDAL
Pipeline::Pipeline(std::string const& json, std::vector<Array*> arrays) :
    m_executor(new PipelineExecutor(json))
{
#ifndef _WIN32
    // See comment in alternate constructor below.
    ::dlopen("libpdal_base.so", RTLD_NOLOAD | RTLD_GLOBAL);
#endif

    if (_import_array() < 0)
        throw pdal_error("Could not impory numpy.core.multiarray.");

    PipelineManager& manager = m_executor->getManager();

    std::stringstream strm(json);
    manager.readPipeline(strm);
    std::vector<Stage *> roots = manager.roots();
    if (roots.size() != 1)
        throw pdal_error("Filter pipeline must contain a single root stage.");

    for (auto array : arrays)
    {
        // Create numpy reader for each array
        // Options

        Options options;
        options.add("order", array->rowMajor() ?
            MemoryViewReader::Order::RowMajor :
            MemoryViewReader::Order::ColumnMajor);
        options.add("shape", MemoryViewReader::Shape(array->shape()));

        Stage& s = manager.makeReader("", "readers.memoryview", options);
        MemoryViewReader& r = dynamic_cast<MemoryViewReader &>(s);
        for (auto f : array->fields())
            r.pushField(f);

        ArrayIter& iter = array->iterator();
        auto incrementer = [&iter](PointId id) -> char *
        {
            if (! iter)
                return nullptr;

            char *c = *iter;
            ++iter;
            return c;
        };

        r.setIncrementer(incrementer);
        PyObject* parray = (PyObject*)array->getPythonArray();
        if (!parray)
            throw pdal_error("array was none!");

        roots[0]->setInput(r);
    }

    manager.validateStageOptions();
}

// Create a pipeline for reading data from PDAL
Pipeline::Pipeline(std::string const& json) :
    m_executor(new PipelineExecutor(json))
{
    // Make the symbols in pdal_base global so that they're accessible
    // to PDAL plugins.  Python dlopen's this extension with RTLD_LOCAL,
    // which means that without this, symbols in libpdal_base aren't available
    // for resolution of symbols on future runtime linking.  This is an issue
    // on Alpine and other Linux variants that don't use UNIQUE symbols
    // for C++ template statics only.  Without this, you end up with multiple
    // copies of template statics.
#ifndef _WIN32
    ::dlopen("libpdal_base.so", RTLD_NOLOAD | RTLD_GLOBAL);
#endif
    if (_import_array() < 0)
        throw pdal_error("Could not impory numpy.core.multiarray.");
}

Pipeline::~Pipeline()
{}


void Pipeline::setLogLevel(int level)
{
    m_executor->setLogLevel(level);
}


int Pipeline::getLogLevel() const
{
    return static_cast<int>(m_executor->getLogLevel());
}


int64_t Pipeline::execute()
{
    return m_executor->execute();
}

bool Pipeline::validate()
{
    auto res =  m_executor->validate();
    return res;
}

std::vector<Array *> Pipeline::getArrays() const
{
    std::vector<Array *> output;

    if (!m_executor->executed())
        throw python_error("call execute() before fetching arrays");

    const PointViewSet& pvset = m_executor->getManagerConst().views();

    for (auto i: pvset)
    {
        //ABELL - Leak?
        Array *array = new python::Array;
        array->update(i);
        output.push_back(array);
    }
    return output;
}

} // namespace python
} // namespace pdal

