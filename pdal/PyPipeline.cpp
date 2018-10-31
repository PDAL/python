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
#ifdef PDAL_HAVE_LIBXML2
#include <pdal/XMLSchema.hpp>
#endif

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include "PyArray.hpp"
#include <pdal/Stage.hpp>
#include <pdal/pdal_features.hpp>
#include <pdal/PipelineWriter.hpp>
#include <pdal/io/NumpyReader.hpp>

namespace libpdalpython
{

using namespace pdal::python;

Pipeline::Pipeline(std::string const& json, std::vector<Array*> arrays)
{

#ifndef _WIN32
    ::dlopen("libpdal_base.so", RTLD_NOLOAD | RTLD_GLOBAL);
    ::dlopen("libpdal_plugin_reader_numpy.so", RTLD_NOLOAD | RTLD_GLOBAL);
#endif

#undef NUMPY_IMPORT_ARRAY_RETVAL
#define NUMPY_IMPORT_ARRAY_RETVAL
    import_array();

    m_executor = std::shared_ptr<pdal::PipelineExecutor>(new pdal::PipelineExecutor(json));

    pdal::PipelineManager& manager = m_executor->getManager();

    std::stringstream strm(json);
    manager.readPipeline(strm);

    pdal::Stage *r = manager.getStage();
    if (!r)
        throw pdal::pdal_error("pipeline had no stages!");

#if PDAL_VERSION_MAJOR > 1 || PDAL_VERSION_MINOR >=8
    int counter = 1;
    for (auto array: arrays)
    {
        // Create numpy reader for each array
        pdal::Options options;
        std::stringstream tag;
        tag << "readers_numpy" << counter;
        pdal::StageCreationOptions opts { "", "readers.numpy", nullptr, options, tag.str()};
        pdal::Stage& reader = manager.makeReader(opts);

        pdal::NumpyReader* np_reader = dynamic_cast<pdal::NumpyReader*>(&reader);
        if (!np_reader)
            throw pdal::pdal_error("couldn't cast reader!");

        PyObject* parray = (PyObject*)array->getPythonArray();
        if (!parray)
            throw pdal::pdal_error("array was none!");

        np_reader->setArray(parray);

        r->setInput(reader);
        counter++;

    }
#endif

    manager.validateStageOptions();
}

Pipeline::Pipeline(std::string const& json)
{
    // Make the symbols in pdal_base global so that they're accessible
    // to PDAL plugins.  Python dlopen's this extension with RTLD_LOCAL,
    // which means that without this, symbols in libpdal_base aren't available
    // for resolution of symbols on future runtime linking.  This is an issue
    // on Apline and other Linux variants that doesn't use UNIQUE symbols
    // for C++ template statics. only
#ifndef _WIN32
    ::dlopen("libpdal_base.so", RTLD_NOLOAD | RTLD_GLOBAL);
#endif
#undef NUMPY_IMPORT_ARRAY_RETVAL
#define NUMPY_IMPORT_ARRAY_RETVAL
    import_array();

    m_executor = std::shared_ptr<pdal::PipelineExecutor>(new pdal::PipelineExecutor(json));
}

Pipeline::~Pipeline()
{
}

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

    int64_t count = m_executor->execute();
    return count;
}

bool Pipeline::validate()
{
    return m_executor->validate();
}

std::vector<Array *> Pipeline::getArrays() const
{
    std::vector<Array *> output;

    if (!m_executor->executed())
        throw python_error("call execute() before fetching arrays");

    const pdal::PointViewSet& pvset = m_executor->getManagerConst().views();

    for (auto i: pvset)
    {
        //ABELL - Leak?
        Array *array = new pdal::python::Array;
        array->update(i);
        output.push_back(array);
    }
    return output;
}
} //namespace libpdalpython

