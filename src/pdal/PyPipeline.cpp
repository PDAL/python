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

#include "PyArray.hpp"
#include "PyPipeline.hpp"
#include <pdal/util/Utils.hpp>

#ifndef _WIN32
#include <dlfcn.h>
#endif

namespace pdal
{
namespace python
{


void CountPointTable::reset()
{
    for (PointId idx = 0; idx < numPoints(); idx++)
        if (!skip(idx))
            m_count++;
    FixedPointTable::reset();
}


PipelineExecutor::PipelineExecutor(
    std::string const& json, std::vector<std::shared_ptr<Array>> arrays, int level)
{
    if (level < 0 || level > 8)
        throw pdal_error("log level must be between 0 and 8!");

    LogPtr log(Log::makeLog("pypipeline", &m_logStream));
    log->setLevel(static_cast<pdal::LogLevel>(level));
    m_manager.setLog(log);

    std::stringstream strm;
    strm << json;
    m_manager.readPipeline(strm);

    addArrayReaders(arrays);
}


point_count_t PipelineExecutor::execute()
{
    point_count_t count = m_manager.execute();
    m_executed = true;
    return count;
}

std::string PipelineExecutor::getSrsWKT2() const
{
    std::string output("");
    pdal::PointTableRef pointTable = m_manager.pointTable();


    pdal::SpatialReference srs = pointTable.spatialReference();
    output = srs.getWKT();

    return output;
}

point_count_t PipelineExecutor::executeStream(point_count_t streamLimit)
{
    CountPointTable table(streamLimit);
    m_manager.executeStream(table);
    m_executed = true;
    return table.count();
}

const PointViewSet& PipelineExecutor::views() const
{
    if (!m_executed)
        throw pdal_error("Pipeline has not been executed!");

    return m_manager.views();
}


std::string PipelineExecutor::getPipeline() const
{
    std::stringstream strm;
    pdal::PipelineWriter::writePipeline(m_manager.getStage(), strm);
    return strm.str();
}


std::string PipelineExecutor::getMetadata() const
{
    if (!m_executed)
        throw pdal_error("Pipeline has not been executed!");

    std::stringstream strm;
    MetadataNode root = m_manager.getMetadata().clone("metadata");
    pdal::Utils::toJSON(root, strm);
    return strm.str();
}


std::string PipelineExecutor::getSchema() const
{
    if (!m_executed)
        throw pdal_error("Pipeline has not been executed!");

    std::stringstream strm;
    MetadataNode root = pointTable().layout()->toMetadata().clone("schema");
    pdal::Utils::toJSON(root, strm);
    return strm.str();
}


MetadataNode computePreview(Stage* stage)
{
    if (!stage)
        throw pdal_error("no valid stage in QuickInfo");

    QuickInfo qi = stage->preview();
    if (!qi.valid())
        throw pdal_error("No summary data available for stage '" + stage->getName()+"'" );

    std::stringstream strm;
    MetadataNode summary(stage->getName());
    summary.add("num_points", qi.m_pointCount);
    if (qi.m_srs.valid())
    {
        MetadataNode srs = qi.m_srs.toMetadata();
        summary.add(srs);
    }
    if (qi.m_bounds.valid())
    {
        MetadataNode bounds = Utils::toMetadata(qi.m_bounds);
        summary.add(bounds.clone("bounds"));
    }

    std::string dims;
    auto di = qi.m_dimNames.begin();
    while (di != qi.m_dimNames.end())
    {
        dims += *di;
        ++di;
        if (di != qi.m_dimNames.end())
           dims += ", ";
    }
    if (dims.size())
        summary.add("dimensions", dims);
    pdal::Utils::toJSON(summary, strm);
    return summary;

}


std::string PipelineExecutor::getQuickInfo() const
{

    Stage* stage(nullptr);
    std::vector<Stage *> stages = m_manager.stages();
    std::vector<Stage *> previewStages;

    for (auto const& s: stages)
    {
        auto n = s->getName();
        auto v = pdal::Utils::split2(n,'.');
        if (v.size() > 0)
            if (pdal::Utils::iequals(v[0], "readers"))
                previewStages.push_back(s);
    }

    MetadataNode summary;
    for (auto const& stage: previewStages)
    {
        MetadataNode n = computePreview(stage);
        summary.add(n);
    }

    std::stringstream strm;
    pdal::Utils::toJSON(summary, strm);
    return strm.str();
}

void PipelineExecutor::addArrayReaders(std::vector<std::shared_ptr<Array>> arrays)
{

    if (arrays.empty())
        return;

    std::vector<Stage *> roots = m_manager.roots();
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

        Stage& s = m_manager.makeReader("", "readers.memoryview", options);
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
        roots[0]->setInput(r);
    }

    m_manager.validateStageOptions();
}


PyObject* buildNumpyDescriptor(PointLayoutPtr layout)
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

    // Ensure that the dimensions are sorted by offset
    // Is there a better way? Can they be sorted by offset already?
    auto sortByOffset = [layout](Dimension::Id id1, Dimension::Id id2) -> bool
    {
        return layout->dimOffset(id1) < layout->dimOffset(id2);
    };
    auto dims = layout->dims();
    std::sort(dims.begin(), dims.end(), sortByOffset);

    PyObject* names = PyList_New(dims.size());
    PyObject* formats = PyList_New(dims.size());
    for (size_t i = 0; i < dims.size(); ++i)
    {
        Dimension::Id id = dims[i];
        auto name = layout->dimName(id);
        PyList_SetItem(names, i, PyUnicode_FromString(name.c_str()));

        std::stringstream format;
        switch (Dimension::base(layout->dimType(id)))
        {
            case Dimension::BaseType::Unsigned:
                format << 'u';
                break;
            case Dimension::BaseType::Signed:
                format << 'i';
                break;
            case Dimension::BaseType::Floating:
                format << 'f';
                break;
            default:
                throw pdal_error("Unable to map dimension '" + name  + "' to Numpy");
        }
        format << layout->dimSize(id);
        PyList_SetItem(formats, i, PyUnicode_FromString(format.str().c_str()));

    }
    PyObject* dtype_dict = PyDict_New();
    PyDict_SetItemString(dtype_dict, "names", names);
    PyDict_SetItemString(dtype_dict, "formats", formats);
    return dtype_dict;
}


PyArrayObject* viewToNumpyArray(PointViewPtr view)
{

    PyObject* dtype_dict = buildNumpyDescriptor(view->layout());
    PyArray_Descr *dtype = nullptr;
    if (PyArray_DescrConverter(dtype_dict, &dtype) == NPY_FAIL)
        throw pdal_error("Unable to build numpy dtype");
    Py_XDECREF(dtype_dict);

    // This is a 1 x size array.
    npy_intp size = view->size();
    PyArrayObject* array = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
            1, &size, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

    // copy the data
    DimTypeList types = view->dimTypes();
    for (PointId idx = 0; idx < view->size(); idx++)
        view->getPackedPoint(types, idx, (char *)PyArray_GETPTR1(array, idx));
    return array;
}


PyArrayObject* meshToNumpyArray(const TriangularMesh* mesh)
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
    PyObject* names = PyList_New(3);
    PyList_SetItem(names, 0, PyUnicode_FromString("A"));
    PyList_SetItem(names, 1, PyUnicode_FromString("B"));
    PyList_SetItem(names, 2, PyUnicode_FromString("C"));

    PyObject* formats = PyList_New(3);
    PyList_SetItem(formats, 0, PyUnicode_FromString("u4"));
    PyList_SetItem(formats, 1, PyUnicode_FromString("u4"));
    PyList_SetItem(formats, 2, PyUnicode_FromString("u4"));

    PyObject* dtype_dict = PyDict_New();
    PyDict_SetItemString(dtype_dict, "names", names);
    PyDict_SetItemString(dtype_dict, "formats", formats);

    PyArray_Descr *dtype = nullptr;
    if (PyArray_DescrConverter(dtype_dict, &dtype) == NPY_FAIL)
        throw pdal_error("Unable to build numpy dtype");
    Py_XDECREF(dtype_dict);

    // This is a 1 x size array.
    npy_intp size = mesh ? mesh->size() : 0;
    PyArrayObject* array = (PyArrayObject*)PyArray_NewFromDescr(&PyArray_Type, dtype,
            1, &size, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);
    for (PointId idx = 0; idx < size; idx++)
    {
        char* p = (char *)PyArray_GETPTR1(array, idx);
        const Triangle& t = (*mesh)[idx];
        uint32_t a = (uint32_t)t.m_a;
        std::memcpy(p, &a, 4);
        uint32_t b = (uint32_t)t.m_b;
        std::memcpy(p + 4, &b, 4);
        uint32_t c = (uint32_t)t.m_c;
        std::memcpy(p + 8, &c,  4);
    }
    return array;
}

} // namespace python
} // namespace pdal
