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

#ifndef _WIN32
#include <dlfcn.h>
#endif

#include <Python.h>

#include <pdal/Stage.hpp>
#include <pdal/pdal_features.hpp>

namespace pdal
{
namespace python
{


void addArrayReaders(PipelineExecutor* executor, std::vector<std::shared_ptr<Array>> arrays)
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
    if (arrays.empty())
        return;

    PipelineManager& manager = executor->getManager();
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
        roots[0]->setInput(r);
    }

    manager.validateStageOptions();
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
    if (_import_array() < 0)
        throw pdal_error("Could not import numpy.core.multiarray.");

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
    if (_import_array() < 0)
        throw pdal_error("Could not import numpy.core.multiarray.");

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
