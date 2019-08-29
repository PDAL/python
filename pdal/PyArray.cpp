/******************************************************************************
* Copyright (c) 2019, Hobu Inc. (info@hobu.co)
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
#include <pdal/io/MemoryViewReader.hpp>

#include <numpy/arrayobject.h>

namespace pdal
{
namespace python
{

namespace
{

Dimension::Type pdalType(int t)
{
    using namespace Dimension;

    switch (t)
    {
    case NPY_FLOAT32:
        return Type::Float;
    case NPY_FLOAT64:
        return Type::Double;
    case NPY_INT8:
        return Type::Signed8;
    case NPY_INT16:
        return Type::Signed16;
    case NPY_INT32:
        return Type::Signed32;
    case NPY_INT64:
        return Type::Signed64;
    case NPY_UINT8:
        return Type::Unsigned8;
    case NPY_UINT16:
        return Type::Unsigned16;
    case NPY_UINT32:
        return Type::Unsigned32;
    case NPY_UINT64:
        return Type::Unsigned64;
    default:
        return Type::None;
    }
    assert(0);

    return Type::None;
}

std::string toString(PyObject *pname)
{
    PyObject* r = PyObject_Str(pname);
    if (!r)
        throw pdal_error("couldn't make string representation value");
    Py_ssize_t size;
    return std::string(PyUnicode_AsUTF8AndSize(r, &size));
}

} // unnamed namespace

Array::Array() : m_array(nullptr)
{
    if (_import_array() < 0)
        throw pdal_error("Could not import numpy.core.multiarray.");
}

Array::Array(PyArrayObject* array) : m_array(array), m_rowMajor(true)
{
    if (_import_array() < 0)
        throw pdal_error("Could not import numpy.core.multiarray.");

    Py_XINCREF(array);

    PyArray_Descr *dtype = PyArray_DTYPE(m_array);
    npy_intp ndims = PyArray_NDIM(m_array);
    npy_intp *shape = PyArray_SHAPE(m_array);
    int numFields = (dtype->fields == Py_None) ?
        0 :
        static_cast<int>(PyDict_Size(dtype->fields));

    int xyz = 0;
    if (numFields == 0)
    {
        if (ndims != 3)
            throw pdal_error("Array without fields must have 3 dimensions.");
        m_fields.push_back({"Intensity", pdalType(dtype->type_num), 0});
    }
    else
    {
        PyObject *names_dict = dtype->fields;
        PyObject *names = PyDict_Keys(names_dict);
        PyObject *values = PyDict_Values(names_dict);
        if (!names || !values)
            throw pdal_error("Bad field specification in numpy array.");

        for (int i = 0; i < numFields; ++i)
        {
            std::string name = toString(PyList_GetItem(names, i));
            if (name == "X")
                xyz |= 1;
            else if (name == "Y")
                xyz |= 2;
            else if (name == "Z")
                xyz |= 4;
            PyObject *tup = PyList_GetItem(values, i);

            // Get offset.
            size_t offset = PyLong_AsLong(PySequence_Fast_GET_ITEM(tup, 1));

            // Get type.
            PyArray_Descr *descriptor =
                (PyArray_Descr *)PySequence_Fast_GET_ITEM(tup, 0);
            Dimension::Type type = pdalType(descriptor->type_num);
            if (type == Dimension::Type::None)
                throw pdal_error("Incompatible type for field '" + name + "'.");

            m_fields.push_back({name, type, offset});
        }

        if (xyz != 0 && xyz != 7)
            throw pdal_error("Array fields must contain all or none "
                "of X, Y and Z");
        if (xyz == 0 && ndims != 3)
            throw pdal_error("Array without named X/Y/Z fields "
                    "must have three dimensions.");
    }
    if (xyz == 0)
        m_shape = { (size_t)shape[0], (size_t)shape[1], (size_t)shape[2] };
    m_rowMajor = !(PyArray_FLAGS(m_array) & NPY_ARRAY_F_CONTIGUOUS);
}

Array::~Array()
{
    if (m_array)
        Py_XDECREF((PyObject *)m_array);
}


void Array::update(PointViewPtr view)
{
    if (m_array)
        Py_XDECREF((PyObject *)m_array);
    m_array = nullptr;  // Just in case of an exception.

    Dimension::IdList dims = view->dims();
    npy_intp size = view->size();

    PyObject *dtype_dict = (PyObject*)buildNumpyDescription(view);
    if (!dtype_dict)
        throw pdal_error("Unable to build numpy dtype "
                "description dictionary");

    PyArray_Descr *dtype = nullptr;
    if (PyArray_DescrConverter(dtype_dict, &dtype) == NPY_FAIL)
        throw pdal_error("Unable to build numpy dtype");
    Py_XDECREF(dtype_dict);

    // This is a 1 x size array.
    m_array = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
            1, &size, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

    // copy the data
    DimTypeList types = view->dimTypes();
    for (PointId idx = 0; idx < view->size(); idx++)
    {
        char *p = (char *)PyArray_GETPTR1(m_array, idx);
        view->getPackedPoint(types, idx, p);
    }
}


//ABELL - Who's responsible for incrementing the ref count?
PyArrayObject *Array::getPythonArray() const
{
    return m_array;
}

PyObject* Array::buildNumpyDescription(PointViewPtr view) const
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

    Dimension::IdList dims = view->dims();

    PyObject* dict = PyDict_New();
    PyObject* sizes = PyList_New(dims.size());
    PyObject* formats = PyList_New(dims.size());
    PyObject* titles = PyList_New(dims.size());

    for (size_t i = 0; i < dims.size(); ++i)
    {
        Dimension::Id id = dims[i];
        Dimension::Type t = view->dimType(id);
        npy_intp stride = view->dimSize(id);

        std::string name = view->dimName(id);

        std::string kind("i");
        Dimension::BaseType b = Dimension::base(t);
        if (b == Dimension::BaseType::Unsigned)
            kind = "u";
        else if (b == Dimension::BaseType::Signed)
            kind = "i";
        else if (b == Dimension::BaseType::Floating)
            kind = "f";
        else
            throw pdal_error("Unable to map kind '" + kind  +
                "' to PDAL dimension type");

        std::stringstream oss;
        oss << kind << stride;
        PyObject* pySize = PyLong_FromLong(stride);
        PyObject* pyTitle = PyUnicode_FromString(name.c_str());
        PyObject* pyFormat = PyUnicode_FromString(oss.str().c_str());

        PyList_SetItem(sizes, i, pySize);
        PyList_SetItem(titles, i, pyTitle);
        PyList_SetItem(formats, i, pyFormat);
    }

    PyDict_SetItemString(dict, "names", titles);
    PyDict_SetItemString(dict, "formats", formats);

    return dict;
}

bool Array::rowMajor() const
{
    return m_rowMajor;
}

Array::Shape Array::shape() const
{
    return m_shape;
}

const Array::Fields& Array::fields() const
{
    return m_fields;
}

ArrayIter& Array::iterator()
{
    ArrayIter *it = new ArrayIter(*this);
    m_iterators.push_back(std::unique_ptr<ArrayIter>(it));
    return *it;
}

ArrayIter::ArrayIter(Array& array)
{
    m_iter = NpyIter_New(array.getPythonArray(),
        NPY_ITER_EXTERNAL_LOOP | NPY_ITER_READONLY | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    if (!m_iter)
        throw pdal_error("Unable to create numpy iterator.");

    char *itererr;
    m_iterNext = NpyIter_GetIterNext(m_iter, &itererr);
    if (!m_iterNext)
    {
        NpyIter_Deallocate(m_iter);
        throw pdal_error(std::string("Unable to create numpy iterator: ") +
            itererr);
    }
    m_data = NpyIter_GetDataPtrArray(m_iter);
    m_stride = NpyIter_GetInnerStrideArray(m_iter);
    m_size = NpyIter_GetInnerLoopSizePtr(m_iter);
    m_done = false;
}

ArrayIter::~ArrayIter()
{
    NpyIter_Deallocate(m_iter);
}

ArrayIter& ArrayIter::operator++()
{
    if (m_done)
        return *this;

    if (--(*m_size))
        *m_data += *m_stride;
    else if (!m_iterNext(m_iter))
        m_done = true;
    return *this;
}

ArrayIter::operator bool () const
{
    return !m_done;
}

char * ArrayIter::operator * () const
{
    return *m_data;
}

} // namespace python
} // namespace pdal

