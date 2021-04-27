/******************************************************************************
* Copyright (c) 2021, Runette Software Ltd (www.runette.co.uk)
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

#include "PyMesh.hpp"

#include <numpy/arrayobject.h>


namespace pdal
{
namespace python
{

// Create new empty mesh
//
Mesh::Mesh() : m_mesh(nullptr)
{
    hasMesh = false;
    if (_import_array() < 0)
        throw pdal_error("Could not import numpy.core.multiarray.");
}

Mesh::~Mesh()
{
    if (m_mesh)
        Py_XDECREF((PyObject *)m_mesh);
}

// Update from a PointView
//
void Mesh::update(PointViewPtr view)
{
    npy_intp size;

    if (m_mesh)
        Py_XDECREF((PyObject *)m_mesh);
    m_mesh = nullptr;  // Just in case of an exception.

    TriangularMesh* mesh = view->mesh();
    if (mesh)
    {
        hasMesh = true;
        size = mesh->size();
    } else {
        hasMesh = false;
        size = 0;
    }

    PyObject *dtype_dict = (PyObject*)buildNumpyDescription(view);
    if (!dtype_dict)
        throw pdal_error("Unable to build numpy dtype "
                "description dictionary");

    PyArray_Descr *dtype = nullptr;
    if (PyArray_DescrConverter(dtype_dict, &dtype) == NPY_FAIL)
        throw pdal_error("Unable to build numpy dtype");
    Py_XDECREF(dtype_dict);

    // This is a 1 x size array.
    m_mesh = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
            1, &size, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

    for (PointId idx = 0; idx < size; idx++)
    {
        char* p = (char *)PyArray_GETPTR1(m_mesh, idx);
        const Triangle& t = (*mesh)[idx];
        uint32_t a = (uint32_t)t.m_a;
        std::memcpy(p, &a, 4);
        uint32_t b = (uint32_t)t.m_b;
        std::memcpy(p + 4, &b, 4);
        uint32_t c = (uint32_t)t.m_c;
        std::memcpy(p + 8, &c,  4);
    }
}



PyArrayObject *Mesh::getPythonArray() const
{
    return m_mesh;
}

PyObject* Mesh::buildNumpyDescription(PointViewPtr view) const
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
    PyObject* formats = PyList_New(3);
    PyObject* titles = PyList_New(3);

    PyList_SetItem(titles, 0, PyUnicode_FromString("A"));
    PyList_SetItem(formats, 0, PyUnicode_FromString("u4"));
    PyList_SetItem(titles, 1, PyUnicode_FromString("B"));
    PyList_SetItem(formats, 1, PyUnicode_FromString("u4"));
    PyList_SetItem(titles, 2, PyUnicode_FromString("C"));
    PyList_SetItem(formats, 2, PyUnicode_FromString("u4"));
   

    PyDict_SetItemString(dict, "names", titles);
    PyDict_SetItemString(dict, "formats", formats);

    return dict;
}

bool Mesh::rowMajor() const
{
    return m_rowMajor;
}

Mesh::Shape Mesh::shape() const
{
    return m_shape;
}


MeshIter& Mesh::iterator()
{
    MeshIter *it = new MeshIter(*this);
    m_iterators.push_back(std::unique_ptr<MeshIter>(it));
    return *it;
}

MeshIter::MeshIter(Mesh& mesh)
{
    m_iter = NpyIter_New(mesh.getPythonArray(),
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

MeshIter::~MeshIter()
{
    NpyIter_Deallocate(m_iter);
}

MeshIter& MeshIter::operator++()
{
    if (m_done)
        return *this;

    if (--(*m_size))
        *m_data += *m_stride;
    else if (!m_iterNext(m_iter))
        m_done = true;
    return *this;
}

MeshIter::operator bool () const
{
    return !m_done;
}

char * MeshIter::operator * () const
{
    return *m_data;
}


} // namespace python
} // namespace pdal

