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
*     * Neither the name of Hobu, Inc. nor the
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

#include <numpy/ndarraytypes.h>

#include <pdal/PointView.hpp>
#include <pdal/io/MemoryViewReader.hpp>

#include <utility>

namespace pdal
{
namespace python
{

class ArrayIter;

class PDAL_DLL Array
{
public:
    using Shape = std::array<size_t, 3>;
    using Fields = std::vector<MemoryViewReader::Field>;

    // Create an array for reading data from PDAL.
    Array();

    // Create an array for writing data to PDAL.
    Array(PyArrayObject* array);

    ~Array();
    void update(PointViewPtr view);
    PyArrayObject *getPythonArray() const;
    bool rowMajor() const;
    Shape shape() const;
    const Fields& fields() const;
    ArrayIter& iterator();


private:
    inline PyObject* buildNumpyDescription(PointViewPtr view) const;


    PyArrayObject* m_array;
    Array& operator=(Array const& rhs);
    Fields m_fields;
    bool m_rowMajor;
    Shape m_shape {};
    std::vector<std::unique_ptr<ArrayIter>> m_iterators;
};

class ArrayIter
{
public:
    ArrayIter(const ArrayIter&) = delete;
    ArrayIter() = delete;

    ArrayIter(Array& array);
    ~ArrayIter();

    ArrayIter& operator++();
    operator bool () const;
    char *operator * () const;

private:
    NpyIter *m_iter;
    NpyIter_IterNextFunc *m_iterNext;
    char **m_data;
    npy_intp *m_size;
    npy_intp *m_stride;
    bool m_done;
};

} // namespace python
} // namespace pdal

