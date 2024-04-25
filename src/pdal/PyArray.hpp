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

#include <pdal/PointView.hpp>

#define NPY_TARGET_VERSION NPY_1_22_API_VERSION
#define NPY_NO_DEPRECATED_API NPY_1_22_API_VERSION

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL PDAL_ARRAY_API

#include <pdal/io/MemoryViewReader.hpp>

#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>

#include <vector>
#include <memory>

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

    Array(PyArrayObject* array);
    ~Array();

    Array(Array&& a) = default;
    Array& operator=(Array&& a) = default;

    Array(const Array&) = delete;
    Array() = delete;

    bool rowMajor() const { return m_rowMajor; };
    Shape shape() const { return m_shape; }
    const Fields& fields() const { return m_fields; };
    ArrayIter& iterator();

private:
    PyArrayObject* m_array;
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

    ArrayIter(PyArrayObject*);
    ~ArrayIter();

    ArrayIter& operator++();
    operator bool () const { return !m_done; }
    char* operator*() const { return *m_data; }

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

