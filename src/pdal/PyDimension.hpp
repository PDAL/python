/******************************************************************************
* Copyright (c) 2018, Howard Butler (howard@hobu.co)
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

#include <pdal/PointView.hpp>
#include <pdal/Dimension.hpp>

#include <algorithm>
#include <vector>

typedef struct Dimension
{
    std::string name;
    std::string description;
    std::string type;
    size_t size;
    std::string units;
} Dimension;

inline std::vector<Dimension> getValidDimensions()
{
    std::vector<Dimension> output;

    int id = (int)pdal::Dimension::Id::Unknown + 1;

    while(1)
    {
        pdal::Dimension::Id pid = (pdal::Dimension::Id)id;
        std::string name(pdal::Dimension::name(pid));
        if (name.empty())
            break;

        pdal::Dimension::Type t = pdal::Dimension::defaultType(pid);

        Dimension d;
        d.name = name;
        d.description = pdal::Dimension::description(pid);
        d.size = pdal::Dimension::size(t);

        std::string kind("i");
        pdal::Dimension::BaseType b = pdal::Dimension::base(t);
        if (b == pdal::Dimension::BaseType::Unsigned)
            kind = "u";
        else if (b == pdal::Dimension::BaseType::Signed)
            kind = "i";
        else if (b == pdal::Dimension::BaseType::Floating)
            kind = "f";
        else
        {
            std::stringstream oss;
            oss << "unable to map kind '" << kind <<"' to PDAL dimension type";
            throw pdal::pdal_error(oss.str());
        }
        d.type = kind;


        output.push_back(d);
        id++;

    }

    return output;


}
