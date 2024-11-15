cmake_minimum_required(VERSION 3.16.0)
project(pdal-python VERSION ${SKBUILD_PROJECT_VERSION}
                    DESCRIPTION "PDAL Python bindings"
                    HOMEPAGE_URL "https://github.com/PDAL/Python")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Python-finding settings
set(Python3_FIND_STRATEGY "LOCATION")
set(Python3_FIND_REGISTRY "LAST")
set(Python3_FIND_FRAMEWORK "LAST")

# Development vs. Development.Module
# https://cmake.org/cmake/help/latest/module/FindPython3.html?highlight=Development.Module
if (${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.0" AND NOT APPLE)
    set(DEVELOPMENT_COMPONENT "Development.Module")
else()
    set(DEVELOPMENT_COMPONENT "Development")
endif()

# find Python3
find_package(Python3 COMPONENTS Interpreter ${DEVELOPMENT_COMPONENT} NumPy REQUIRED)

# find PDAL. Require 2.1+
find_package(PDAL 2.7 REQUIRED)

# find PyBind11
find_package(pybind11 REQUIRED)

set(extension "libpdalpython")
pybind11_add_module(${extension} MODULE
    src/pdal/PyArray.cpp
    src/pdal/PyPipeline.cpp
    src/pdal/StreamableExecutor.cpp
    src/pdal/libpdalpython.cpp
)
target_include_directories(${extension} PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(${extension} PRIVATE ${PDAL_LIBRARIES})
install(TARGETS ${extension} LIBRARY DESTINATION "pdal")
