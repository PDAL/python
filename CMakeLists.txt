cmake_minimum_required(VERSION 3.11.0)
project(pdal-python VERSION ${SKBUILD_PROJECT_VERSION}
                    DESCRIPTION "PDAL Python bindings"
                    HOMEPAGE_URL "https://github.com/PDAL/Python")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")

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
find_package(PDAL 2.6 REQUIRED)

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

#
# This places the necessary files into the binary directory to allow running
# without installation as long as PYTHONPATH is set to the binary (build) directory.
# Note that if you add ".py" source files, you'll have to re-run cmake to get the
# new file copied to the binary directory.
#
set_target_properties(${extension}
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/pdal
)
file(GLOB PDAL_PYTHON_SOURCES src/pdal/*.py)
foreach(PDAL_PYTHON_FILE ${PDAL_PYTHON_SOURCES})
    configure_file(${PDAL_PYTHON_FILE} pdal COPYONLY)
endforeach()

install(TARGETS ${extension} LIBRARY DESTINATION "pdal")
