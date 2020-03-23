cmake_minimum_required(VERSION 3.11.0)
project(pdal)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")

find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(PDAL REQUIRED)
find_package(NumPy 1.13 REQUIRED)

if(SKBUILD)
  find_package(PythonExtensions REQUIRED)
  find_package(Cython REQUIRED)
  message(STATUS "The project is built using scikit-build")
endif()


# macros for creating targets
include(${PROJECT_SOURCE_DIR}/macros.cmake NO_POLICY_SCOPE)

set(EXTENSION_SRC
    PyArray.cpp
    PyArray.hpp
    PyDimension.hpp
    PyPipeline.cpp
    PyPipeline.hpp)

set(extension "libpdalpython")
add_cython_target(${extension} libpdalpython.pyx CXX PY3)

add_library(${extension} MODULE ${EXTENSION_SRC} libpdalpython)
target_include_directories( ${extension}
    PRIVATE
    .
    ${PDAL_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS})

target_link_libraries(${extension} ${PDAL_LIBRARIES})
python_extension_module(${extension})

if (MSVC)
set(reader "libpdal_plugin_reader_numpy")
else()
set(reader "pdal_plugin_reader_numpy")
endif()

set(PLANG ./plang/Invocation.cpp
          ./plang/Environment.cpp
          ./plang/Redirector.cpp
          ./plang/Script.cpp)
set(READER_SRC
    ./io/NumpyReader.cpp
    ./io/NumpyReader.hpp
    ${PLANG})

install(TARGETS ${extension} LIBRARY DESTINATION ${PROJECT_NAME})

PDAL_PYTHON_ADD_PLUGIN(numpy_reader reader numpy
    FILES  
    ./io/NumpyReader.cpp
    ./io/NumpyReader.hpp
    ./plang/Invocation.cpp
          ./plang/Environment.cpp
          ./plang/Redirector.cpp
          ./plang/Script.cpp
    LINK_WITH 
        ${PDAL_LIBRARIES} 
        ${PYTHON_LIBRARY}
    SYSTEM_INCLUDES
        ${PDAL_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
    )

PDAL_PYTHON_ADD_PLUGIN(python_filter reader python
    FILES  
    ./filters/PythonFilter.cpp
    ./filters/PythonFilter.hpp
    ./plang/Invocation.cpp
          ./plang/Environment.cpp
          ./plang/Redirector.cpp
          ./plang/Script.cpp
    LINK_WITH 
        ${PDAL_LIBRARIES} 
        ${PYTHON_LIBRARY}
    SYSTEM_INCLUDES
        ${PDAL_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
    )
