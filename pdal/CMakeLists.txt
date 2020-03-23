cmake_minimum_required(VERSION 3.11.0)
project(pdal)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(Python3 COMPONENTS Interpreter Development NumPy)
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)

find_package(PDAL REQUIRED)
find_package(NumPy 1.13 REQUIRED)

if(SKBUILD)
  message(STATUS "The project is built using scikit-build")
endif()

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

add_library(${reader} SHARED ${READER_SRC} )
target_include_directories( ${reader}
    PRIVATE
    .
    ${PDAL_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS})

target_link_libraries(${reader} ${PDAL_LIBRARIES} ${PYTHON_LIBRARY})
target_compile_definitions(${reader} PRIVATE
    PDAL_PYTHON_LIBRARY="${PYTHON_LIBRARY}" PDAL_DLL_EXPORT)
if (MSVC)
  target_link_libraries(${reader} ${PDAL_LIBRARIES} ws2_32)
endif()

if (MSVC)
set(filter "libpdal_plugin_filter_python")
else()
set(filter "pdal_plugin_filter_python")
endif()

set(FILTER_SRC
    ./filters/PythonFilter.cpp
    ./filters/PythonFilter.hpp
    ${PLANG})

add_library(${filter} SHARED ${FILTER_SRC} )
target_include_directories( ${filter}
    PRIVATE
    .
    ${PDAL_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS})

target_link_libraries(${filter} ${PDAL_LIBRARIES} ${PYTHON_LIBRARY})
if (MSVC)
  target_link_libraries(${filter} ${PDAL_LIBRARIES} ws2_32)
endif()

target_compile_definitions(${filter} PRIVATE
    PDAL_PYTHON_LIBRARY="${PYTHON_LIBRARY}" PDAL_DLL_EXPORT)


install(TARGETS ${extension} LIBRARY DESTINATION ${PROJECT_NAME})
install(TARGETS ${reader} LIBRARY DESTINATION ${CMAKE_INSTALL_DIR})
install(TARGETS ${filter} LIBRARY DESTINATION ${CMAKE_INSTALL_DIR})
