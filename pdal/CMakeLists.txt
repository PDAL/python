cmake_minimum_required(VERSION 3.11.0)
project(pdal)

# macros for creating targets
include(${PROJECT_SOURCE_DIR}/macros.cmake NO_POLICY_SCOPE)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_BUILD_TYPE "Release")

enable_testing()

# Python-finding settings
set(Python3_FIND_STRATEGY "LOCATION")
set(Python3_FIND_REGISTRY "LAST")
set(Python3_FIND_FRAMEWORK "LAST")
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

# find PDAL. Require 2.1+
find_package(PDAL 2.1 REQUIRED)

if(SKBUILD)
  find_package(PythonExtensions REQUIRED)
  find_package(Cython REQUIRED)
  message(STATUS "The project is built using scikit-build")
endif()

execute_process(
    COMMAND
    ${Python3_EXECUTABLE} -c "from distutils import sysconfig; print(sysconfig.get_config_var('Py_ENABLE_SHARED'))"
    OUTPUT_VARIABLE Py_ENABLE_SHARED
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

#execute_process(
#    COMMAND
#    ${Python3_EXECUTABLE} -c "from distutils import sysconfig; print(sysconfig.get_config_var('LDSHARED').split(' ', 1)[1])"
#    OUTPUT_VARIABLE PYTHON_LDSHARED
#    OUTPUT_STRIP_TRAILING_WHITESPACE
#)

#if (NOT Py_ENABLE_SHARED)
#    message(STATUS "Python ${Python3_EXECUTABLE} is statically linked")
#    if (APPLE)
#        # conda gives us -bundle, which isn't valid
#        message(STATUS "Removing extra -bundle argument from sysconfig.get_config_var('LDSHARED')")
#        string(REPLACE  "-bundle" "" PYTHON_LDSHARED "${PYTHON_LDSHARED}")
#        string(STRIP  ${PYTHON_LDSHARED} PYTHON_LDSHARED)
#    endif()
#  #  set(Python3_LIBRARIES ${PYTHON_LDSHARED})
#    message(STATUS "Setting Python3_LIBRARIES to '${Python3_LIBRARIES}' due to static Python")
#endif()

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
        ${Python3_LIBRARIES}
        ${CMAKE_DL_LIBS}
    SYSTEM_INCLUDES
        ${PDAL_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
    COMPILE_OPTIONS
        ${PYTHON_LINK_LIBRARY}
    )


set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(test/gtest)
enable_testing()
include_directories(test/gtest/include .. ${CMAKE_CURRENT_BINARY_DIR})


PDAL_PYTHON_ADD_PLUGIN(python_filter filter python
    FILES
        ./filters/PythonFilter.cpp
        ./filters/PythonFilter.hpp
        ./plang/Invocation.cpp
        ./plang/Environment.cpp
        ./plang/Redirector.cpp
        ./plang/Script.cpp
    LINK_WITH
        ${PDAL_LIBRARIES}
        ${Python3_LIBRARIES}
        ${CMAKE_DL_LIBS}
    SYSTEM_INCLUDES
        ${PDAL_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
    COMPILE_OPTIONS
        ${PYTHON_LINK_LIBRARY}
    )

PDAL_PYTHON_ADD_TEST(pdal_io_numpy_test
    FILES
        ./test/NumpyReaderTest.cpp
        ./test/Support.cpp
        ./plang/Invocation.cpp
        ./plang/Environment.cpp
        ./plang/Redirector.cpp
        ./plang/Script.cpp
    LINK_WITH
        ${numpy_reader}
        ${Python3_LIBRARIES}
        ${PDAL_LIBRARIES}
        ${CMAKE_DL_LIBS}
    SYSTEM_INCLUDES
        ${PDAL_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)

PDAL_PYTHON_ADD_TEST(pdal_filters_python_test
    FILES
        ./test/PythonFilterTest.cpp
        ./test/Support.cpp
        ./plang/Invocation.cpp
        ./plang/Environment.cpp
        ./plang/Redirector.cpp
        ./plang/Script.cpp
    LINK_WITH
        ${python_filter}
        ${Python3_LIBRARIES}
        ${PDAL_LIBRARIES}
        ${CMAKE_DL_LIBS}
    SYSTEM_INCLUDES
        ${PDAL_INCLUDE_DIRS}
        ${Python3_INCLUDE_DIRS}
        ${Python3_NumPy_INCLUDE_DIRS}
)
