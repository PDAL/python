cmake_minimum_required(VERSION 3.11.0)
project(PDAL)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(WITH_TESTS "Enable tests" OFF)
set(CMAKE_BUILD_TYPE "Release")

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

add_subdirectory(pdal)

# Taken and adapted from PDAL's cmake macros.cmake

function(pdal_python_target_compile_settings target)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 11)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED TRUE)
    target_compile_definitions(${target} PRIVATE
        -DWIN32_LEAN_AND_MEAN)
    if (MSVC)
        # check for MSVC 8+
        if (NOT (MSVC_VERSION VERSION_LESS 1400))
            target_compile_definitions(${target} PRIVATE
                -D_CRT_SECURE_NO_DEPRECATE
                -D_CRT_SECURE_NO_WARNINGS
                -D_CRT_NONSTDC_NO_WARNING
                -D_SCL_SECURE_NO_WARNINGS
            )
            target_compile_options(${target} PRIVATE
                # Yes, we don't understand GCC pragmas
                /wd4068
                # Nitro makes use of Exception Specifications, which results in
                # numerous warnings when compiling in MSVC. We will ignore
                # them for now.
                /wd4290
                /wd4800
                # Windows warns about integer narrowing like crazy and it's
                # annoying.  In most cases the programmer knows what they're
                # doing.  A good static analysis tool would be better than
                # turning this warning off.
                /wd4267
                # Annoying warning about function hiding with virtual
                # inheritance.
                /wd4250
                # some templates don't return
#                /wd4716
                # unwind semantics
#                /wd4530
                # Standard C++-type exception handling.
                /EHsc
                )
        endif()

    endif()
endfunction()


###############################################################################
# Add a plugin target.
# _name The plugin name.
# ARGN :
#    FILES the source files for the plugin
#    LINK_WITH link plugin with libraries
#    INCLUDES header directories
#
# The "generate_dimension_hpp" ensures that Dimension.hpp is built before
#  attempting to build anything else in the "library".
#
# NOTE: _name is the name of a variable that will hold the plugin name
#    when the macro completes
macro(PDAL_PYTHON_ADD_PLUGIN _name _type _shortname)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs FILES LINK_WITH INCLUDES SYSTEM_INCLUDES COMPILE_OPTIONS)
    cmake_parse_arguments(PDAL_PYTHON_ADD_PLUGIN "${options}" "${oneValueArgs}"
        "${multiValueArgs}" ${ARGN})
    if(WIN32)
        set(WINSOCK_LIBRARY ws2_32)
        set(${_name} "libpdal_plugin_${_type}_${_shortname}")
    else()
        set(${_name} "pdal_plugin_${_type}_${_shortname}")
    endif()


    add_library(${${_name}} SHARED ${PDAL_PYTHON_ADD_PLUGIN_FILES})
    pdal_python_target_compile_settings(${${_name}})
    target_include_directories(${${_name}} PRIVATE
        ${PROJECT_BINARY_DIR}/include
        ${PDAL_INCLUDE_DIR}
        ${PDAL_PYTHON_ADD_PLUGIN_INCLUDES}
    )
    target_link_options(${${_name}} BEFORE PRIVATE ${PDAL_PYTHON_ADD_PLUGIN_COMPILE_OPTIONS})
    target_compile_definitions(${${_name}} PRIVATE
     PDAL_PYTHON_LIBRARY="${PYTHON_LIBRARY}" PDAL_DLL_EXPORT)
    target_compile_definitions(${${_name}} PRIVATE PDAL_DLL_EXPORT)
    if (PDAL_PYTHON_ADD_PLUGIN_SYSTEM_INCLUDES)
        target_include_directories(${${_name}} SYSTEM PRIVATE
            ${PDAL_PYTHON_ADD_PLUGIN_SYSTEM_INCLUDES})
    endif()
    target_link_libraries(${${_name}}
        PRIVATE
            ${PDAL_PYTHON_ADD_PLUGIN_LINK_WITH}
            ${WINSOCK_LIBRARY}
    )

    message(STATUS  "PROJECT_NAME: ${PROJECT_NAME}")
    install(TARGETS ${${_name}}
        LIBRARY DESTINATION "pdal"
        )
    if (APPLE)
        set_target_properties(${${_name}} PROPERTIES
            INSTALL_NAME_DIR "@rpath")
    endif()
endmacro(PDAL_PYTHON_ADD_PLUGIN)


macro(PDAL_PYTHON_ADD_TEST _name)
    set(options)
    set(oneValueArgs)
    set(multiValueArgs FILES LINK_WITH INCLUDES SYSTEM_INCLUDES)
    cmake_parse_arguments(PDAL_PYTHON_ADD_TEST "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if (WIN32)
        set(WINSOCK_LIBRARY ws2_32)
    endif()
    add_executable(${_name} ${PDAL_PYTHON_ADD_TEST_FILES})

    pdal_python_target_compile_settings(${_name})
    target_include_directories(${_name} PRIVATE
        ${PDAL_PYTHON_ADD_TEST_INCLUDES})
    if (PDAL_PYTHON_ADD_TEST_SYSTEM_INCLUDES)
        target_include_directories(${_name} SYSTEM PRIVATE
            ${PDAL_PYTHON_ADD_TEST_SYSTEM_INCLUDES})
    endif()
    set_property(TARGET ${_name} PROPERTY FOLDER "Tests")
    target_link_libraries(${_name}
        PRIVATE
            ${PDAL_PYTHON_ADD_TEST_LINK_WITH}
            gtest
            ${WINSOCK_LIBRARY}
    )
    target_compile_definitions(${_name} PRIVATE
        PDAL_PYTHON_LIBRARY="${PYTHON_LIBRARY}")
    add_test(NAME ${_name}
        COMMAND
            "${PROJECT_BINARY_DIR}/bin/${_name}"
        WORKING_DIRECTORY
            "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/..")
    # Ensure plugins are loaded from build dir
    # https://github.com/PDAL/PDAL/issues/840
#    if (WIN32)
#        set_property(TEST ${_name} PROPERTY ENVIRONMENT
#            "PDAL_DRIVER_PATH=${PROJECT_BINARY_DIR}/bin")
#    else()
#        set_property(TEST ${_name} PROPERTY ENVIRONMENT
#            "PDAL_DRIVER_PATH=${PROJECT_BINARY_DIR}/lib")
#    endif()
endmacro(PDAL_PYTHON_ADD_TEST)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # For newer versions of python (3.8+), C extensions don't link against
    # libpython and instead get symbol definitions from the python interpreter
    # executable. PDAL plugins need to link against libpython, but if a plugin
    # is loaded inside a python process, it must resolve symbols from the python
    # executable instead of libpython. Using flat namespace allows that.
    set(PYTHON_LINK_LIBRARY ${PYTHON_LINK_LIBRARY} -Wl,-flat_namespace)
endif()

PDAL_PYTHON_ADD_PLUGIN(numpy_reader reader numpy
    FILES
        ./pdal/io/NumpyReader.cpp
        ./pdal/io/NumpyReader.hpp
        ./pdal/plang/Invocation.cpp
        ./pdal/plang/Environment.cpp
        ./pdal/plang/Redirector.cpp
        ./pdal/plang/Script.cpp
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

PDAL_PYTHON_ADD_PLUGIN(python_filter filter python
    FILES
        ./pdal/filters/PythonFilter.cpp
        ./pdal/filters/PythonFilter.hpp
        ./pdal/plang/Invocation.cpp
        ./pdal/plang/Environment.cpp
        ./pdal/plang/Redirector.cpp
        ./pdal/plang/Script.cpp
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


if (WITH_TESTS)
    enable_testing()
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(pdal/test/gtest)
    enable_testing()
    include_directories(pdal/test/gtest/include .. ${CMAKE_CURRENT_BINARY_DIR})




    PDAL_PYTHON_ADD_TEST(pdal_io_numpy_test
        FILES
            ./pdal/test/NumpyReaderTest.cpp
            ./pdal/test/Support.cpp
            ./pdal/plang/Invocation.cpp
            ./pdal/plang/Environment.cpp
            ./pdal/plang/Redirector.cpp
            ./pdal/plang/Script.cpp
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
            ./pdal/test/PythonFilterTest.cpp
            ./pdal/test/Support.cpp
            ./pdal/plang/Invocation.cpp
            ./pdal/plang/Environment.cpp
            ./pdal/plang/Redirector.cpp
            ./pdal/plang/Script.cpp
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
endif (WITH_TESTS)
