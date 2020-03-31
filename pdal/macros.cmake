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
    install(TARGETS ${${_name}}
        LIBRARY DESTINATION ${CMAKE_INSTALL_DIR}
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
