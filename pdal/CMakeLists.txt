set(EXTENSION_SRC
    PyArray.cpp
    PyArray.hpp
    PyMesh.cpp
    PyMesh.hpp
    PyDimension.hpp
    PyPipeline.cpp
    PyPipeline.hpp)

set(extension "libpdalpython")
add_cython_target(${extension} "libpdalpython.pyx" CXX PY3)

add_library(${extension} MODULE ${EXTENSION_SRC} libpdalpython)
target_include_directories( ${extension}
    PRIVATE
    .
    ${PDAL_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS})

target_link_libraries(${extension} ${PDAL_LIBRARIES})
python_extension_module(${extension})

install(TARGETS ${extension} LIBRARY DESTINATION "pdal")


