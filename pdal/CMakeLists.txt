find_package(pybind11 REQUIRED)
set(extension "libpdalpython")
pybind11_add_module(${extension} MODULE
    PyArray.cpp PyPipeline.cpp StreamableExecutor.cpp libpdalpython.cpp
)
target_include_directories(${extension} PRIVATE ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(${extension} PRIVATE ${PDAL_LIBRARIES})
install(TARGETS ${extension} LIBRARY DESTINATION "pdal")
